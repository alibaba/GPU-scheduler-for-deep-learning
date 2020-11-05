/*
Copyright 2020 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package antman

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	"k8s.io/kubernetes/pkg/scheduler/util"
	ctrl "sigs.k8s.io/controller-runtime"
	runtimeClient "sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	KubeDLPrefix      = "kubedl.io"
	ResourceNvidiaGPU = v1.ResourceName("nvidia.com/gpu")
	// AnnotationGPUVisibleDevices is the visible gpu devices in view of process.
	AnnotationGPUVisibleDevices = "gpus." + KubeDLPrefix + "/visible-devices"

	// for antman
	AntManPrefix = "alibaba.pai.antman"
	// for DL jobs to report DL framework information to local coordinator.
	AnnotationGpuStatusFile = AntManPrefix + "/gpu-status-file"
	// for local coordinator to control the GPU usage of a DL job.
	AnnotationGpuConfigFile = AntManPrefix + "/gpu-config-file"
	// for local coordinator to access file
	LocalCoordinatorCommFolder = "/tmp/antman/"
)

// Antman is a plugin that implements the mechanism of gang scheduling.
type Antman struct {
	frameworkHandle framework.FrameworkHandle
	podLister       corelisters.PodLister
	// key is <namespace>/<Pod name> and value is *PodInfo.
	podInfos sync.Map
	// clock is used to get the current time.
	clock util.Clock
	// etcdWrapper is used to handle etcd db
	etcdWrapper *EtcdWrapper
	runtimeCli  runtimeClient.Client
}

// PodInfo is a wrapper to a PodGroup with additional information.
// A PodGroup's priority, timestamp and minAvailable are set according to
// the values of the PodGroup's first pod that is added to the scheduling queue.
type PodInfo struct {
	// key is a unique PodGroup ID and currently implemented as <namespace>/<PodGroup name>.
	key string
	// podType is a string among "auto", "true", "false", which indicates the type of this pod.
	podType string
	// name is the PodGroup name and defined through a Pod label.
	// The PodGroup name of a regular pod is empty.
	// name string
	// priority is the priority of pods in a PodGroup.
	// All pods in a PodGroup should have the same priority.
	// priority int32
	// timestamp stores the initialization timestamp of a PodGroup.
	// timestamp time.Time
	// minAvailable is the minimum number of pods to be co-scheduled in a PodGroup.
	// All pods in a PodGroup should have the same minAvailable.
	// minAvailable int
	// deletionTimestamp stores the timestamp when the PodGroup marked as expired.
	// deletionTimestamp *time.Time
}

// var _ framework.QueueSortPlugin = &Antman{}
var _ framework.PreFilterPlugin = &Antman{}
var _ framework.ScorePlugin = &Antman{}
var _ framework.ReservePlugin = &Antman{}

var scheme = runtime.NewScheme()

func init() {
	_ = clientgoscheme.AddToScheme(scheme)
}

const (
	// Name is the name of the plugin used in Registry and configurations.
	Name = "Antman"
	// OpportunisticPod is the label name for antman to identify the type of a Pod.
	OpportunisticPod = "opportunistic"
)

// Name returns name of the plugin. It is used in logs, etc.
func (am *Antman) Name() string {
	return Name
}

// New initializes a new plugin and returns it.
func New(config *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	podLister := handle.SharedInformerFactory().Core().V1().Pods().Lister()
	ew := NewEtcdWrapper()
	conf := ctrl.GetConfigOrDie()
	client, err := runtimeClient.New(conf, runtimeClient.Options{
		Scheme: scheme,
	})
	if err != nil {
		return nil, err
	}
	am := &Antman{
		frameworkHandle: handle,
		podLister:       podLister,
		clock:           util.RealClock{},
		etcdWrapper:     ew,
		runtimeCli:      client,
	}
	podInformer := handle.SharedInformerFactory().Core().V1().Pods().Informer()
	podInformer.AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Pod:
					return true
				case cache.DeletedFinalStateUnknown:
					if _, ok := t.Obj.(*v1.Pod); ok {
						return true
					}
					return false
				default:
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				UpdateFunc: am.podUpdate,
			},
		},
	)
	return am, nil
}

// getOrCreatePodGroupInfo returns the existing PodGroup in PodGroupInfos if present.
// Otherwise, it creates a PodGroup and returns the value, It stores
// the created PodGroup in PodGroupInfo if the pod defines a  PodGroup and its
func (am *Antman) getOrCreatePodInfo(pod *v1.Pod, ts time.Time) *PodInfo {
	opportunistic, _ := GetPodTypeLabels(pod)

	var podKey string
	if len(opportunistic) > 0 {
		podKey = fmt.Sprintf("%v/%v", pod.Namespace, pod.Name)
	}

	// If it is a Pod and present in PodInfos, return it.
	if len(podKey) != 0 {
		value, exist := am.podInfos.Load(podKey)
		if exist {
			podInfo := value.(*PodInfo)
			return podInfo
		}
	}

	// If the Pod is not present in PodInfos or the pod is a regular pod,
	// create a PodGroup for the Pod and store it in PodGroupInfos if it's not a regular pod.
	podInfo := &PodInfo{
		key:     podKey,
		podType: opportunistic,
	}

	// If it's not a regular Pod, store the PodGroup in PodGroupInfos
	if len(podKey) > 0 {
		am.podInfos.Store(podKey, podInfo)
	}
	return podInfo
}

// PreFilter performs the following validations.
// 1. Validate if minAvailables and priorities of all the pods in a PodGroup are the same.
// 2. Validate if the total number of pods belonging to the same `PodGroup` is less than `minAvailable`.
//    If so, the scheduling process will be interrupted directly to avoid the partial Pods and hold the system resources
//    until a timeout. It will reduce the overall scheduling time for the whole group.
func (am *Antman) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) *framework.Status {
	podInfo := am.getOrCreatePodInfo(pod, time.Now())
	podKey := podInfo.key
	if len(podKey) == 0 {
		// normal pod, not opportunistic pod in antman
		klog.Infof("[%v] normal pod", pod.Name)
		return framework.NewStatus(framework.Success, "")
	}
	opportunistic := podInfo.podType
	// TODO(wencong.xwc) we need to add the logic to handle auto later
	if opportunistic != "auto" && opportunistic != "true" && opportunistic != "false" {
		klog.Infof("Pod %v has a different podType (%v) as the expected value among auto, true, false", pod.Name, opportunistic)
		return framework.NewStatus(framework.Unschedulable, "podType do not match AntMan's assumption")
	}
	klog.Infof("[%v] opportunistic : %v", pod.Name, opportunistic)
	return framework.NewStatus(framework.Success, "")
}

// PreFilterExtensions returns nil.
func (am *Antman) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

// This function returns the average gpu util and the peak gpu me in the last minutes for all gpus in that node
func (am *Antman) GetGpusInfoFromEtcd(nodeName string) (gpuUtils []float64, gpuMems []float64) {
	klog.Infof("in GetGpusInfoFromEtcd for nodeName : %v", nodeName)
	var utils []float64
	var mems []float64
	const (
		PeriodInSeconds = 30
	)
	var count, nowIndex, nowTime int
	var err error
	// TODO # GPU from k8s api
	nodeInfo, err := am.frameworkHandle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	num_gpus := int(nodeInfo.AllocatableResource().ScalarResources["nvidia.com/gpu"])
	for gpu_id := 0; gpu_id < num_gpus; gpu_id++ {
		// the number of gpu statistics maintained
		countKey := nodeName + "/" + strconv.Itoa(gpu_id) + "/cnt"
		countStr := am.etcdWrapper.ReadEtcd(&countKey)
		if count, err = strconv.Atoi(*countStr); err == nil {
			klog.Infof("[%v] count: %v", gpu_id, count)
		}

		// the index of the last statistic
		nowIndexKey := nodeName + "/" + strconv.Itoa(gpu_id) + "/nowIndex"
		nowIndexStr := am.etcdWrapper.ReadEtcd(&nowIndexKey)
		if nowIndex, err = strconv.Atoi(*nowIndexStr); err == nil {
			klog.Infof("[%v] nowIndex: %v", gpu_id, nowIndex)
		}

		// the timestamp of the last index
		nowTimeKey := nodeName + "/" + strconv.Itoa(gpu_id) + "/nowTime"
		nowTimeStr := am.etcdWrapper.ReadEtcd(&nowTimeKey)
		if nowTime, err = strconv.Atoi(*nowTimeStr); err == nil {
			klog.Infof("[%v] nowTime: %v", gpu_id, nowTime)
		}

		// the statistics are stored in [0, count) using a Circular Queue
		// make sure that PeriodInSeconds is smaller than count
		if PeriodInSeconds > count {
			klog.Errorf("PeriodInSeconds[%v] should smaller than count[%v]", PeriodInSeconds, count)
		}

		// ave gpu util in PeriodInSeconds
		// peak gpu mem in PeriodInSeconds
		aveUtil := 0.0
		peakMem := 0.0
		for index := 0; index < PeriodInSeconds; index++ {
			i := (index - (PeriodInSeconds - nowIndex) + count) % count
			utilKey := nodeName + "/" + strconv.Itoa(gpu_id) + "/util" + "/" + strconv.Itoa(i)
			memKey := nodeName + "/" + strconv.Itoa(gpu_id) + "/mem" + "/" + strconv.Itoa(i)
			gpuUtilStr := am.etcdWrapper.ReadEtcd(&utilKey)
			gpuMemStr := am.etcdWrapper.ReadEtcd(&memKey)
			if gpuUtil, err := strconv.ParseFloat(*gpuUtilStr, 64); err == nil {
				aveUtil += gpuUtil
			}
			if gpuMem, err := strconv.ParseFloat(*gpuMemStr, 64); err == nil {
				peakMem = math.Max(peakMem, gpuMem)
			}
		}
		aveUtil = aveUtil / PeriodInSeconds

		utils = append(utils, aveUtil)
		mems = append(mems, peakMem)
	}
	return utils, mems
}

type MyString string

func (ms *MyString) Clone() framework.StateData {
	c := *ms
	return &c
}

// Score invoked at the score extension point.
func (am *Antman) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	klog.Infof("in Score for podName : %v", pod.Name)
	nodeInfo, err := am.frameworkHandle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	podInfo := am.getOrCreatePodInfo(pod, time.Now())
	podKey := podInfo.key
	if len(podKey) == 0 {
		// normal pod, not opportunistic pod in antman
		// return 100 for every node to bypass the scoring
		return int64(100), nil
	}

	// note that, because there are multiple score plugins in k8s
	// it is not the final score for this node
	return am.GpuScore(state, pod, nodeInfo)
}

// ScoreExtensions of the Score plugin.
func (am *Antman) ScoreExtensions() framework.ScoreExtensions {
	return am
}

// NormalizeScore invoked after scoring all nodes.
func (am *Antman) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	klog.Infof("in NormalizeScore for podName : %v", pod.Name)
	// Find highest and lowest scores.
	var highest int64 = -math.MaxInt64
	var lowest int64 = math.MaxInt64
	for _, nodeScore := range scores {
		if nodeScore.Score > highest {
			highest = nodeScore.Score
		}
		if nodeScore.Score < lowest {
			lowest = nodeScore.Score
		}
	}

	// Transform the highest to lowest score range to fit the framework's min to max node score range.
	oldRange := highest - lowest
	newRange := framework.MaxNodeScore - framework.MinNodeScore
	for i, nodeScore := range scores {
		if oldRange == 0 {
			scores[i].Score = framework.MinNodeScore
		} else {
			scores[i].Score = ((nodeScore.Score - lowest) * newRange / oldRange) + framework.MinNodeScore
		}
	}

	return nil
}

func (am *Antman) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	klog.Infof("in Reserve for podName : %v", pod.Name)

	if pod.ObjectMeta.Labels["opportunistic"] == "true" {
		// for opportunistic pods: select the most-free one
		// Note: we currently only support one gpu per pod
		gpuInfos := am.GetNodeGpuInfo(nodeName)
		var limitedUtil, limitedMem float64
		limitedUtil = 80
		limitedMem = 16 * 1024 * 1024 * 1024
		bestIndex := -1
		var minUtil float64
		minUtil = 101
		for _, gpu := range *gpuInfos {
			klog.Infof("[%v]: [%v, %v]", gpu.index, gpu.usedUtil, gpu.usedMem)
			if gpu.usedUtil < limitedUtil && gpu.usedMem < limitedMem {
				if minUtil > gpu.usedUtil {
					minUtil = gpu.usedUtil
					bestIndex = gpu.index
				}
			}
		}
		if bestIndex != -1 {
			// TODO
			// assign env variable: NVIDIA_VISIBLE_DEVICE
			podCopy := pod.DeepCopy()
			if podCopy.Annotations == nil {
				podCopy.Annotations = make(map[string]string)
			}

			gpuDevices := strconv.Itoa(bestIndex)
			podCopy.Annotations[AnnotationGPUVisibleDevices] = gpuDevices
			podCopy.Annotations[AnnotationGpuConfigFile] = LocalCoordinatorCommFolder + podCopy.ObjectMeta.Name + "_sche2pai.json"
			podCopy.Annotations[AnnotationGpuStatusFile] = LocalCoordinatorCommFolder + podCopy.ObjectMeta.Name + "_pai2sche.json"

			// Patch annotations to pod
			updateErr := am.runtimeCli.Patch(ctx, podCopy, runtimeClient.MergeFrom(pod))

			if updateErr == nil {
				klog.Infof("Pod %s patched successfully.", pod.GetName())
			} else {
				klog.Errorf("Pod %s patched failed : %v", pod.GetName(), updateErr)
			}

			am.NotifyLocalCoordinator(pod, nodeName, gpuDevices, "opportunistic")
		} else {
			// no qualified gpu found
			// TODO(wencong.xwc): need to rollback the whole scheduling decision, put the pod back to un-schedule
			return nil
		}
	} else {
		// allocate specific gpus to GPU pods
		freeGpus := am.GetNodeFreeGpus(nodeName)
		// random shuffle
		for i, _ := range freeGpus {
			j := rand.Intn(i + 1)
			freeGpus[i], freeGpus[j] = freeGpus[j], freeGpus[i]
		}

		for i := range pod.Spec.Containers {
			isWithGpu := false
			requestGpus := 0
			if val, ok := pod.Spec.Containers[i].Resources.Limits[ResourceNvidiaGPU]; ok {
				isWithGpu = true
				tmp, _ := val.AsInt64()
				requestGpus = int(tmp)
			}
			if val, ok := pod.Spec.Containers[i].Resources.Requests[ResourceNvidiaGPU]; ok {
				isWithGpu = true
				tmp, _ := val.AsInt64()
				requestGpus = int(tmp)
			}
			if isWithGpu {
				if requestGpus > len(freeGpus) {
					klog.Errorf("requestGpus should be larger than freeGpus")
					return nil
				}
				gpuDevices := strconv.Itoa(freeGpus[0])
				for i := 1; i < requestGpus; i++ {
					gpuDevices += "," + strconv.Itoa(freeGpus[i])
				}

				podCopy := pod.DeepCopy()
				if podCopy.Annotations == nil {
					podCopy.Annotations = make(map[string]string)
				}

				podCopy.Annotations[AnnotationGPUVisibleDevices] = gpuDevices
				podCopy.Annotations[AnnotationGpuConfigFile] = LocalCoordinatorCommFolder + podCopy.ObjectMeta.Name + "_sche2pai.json"
				podCopy.Annotations[AnnotationGpuStatusFile] = LocalCoordinatorCommFolder + podCopy.ObjectMeta.Name + "_pai2sche.json"

				// Patch annotations to pod
				updateErr := am.runtimeCli.Patch(ctx, podCopy, runtimeClient.MergeFrom(pod))

				if updateErr == nil {
					klog.Infof("Pod %s patched successfully.", pod.GetName())
				} else {
					klog.Errorf("Pod %s patched failed : %v", pod.GetName(), updateErr)
				}

				am.SetNodeFreeGpus(nodeName, freeGpus[requestGpus:])

				am.NotifyLocalCoordinator(pod, nodeName, gpuDevices, "resource-guarantee")
				break
			}
		}
	}
	return nil
}

func (am *Antman) NotifyLocalCoordinator(pod *v1.Pod, nodeName string, gpuDevices string, workerType string) {
	// notify local coordinator
	workerTypeKey := pod.ObjectMeta.Name + "/type"
	am.etcdWrapper.WriteEtcd(&workerTypeKey, &workerType)

	workerGpusKey := nodeName + "/" + pod.ObjectMeta.Name + "/gpus"
	am.etcdWrapper.WriteEtcd(&workerGpusKey, &gpuDevices)

	am.RegisterPods(nodeName, pod)
}

func (am *Antman) RegisterPods(nodeName string, pod *v1.Pod) {
	podsKey := nodeName + "/pods"
	activePodsPtr := am.etcdWrapper.ReadEtcd(&podsKey)
	activePods := ""
	if activePodsPtr == nil {
		// no pods in this node
		activePods = pod.ObjectMeta.Name
	} else {
		activePods = (*activePodsPtr) + "," + pod.ObjectMeta.Name
	}
	am.etcdWrapper.WriteEtcd(&podsKey, &activePods)
}

func (am *Antman) UnregisterPods(nodeName string, pod *v1.Pod) {
	podsKey := nodeName + "/pods"
	activePodsPtr := am.etcdWrapper.ReadEtcd(&podsKey)
	if activePodsPtr == nil {
		// no pods in this node
		klog.Errorf("UnregisterPods failed, no pods found")
		return
	} else {
		activePodsSlices := strings.Split(*activePodsPtr, ",")
		idx := -1
		for i, val := range activePodsSlices {
			if val == pod.ObjectMeta.Name {
				idx = i
				break
			}
		}
		newActivePodsSlices := append(activePodsSlices[:idx], activePodsSlices[idx+1:]...)
		activePods := strings.Join(newActivePodsSlices, ",")
		am.etcdWrapper.WriteEtcd(&podsKey, &activePods)
	}
}

// Unreserve rejects all other Pods in the PodGroup when one of the pods in the group times out.
func (am *Antman) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
	klog.Infof("in Unreserve for podName : %v", pod.Name)
	return
}

// GetPodTypeLabels checks if the pod is an opportunistic pod. If so, it will return the
// value of antman job type. If not, it will return "".
func GetPodTypeLabels(pod *v1.Pod) (string, error) {
	opportunistic, exist := pod.Labels[OpportunisticPod]
	if !exist || len(opportunistic) == 0 {
		klog.Infof("Pod %v/%v is a normal pod", pod.Namespace, pod.Name)
		return "", nil
	}
	klog.Infof("Pod %v/%v is a pod with opportunistic label set as: %v", pod.Namespace, pod.Name, opportunistic)
	return opportunistic, nil
}

type GpuInfo struct {
	index    int
	usedUtil float64
	usedMem  float64
}

func (am *Antman) GetNodeGpuInfo(nodeName string) *[]GpuInfo {
	gpuUtils, gpuMems := am.GetGpusInfoFromEtcd(nodeName)
	gpuInfos := []GpuInfo{}
	for i, util := range gpuUtils {
		mem := gpuMems[i]
		gpuInfos = append(gpuInfos, GpuInfo{
			index:    i,
			usedUtil: util,
			usedMem:  mem,
		})
	}
	return &gpuInfos
}

func (am *Antman) GetNodeFreeGpus(nodeName string) []int {
	gpuStatusKey := nodeName + "/gpuStatus"
	gpuStatusStr := am.etcdWrapper.ReadEtcd(&gpuStatusKey)
	var freeGpus []int
	if val, err := strconv.Atoi(*gpuStatusStr); err == nil {
		klog.Infof("[%v] gpuStatus: %08b", nodeName, val)
		index := 0
		for val != 0 {
			if val&1 == 1 {
				freeGpus = append(freeGpus, index)
			}
			index++
			val >>= 1
		}
	}
	return freeGpus
}

func (am *Antman) SetNodeFreeGpus(nodeName string, freeGpus []int) {
	gpuStatusVal := 0
	for _, gpuId := range freeGpus {
		gpuStatusVal |= 1 << gpuId
	}
	gpuStatusKey := nodeName + "/gpuStatus"
	valStr := strconv.Itoa(gpuStatusVal)
	klog.Infof("[%v] update gpuStatus to %08b", nodeName, gpuStatusVal)
	am.etcdWrapper.WriteEtcd(&gpuStatusKey, &valStr)
	return
}

func (am *Antman) GpuScore(state *framework.CycleState, pod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) (int64, *framework.Status) {
	node := nodeInfo.Node()
	if node == nil {
		return 0, framework.NewStatus(framework.Error, "node not found")
	}

	// bypass normal pods
	opportunistic, _ := GetPodTypeLabels(pod)
	if len(opportunistic) == 0 {
		// all nodes are set as 0 to disable this score plugin
		return 0, nil
	}

	var score float64
	score = 0
	gpuInfos := am.GetNodeGpuInfo(node.Name)
	var limitedUtil, limitedMem float64
	limitedUtil = 80
	limitedMem = 16 * 1024 * 1024 * 1024 // 16GB
	bestIndex := -1
	var minUtil float64
	minUtil = 101
	for _, gpu := range *gpuInfos {
		klog.Infof("%v: %v, %v", gpu.index, gpu.usedUtil, gpu.usedMem)
		if gpu.usedUtil < limitedUtil && gpu.usedMem < limitedMem {
			if minUtil > gpu.usedUtil {
				minUtil = gpu.usedUtil
				bestIndex = gpu.index
			}
		}
	}
	if bestIndex != -1 {
		score = 100 - minUtil
	} else {
		// no qualified gpu found
		// TODO
	}
	// score should be [0,100]
	return int64(score), nil
}

func (am *Antman) returnGpuResources(obj interface{}) {
	pod := obj.(*v1.Pod)

	opportunistic, _ := GetPodTypeLabels(pod)
	// only for resource-guarantee job
	if len(opportunistic) == 0 {
		isWithGpu := false
		requestGpusStr := ""
		for i := range pod.Spec.Containers {
			if _, ok := pod.Spec.Containers[i].Resources.Limits[ResourceNvidiaGPU]; ok {
				isWithGpu = true
				requestGpusStr = pod.Annotations[AnnotationGPUVisibleDevices]
			} else if _, ok := pod.Spec.Containers[i].Resources.Requests[ResourceNvidiaGPU]; ok {
				isWithGpu = true
				requestGpusStr = pod.Annotations[AnnotationGPUVisibleDevices]
			}
			if isWithGpu {
				break
			}
		}
		if isWithGpu {
			nodeName := pod.Spec.NodeName

			freeGpus := am.GetNodeFreeGpus(nodeName)

			requestGpusStrs := strings.Split(requestGpusStr, ",")
			for _, gpuIdStr := range requestGpusStrs {
				if gpuId, err := strconv.Atoi(gpuIdStr); err == nil {
					freeGpus = append(freeGpus, gpuId)
				}
			}
			am.SetNodeFreeGpus(nodeName, freeGpus)
		}
	}
	return
}

func (am *Antman) podUpdate(old, new interface{}) {
	oldPod := old.(*v1.Pod)
	newPod := new.(*v1.Pod)
	klog.Infof(
		"POD UPDATED. %s/%s %s->%s",
		oldPod.Namespace, oldPod.Name, oldPod.Status.Phase, newPod.Status.Phase,
	)
	if (oldPod.Status.Phase != newPod.Status.Phase) &&
		(newPod.Status.Phase == "Succeeded" || newPod.Status.Phase == "Failed" || (oldPod.Status.Phase == "Running" && newPod.Status.Phase == "Pending")) {
		// Running->Pending is possible when killing a running tfjob
		am.UnregisterPods(newPod.Spec.NodeName, newPod)
		am.returnGpuResources(new)
	}
}
