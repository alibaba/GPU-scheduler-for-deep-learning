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

// TODO we need to add some ut

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/scheduler/listers"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// FakeNew is used for test.
func FakeNew(clock util.Clock, stop chan struct{}) (*Antman, error) {
	am := &Antman{
		clock: clock,
	}
	// go wait.Until(am.podGroupInfoGC, time.Duration(am.args.PodGroupGCIntervalSeconds)*time.Second, stop)
	return am, nil
}

func TestLess(t *testing.T) {

}

func TestPreFilter(t *testing.T) {

}

func TestPermit(t *testing.T) {

}

func TestPodGroupClean(t *testing.T) {

}

var _ listers.SharedLister = &fakeSharedLister{}

type fakeSharedLister struct {
	pods []*v1.Pod
}

func (f *fakeSharedLister) Pods() listers.PodLister {
	return f
}

func (f *fakeSharedLister) List(_ labels.Selector) ([]*v1.Pod, error) {
	return f.pods, nil
}

func (f *fakeSharedLister) FilteredList(podFilter listers.PodFilter, selector labels.Selector) ([]*v1.Pod, error) {
	pods := make([]*v1.Pod, 0, len(f.pods))
	for _, pod := range f.pods {
		if podFilter(pod) && selector.Matches(labels.Set(pod.Labels)) {
			pods = append(pods, pod)
		}
	}
	return pods, nil
}

func (f *fakeSharedLister) NodeInfos() listers.NodeInfoLister {
	return nil
}
