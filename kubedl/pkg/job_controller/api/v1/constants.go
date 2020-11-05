package v1

const (
	// ReplicaIndexLabel represents the label key for the replica-index, e.g. the value is 0, 1, 2.. etc
	ReplicaIndexLabel = "replica-index"

	// ReplicaTypeLabel represents the label key for the replica-type, e.g. the value is ps , worker etc.
	ReplicaTypeLabel = "replica-type"

	// GroupNameLabel represents the label key for group name, e.g. the value is kubeflow.org
	GroupNameLabel = "group-name"

	// JobNameLabel represents the label key for the job name, the value is job name
	JobNameLabel = "job-name"

	// JobRoleLabel represents the label key for the job role, e.g. the value is master
	JobRoleLabel = "job-role"
)

// Constant label/annotation keys for job configuration.
const (
	KubeDLPrefix = "kubedl.io"

	// AnnotationGitSyncConfig annotate git sync configurations.
	AnnotationGitSyncConfig = KubeDLPrefix + "/git-sync-config"
	// AnnotationTenancyInfo annotate tenancy information.
	AnnotationTenancyInfo = KubeDLPrefix + "/tenancy"
	// AnnotationGPUVisibleDevices is the visible gpu devices in view of process.
	AnnotationGPUVisibleDevices = "gpus." + KubeDLPrefix + "/visible-devices"

	// for antman
	AntManPrefix = "alibaba.pai.antman"
	// for DL jobs to report DL framework information to local coordinator.
	AnnotationGpuStatusFile = AntManPrefix + "/gpu-status-file"
	// for local coordinator to control the GPU usage of a DL job.
	AnnotationGpuConfigFile = AntManPrefix + "/gpu-config-file"
)

const (
	DefaultKubeDLNamespace = "kubedl"
)
