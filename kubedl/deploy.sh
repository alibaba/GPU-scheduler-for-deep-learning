#scp bin/manager root@47.92.53.19:/root/deployment/kubedl/
#scp config/crd/bases/kubeflow.org_tfjobs.yaml root@47.92.53.19:/root/deployment/kubedl/
KUBEDL_CI=true make docker-build
docker tag kubedl/kubedl:v0.1.0 registry.cn-huhehaote.aliyuncs.com/pai-image/antman-public:kubedl-v0.1.0
docker push registry.cn-huhehaote.aliyuncs.com/pai-image/antman-public:kubedl-v0.1.0
