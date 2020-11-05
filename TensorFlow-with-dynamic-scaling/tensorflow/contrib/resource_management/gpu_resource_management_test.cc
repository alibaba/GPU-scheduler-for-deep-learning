/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

#include "tensorflow/contrib/resource_management/gpu_resource_management.h"
#include "tensorflow/contrib/resource_management/file_listener.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_vmem_allocator.h"
#include "tensorflow/core/common_runtime/session_run_action_registry.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

namespace {

const char* kDeviceNamePrefix = "/job:localhost/replica:0/task:0";

// Get the value of pci bus id in the giving string.
std::string GetPciBusId(const std::string& line) {
  const std::string str_pci_bus_id = "pci bus id: ";
  std::string::size_type start = line.find(str_pci_bus_id);
  if (start == std::string::npos) {
    return std::string();  // empty string
  }
  start += str_pci_bus_id.length();
  std::string::size_type end = line.find(",", start);
  if (end == std::string::npos) {
    end = line.length();
  }
  std::string::size_type start_r = line.find(":", start) + 1;
  if (start_r >= end) {
    return std::string();  // empty string
  }
  std::string gpu_pci_id = line.substr(start_r, end - start_r);
  return gpu_pci_id;
}

class GPUResourceManagementTest {
 public:
  GPUResourceManagementTest(
      const string& visible_device_list = "",
      double per_process_gpu_memory_fraction = 0, int gpu_device_count = 1,
      const std::vector<std::vector<float>>& memory_limit_mb = {}) {
    ConfigProto* config = &options_.config;
    (*config->mutable_device_count())["GPU"] = gpu_device_count;
    GPUOptions* gpu_options = config->mutable_gpu_options();
    gpu_options->set_visible_device_list(visible_device_list);
    gpu_options->set_per_process_gpu_memory_fraction(
        per_process_gpu_memory_fraction);
    for (const auto& v : memory_limit_mb) {
      auto virtual_devices =
          gpu_options->mutable_experimental()->add_virtual_devices();
      for (float mb : v) {
        virtual_devices->add_memory_limit_mb(mb);
      }
    }

    Session* session_(NewSession(options_));
    session_->LocalDeviceManager(&device_mgr_);
    DeviceFactory::GetFactory("GPU")->CreateDevices(
      options_, kDeviceNamePrefix, &devices_);

    int64 memory_limit = devices_[0]->attributes().memory_limit();

    AllocatorAttributes allocator_attributes = AllocatorAttributes();
    allocator_attributes.set_gpu_compatible(true);
    Allocator* allocator = devices_[0]->GetAllocator(allocator_attributes);

    for (const auto& d : device_mgr_->ListDevices()) {
      if (d->device_type() == "GPU") {
        std::string device_desc = d->attributes().physical_device_desc();
        gpu_name_.append(GetPciBusId(device_desc));
      }
    }
  }

  const tensorflow::DeviceMgr* GetDeviceMgr() {
    return device_mgr_;
  }

  std::string GetGPUName() {
    return gpu_name_;
  }

 private:
  Session* session_;
  SessionOptions options_;
  const tensorflow::DeviceMgr* device_mgr_;
  std::vector<std::unique_ptr<tensorflow::Device>> devices_;
  std::string gpu_name_;
};

std::string GenerateJson(std::string gpu_name, size_t mem_limit) {
  std::string json =
      "{\n"
      "    \"name\": \"Odps\/trip_search_20190826062456818guq58p6m_0"
      "\/worker@h04a13286.nt12#18\",\n"
      "    \"gpuConfigInfo\": {\n" 
      "        \"" +  gpu_name + "\": {\n"
      "            \"maxDeviceMem\":" + std::to_string(mem_limit) + "\n"
      "        }\n"
      "    }\n"
      "}\n";
  LOG(INFO) << json;
  return json;
}

GPUVMemAllocator * GetVMemAllocatorFromDevice(
    std::unique_ptr<const tensorflow::DeviceMgr>* p,
    const std::string gpu_name) {
  for (const auto& d : (*p)->ListDevices()) {
    if (d->device_type() != "GPU") {
      continue;
    }

    // Get the pci bus id of this GPU
    std::string device_desc = d->attributes().physical_device_desc();
    std::string device_pci_bus_id = GetPciBusId(device_desc);

    // Find the target GPU, return its allocator.
    if (device_pci_bus_id == gpu_name) {
      AllocatorAttributes alloc_attrs;
      Allocator* allocator = d->GetAllocator(alloc_attrs);
      GPUVMemAllocator * vmem_allocator =
          dynamic_cast<GPUVMemAllocator *>(allocator);
      return vmem_allocator;
    }
  }
  return nullptr;
}

}  // namespace

TEST(GPUResourceManagement, ParseAndAdjustMemoryLimit) {
#ifdef GOOGLE_CUDA

  setenv("TF_GPU_VMEM", "true", 1);
  setenv("TF_CUDA_HOST_MEM_LIMIT_IN_MB", "4096", 1);

  GPUResourceManagementTest* ut = new GPUResourceManagementTest();
  GPUResourceManagement* rm = new GPUResourceManagement();
  SessionRunActionOptions action_options;
  std::unique_ptr<const tensorflow::DeviceMgr> p(ut->GetDeviceMgr());
  action_options.device_mgr = &p;

  size_t cur_limit = 1024 * 1024 * 1UL;

  rm->ParseManageInfoFromJson(GenerateJson(ut->GetGPUName(), cur_limit));
  rm->RunAction(action_options);

  absl::optional<AllocatorStats> stats;
  GPUVMemAllocator * a = GetVMemAllocatorFromDevice(&p, ut->GetGPUName());
  EXPECT_NE(a, nullptr);
  stats = a->DeviceAllocator()->GetStats();
  EXPECT_EQ(*stats->bytes_limit, cur_limit);

  cur_limit = 1024 * 1024 * 1024 * 2UL;
  rm->ParseManageInfoFromJson(GenerateJson(ut->GetGPUName(), cur_limit));
  rm->RunAction(action_options);

  a = GetVMemAllocatorFromDevice(&p, ut->GetGPUName());
  EXPECT_NE(a, nullptr);
  stats = a->DeviceAllocator()->GetStats();
  EXPECT_EQ(*stats->bytes_limit, cur_limit);

#endif  // GOOGLE_CUDA
}

}  // namespace tensorflow
