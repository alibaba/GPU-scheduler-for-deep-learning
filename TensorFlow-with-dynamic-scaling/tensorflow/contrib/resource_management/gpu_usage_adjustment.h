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

#ifndef TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_GPU_USAGE_ADJUSTMENT_H_
#define TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_GPU_USAGE_ADJUSTMENT_H_

#include <string>
#include <memory>
#include <unordered_map>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/gpu/gpu_adjustable_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_vmem_allocator.h"

namespace tensorflow {

struct GPUResourceLimitInfo {
  size_t mem_limit_;
  // For recording VGPU_MEMORY_LIMIT
  size_t initial_mem_limit_;
  // uint32 sm_util;
};

// For performing the adjustment of the usage of both the GPU memory and
// the SM (streaming multiprocessor).
class GPUUsageAdjustment {
 public:
  // Adjust the memory limit of the giving GPU.
  bool AdjustMemLimit(const std::string& gpu_pci_bus_id,
                      size_t new_mem_limit,
                      const std::unique_ptr<const tensorflow::DeviceMgr>* device_mgr,
                      const std::unique_ptr<DeviceSet>* device_set);

 private:
  GPUBFCAllocator*
      GetGPUAllocator(const std::unique_ptr<const tensorflow::DeviceMgr>* device_mgr,
                      const std::unique_ptr<DeviceSet>* device_set,
                      const std::string& gpu_pci_bus_id);

  // Acquire this mutex before adjusting the GPU usage.
  mutable mutex adj_mu_;

  struct GPUUsageInfo {
    GPUBFCAllocator* gpu_allocator_;
    GPUResourceLimitInfo cur_limit_;
  };

  // For recording current GPU usage info.
  std::unordered_map<std::string, GPUUsageInfo> cur_usage_info_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_GPU_USAGE_ADJUSTMENT_H_
