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

#include <algorithm>
#include <climits>

#include "tensorflow/contrib/resource_management/gpu_usage_adjustment.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

// Determine if this device is a GPU device.
bool IsGpuDevice(const std::string& device_type) {
  std::string dt;
  std::transform(device_type.begin(), device_type.end(),
      std::back_inserter(dt), ::toupper);
  return (dt == "GPU");
}

// Get the value of pci bus id in the giving string.
std::string GetPciBusId(const std::string& ori_line) {
  std::string line;
  // Convert uppercase letter to lowercase.
  std::transform(ori_line.begin(), ori_line.end(),
      std::back_inserter(line), ::tolower);

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

GPUBFCAllocator* GetAllocatorByPciBusId(
    tensorflow::Device* device,
    const std::string& gpu_pci_bus_id) {
  if (!IsGpuDevice(device->device_type())) {
    return nullptr;
  }

  // Get the pci bus id of this GPU
  std::string device_desc = device->attributes().physical_device_desc();
  std::string device_pci_bus_id = GetPciBusId(device_desc);

  // Find the target GPU, return its allocator.
  if (device_pci_bus_id == gpu_pci_bus_id) {
    AllocatorAttributes alloc_attrs;
    // TODO(shiru): need to support adjusting cuda_host_allocators
    // alloc_attrs.set_on_host(true);
    // alloc_attrs.set_gpu_compatible(true);
    Allocator* allocator = device->GetAllocator(alloc_attrs);
    GPUBFCAllocator * gpu_allocator =
        dynamic_cast<GPUBFCAllocator *>(allocator);
    if (gpu_allocator == nullptr) {
      GPUVMemAllocator * vmem_allocator =
          dynamic_cast<GPUVMemAllocator *>(allocator);
      if (vmem_allocator == nullptr) {
        return nullptr;
      }
      gpu_allocator = dynamic_cast<GPUBFCAllocator *>(
          vmem_allocator->DeviceAllocator());
    }
    return gpu_allocator;
  }

  return nullptr;
}

}  // namespace

GPUBFCAllocator* GPUUsageAdjustment::GetGPUAllocator(
    const std::unique_ptr<const tensorflow::DeviceMgr>* device_mgr,
    const std::unique_ptr<DeviceSet>* device_set,
    const std::string& gpu_pci_bus_id) {

  if (device_mgr != nullptr) {
    for (const auto& iter : (*device_mgr)->ListDevices()) {
      GPUBFCAllocator* alloc = GetAllocatorByPciBusId(iter, gpu_pci_bus_id);
      if (alloc != nullptr) {
        return alloc;
      }
    }
  } else if (device_set != nullptr) {
    for (const auto& iter : (*device_set)->devices()) {
      GPUBFCAllocator* alloc = GetAllocatorByPciBusId(iter, gpu_pci_bus_id);
      if (alloc != nullptr) {
        return alloc;
      }
    }
  } else {
    LOG(ERROR) << "Failed to get the device_list from "
               << "the SessionRunActionOptions";
    return nullptr;
  }
  return nullptr;
}

bool GPUUsageAdjustment::AdjustMemLimit(const std::string& gpu_pci_bus_id,
    size_t new_mem_limit,
    const std::unique_ptr<const tensorflow::DeviceMgr>* device_mgr,
    const std::unique_ptr<DeviceSet>* device_set) {
  mutex_lock l(adj_mu_);

  auto cur_info = cur_usage_info_.find(gpu_pci_bus_id);
  if (cur_info == cur_usage_info_.end()) {
    GPUBFCAllocator* allo = GetGPUAllocator(device_mgr,
        device_set, gpu_pci_bus_id);
    if (allo == nullptr) {
      LOG(ERROR) << "Failed to get the allocator of gpu_pci_bus_id: "
                 << gpu_pci_bus_id;
      return false;
    }
    GPUUsageInfo usage_info;
    usage_info.gpu_allocator_ = allo;
    usage_info.cur_limit_.mem_limit_ = ULONG_MAX;
    // Get the VGPU_MEMORY_LIMIT
    absl::optional<AllocatorStats> device_stats = allo->GetStats();
    usage_info.cur_limit_.initial_mem_limit_ = 
      device_stats ? *device_stats->bytes_limit : ULONG_MAX;

    auto ret = cur_usage_info_.emplace(gpu_pci_bus_id, usage_info);
    if (ret.second == false) {
      return false;
    }
    cur_info = ret.first;
  }

  if (new_mem_limit > cur_info->second.cur_limit_.initial_mem_limit_) {
    // The new mem size limit exceeds VGPU_MEMORY_LIMIT
    new_mem_limit = cur_info->second.cur_limit_.initial_mem_limit_;
    LOG(WARNING) << "The new mem size limit exceeds VGPU_MEMORY_LIMIT, "
                 << "therefore, adjust the new mem size limit to : "
                 << new_mem_limit;
  }

  if (cur_info->second.cur_limit_.mem_limit_ != new_mem_limit
      && new_mem_limit >= 0) {
    // Adjust the memory limit of this GPU
    LOG(INFO) << "Start to manage the mem size limit to "
              << new_mem_limit
              << " of device gpu_pci_bus_id: "
              << gpu_pci_bus_id;
    GPUAdjustableAllocator* adj = new GPUAdjustableAllocator();
    size_t cur_mem_limit = adj->AdjustMemoryLimit(new_mem_limit,
        cur_info->second.gpu_allocator_);
    cur_info->second.cur_limit_.mem_limit_ = cur_mem_limit;
    if (cur_mem_limit > new_mem_limit) {
      LOG(ERROR) << "Failed to manage the mem size limit to "
                 << new_mem_limit
                 << " of device gpu_pci_bus_id: "
                 << gpu_pci_bus_id;
      // TODO(shiru): need to check is gpu_allocator_ has been changed!
      return false;
    }
    return true;
  }
  return false;
}

}  // namespace tensorflow
