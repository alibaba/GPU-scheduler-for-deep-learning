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

#ifndef TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_GPU_STATISTICS_H_
#define TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_GPU_STATISTICS_H_

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <fstream>
#include <vector>

#include "include/json/json.h"
#include "include/json/writer.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/common_runtime/gpu/gpu_vmem_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_adjustable_allocator.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/session_run_action_registry.h"

namespace tensorflow {

// Maintain the status info of this GPUVMemAllocator.
struct GPUVMemAllocatorStatus {
  // Json values
  // Max mini-batch usage of device allocator
  int64_t deviceMemUsedMax{0};
  // Min mini-batch usage of device allocator
  int64_t deviceMemUsedMin{0};
  // Total memory pool size of device allocator
  int64_t deviceMemPoolSize{0};
  // In used memory size of device allocator
  int64_t deviceMemStable{0};
  // Max mini-batch usage of host allocator
  int64_t hostMemUsedMax{0};
  // Min mini-batch usage of host allocator
  int64_t hostMemUsedMin{0};
  // Total memory pool size of host allocator
  int64_t hostMemPoolSize{0};
  // SwapReason:
  //   "OOM": the device memory has been swapped out to the host
  //    since the it has run out of memory.
  //   "FRAGMENT": the device memory has been swapped out to the host
  //    since there are some disjointed fragments in memory regions.
  // TODO(shiru): This may cause misunderstanding when the GPUResourceManagement
  // feature is enabled (i.e., GPU memory shrinking happens).
  std::string swapReason{""};
  // Allocator info
  GPUVMemAllocator* vmem_allocator{nullptr};
  std::string gpu_pci_bus_id{""};

  // Check whether the status of this GPUVMemAllocator has been changed.
  bool CheckChanged(const GPUVMemAllocatorStatus& target) const;

  // Update the status of this GPUVMemAllocator.
  void Update();
};

// Dump the statistics at the end of each RunStep.
class GPUStatistics : public SessionRunAction {
 public:
  GPUStatistics();
  ~GPUStatistics();

  // For outputting the GPU Statistics.
  Status RunAction(const SessionRunActionOptions& options) override;

 private:
  // Check whether should we check the GPU statistics.
  bool ShouldCheckGPUStatistics() const;

  // Init the allocator_status_lists_
  void InitAllocatorStatus(Device* device);

  // Check the status of each GPUVMemAllocator to determine if we
  // need to dump the statistics.
  bool CheckGPUVMemAllocatorStatistics(
      const std::unique_ptr<const tensorflow::DeviceMgr>* device_mgr,
      const std::unique_ptr<DeviceSet>* device_set);

  // Dump GPU statistics.
  void dumpGPUStatistics();

  // Check the duration of each SessionRun to determine if we
  // need to dump the statistics.
  bool CheckSessionRunDuration(const uint64 graph_id,
      const uint64 sess_duration_us);

  // Record the duration time of this SessionRun.
  bool RecordSessionRunDuration(const uint64 graph_id,
      const uint64 sess_duration_us);

  // Readers–writer lock.
  mutable mutex histry_mu_;
  // Global lock.
  mutable mutex check_mu_;

  // To store the status info of each GPUVMemAllocator.
  std::vector<GPUVMemAllocatorStatus> allocator_status_lists_;

  // To record the time that last statistics dump.
  time_t gpu_statistics_last_write_;

  // Mark whether the GPUStatistics feature
  // is enabled.
  std::atomic<bool> need_to_dump_statistics_;

  // To store the file path.
  std::string gpu_statistics_file_;

  // To store the time duration and the recording time.
  struct DurationInfo {
    time_t recording_time_;
    uint64 duration_;
  };

  // To store the list of the time durations.
  struct DurationHistry {
    mutable mutex histry_mu_;
    std::list<DurationInfo> histry_;
  };

  // To store the time durations (in microsecond) of all the sub_graphs
  // of the session run (specified by the void* graph_id).
  std::map<const uint64, DurationInfo> sess_run_durations_;

  std::map<const uint64, DurationHistry*> graph_durations_histry_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_GPU_STATISTICS_H_
