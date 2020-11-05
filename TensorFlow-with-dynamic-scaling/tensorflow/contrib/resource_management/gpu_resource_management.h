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

#ifndef TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_GPU_RESOURCE_MANAGEMENT_H_
#define TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_GPU_RESOURCE_MANAGEMENT_H_

#include <string>
#include <unordered_map>

#include "include/json/json.h"

#include "tensorflow/contrib/resource_management/file_listener.h"
#include "tensorflow/contrib/resource_management/gpu_usage_adjustment.h"
#include "tensorflow/core/common_runtime/session_run_action_registry.h"

namespace tensorflow {

// Determine if we need to adjust the GPU resource at the end of
// each RunStep.
class GPUResourceManagement : public SessionRunAction {
 public:
  // Note that we will enable TF_FORCE_GPU_ALLOW_GROWTH and TF_GPU_VMEM
  // automatically if the GPUResourceManagement feature is enabled.
  GPUResourceManagement();
  ~GPUResourceManagement() override;

  // For adjusting the limit of each resource after each SessionRun.
  Status RunAction(const SessionRunActionOptions& options) override;

  // Get the new gpu resource limit from the json string.
  bool ParseManageInfoFromJson(const std::string& json_str);

  // Disable the GPUResourceManagement feature.
  void DisableGPUResourceManagement();

 private:
  // Adjust the GPU usage limit of this job.
  // void AdjustUsage();

  // Get the new GPU memory limit from the json string.
  void ParseMemoryLimitFromJson(const Json::Value& json);

  // Get the new GPU usage limit from the json string.
  void ParseUsageLimitFromJson(const Json::Value& json);

  // Sleep a specific time after each SessionRun or Suspend this job.
  void DoSleepOrSuspend(uint64 sess_duration_us);

  mutable mutex manage_mu_;
  mutable mutex usage_mu_;

  std::string gpu_resource_manage_file_path_;

  // Mark whether the GPUResourceManagement feature
  // is enabled.
  std::atomic<bool> enable_gpu_resource_manage_;

  // For recording the parsed new gpu resource limit.
  std::unordered_map<std::string, GPUResourceLimitInfo>
      gpu_resource_management_info_;

  // For recording the parsed new gpu performance limitation
  // (if the value is 0, then it means to suspend this job).
  std::atomic<int> gpu_perf_control_;

  // Determine if we need to adjust the GPU usage limit.
  std::atomic<bool> need_to_adjust_memory_;

  // For performing the adjustment.
  GPUUsageAdjustment* gpu_usage_adjustment_;

  const std::string FILE_LISTENER_NAME = "GPUResourceManage";
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_GPU_RESOURCE_MANAGEMENT_H_
