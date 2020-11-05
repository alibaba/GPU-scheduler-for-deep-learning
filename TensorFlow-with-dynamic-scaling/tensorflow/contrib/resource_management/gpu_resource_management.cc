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

#include "tensorflow/contrib/resource_management/gpu_resource_management.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"

const int fixed_cost = 50; //us

namespace tensorflow {

GPUResourceManagement::GPUResourceManagement()
    : need_to_adjust_memory_(false),
    gpu_perf_control_(100),
    estimated_total_idle_time_(0),
    total_time_slot_(0),
    gpu_usage_adjustment_(new GPUUsageAdjustment()) {
  ReadStringFromEnvVar("GPU_CONFIG_FILE", "", &gpu_resource_manage_file_path_);
  if (gpu_resource_manage_file_path_.empty()) {
    enable_gpu_resource_manage_ = false;
  } else {
    enable_gpu_resource_manage_ = true;
    // Note that we will enable TF_FORCE_GPU_ALLOW_GROWTH and TF_GPU_VMEM
    // automatically if the GPUResourceManagement feature is enabled.
    setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true", 1);
    setenv("TF_GPU_VMEM", "true", 1);

    // Register a handler that will be triggered when the file named
    FileListener::GlobalFileListener()->RegisterFileListener(
        gpu_resource_manage_file_path_, FILE_LISTENER_NAME,
        [](const std::string& str) {
          // The callback func which is invoked when file changed.
          // std::cout << " I'm the callback func"
          //           << " on GPUResourceManage.json."
          //           << " File content: \n"
          //           << str << '\n';
          SessionRunAction* act =
              SessionRunActionRegistry::Global()->GetAction(
                  SessionRunActionRegistry::POST_SESSION_RUN, 2,
                  "GPUResourceManagement");
          if (act == nullptr) {
            std::cout << "Cannot get the instance of GPUResourceManagement \n";
          }
          if (act != nullptr) {
            GPUResourceManagement* rm =
                dynamic_cast<GPUResourceManagement *>(act);
            if (rm != nullptr) {
              rm->ParseManageInfoFromJson(str);
            }
          }
        });
  }
}

GPUResourceManagement::~GPUResourceManagement() {
  if (enable_gpu_resource_manage_) {
    FileListener::GlobalFileListener()->UnregisterFileListener(
        gpu_resource_manage_file_path_, FILE_LISTENER_NAME);
  }
  delete gpu_usage_adjustment_;
}

void GPUResourceManagement::ParseMemoryLimitFromJson(const Json::Value& json) {
  if (json["gpuConfigInfo"].isNull()) {
    return;
  }
  mutex_lock l(manage_mu_);
  Json::Value::Members members = json["gpuConfigInfo"].getMemberNames();
  for (const auto& it : members) {
    if (json["gpuConfigInfo"][it]["maxDeviceMem"].isNull() ||
        !json["gpuConfigInfo"][it]["maxDeviceMem"].isUInt64() ) {
      continue;
    }

    GPUResourceLimitInfo limit_info;
    limit_info.mem_limit_ =
        json["gpuConfigInfo"][it]["maxDeviceMem"].asUInt64();
    auto res = gpu_resource_management_info_.emplace(it, limit_info);
    if (res.second == false) {
      // To store every new resource limit properly, we will never clear
      // the 'gpu_resource_management_info_' map, therefore the emplace
      // operation may return false since there is a equivalent
      // element existed already in the map.
      res.first->second.mem_limit_ = limit_info.mem_limit_;
    }
    VLOG(2) << "Parse GPU pci_bus: " << res.first->first
            << " maxDeviceMem: " << res.first->second.mem_limit_;
  }

  if (gpu_resource_management_info_.size() != 0) {
    need_to_adjust_memory_ = true;
  }
}

void GPUResourceManagement::ParseUsageLimitFromJson(const Json::Value& json) {
  if (json["perfControl"].isNull() || !json["perfControl"].isUInt64()) {
    return;
  }

  mutex_lock l(usage_mu_);
  int parsed_new_limit = json["perfControl"].asUInt64();
  if (parsed_new_limit == gpu_perf_control_) {
    return;
  }

  // LOG(INFO) << "Parse GPU usage limit: " << parsed_new_limit;
  if (parsed_new_limit < 0 && parsed_new_limit > 100) {
    gpu_perf_control_ = 100;
    LOG(INFO) << "Invalid new perfControl (should > 0 && < 100):"
                  << parsed_new_limit;
    return;
  }
  gpu_perf_control_ = parsed_new_limit;
}

bool GPUResourceManagement::ParseManageInfoFromJson(
    const std::string& json_str) {
  Json::Reader reader;
  Json::Value json;

  if (!reader.parse(json_str, json)) {
    LOG(INFO) << "Failed to parse the json string";
    return false;
  }

  ParseMemoryLimitFromJson(json);
  ParseUsageLimitFromJson(json);
  return true;
}

void GPUResourceManagement::DisableGPUResourceManagement() {
  enable_gpu_resource_manage_ = false;
}

void GPUResourceManagement::DoSleepOrSuspend(
    uint64 sess_duration_us) {
  if (gpu_perf_control_ >= 100) {
    total_time_slot_ = 0;
    SetEstimatedIdleTime(0);
    return;
  }
  if (gpu_perf_control_ == 0) {
    // need to suspend.
    while (gpu_perf_control_ == 0) {
      usleep(100);
    }
  } else {
    // Need to sleep a specific time after each SessionRun
    // if we don't insert enough time slots.
    uint64 actual_sleep_time = total_time_slot_ > estimated_total_idle_time_ ?
        (total_time_slot_ - estimated_total_idle_time_) : 0;
    uint64 actual_sess_run_time = sess_duration_us > actual_sleep_time ?
        (sess_duration_us - actual_sleep_time) : 1;
    uint64 total_time = actual_sess_run_time * 100 / gpu_perf_control_;

    if (total_time > sess_duration_us + fixed_cost) {
      uint64 sleep_time = total_time - sess_duration_us - fixed_cost;
      SetEstimatedIdleTime(sleep_time);
      total_time_slot_ = sleep_time;
      usleep(sleep_time);
    } else {
      SetEstimatedIdleTime(0);
      total_time_slot_ = 0;
    }
  }
}

Status GPUResourceManagement::RunAction(
    const SessionRunActionOptions& options) {
  if (!need_to_adjust_memory_ && gpu_perf_control_ >= 100) {
    // TODO(shiru): do we need to unregister the
    // GPUResourceManagement if the environment variable
    // GPU_CONFIG_FILE is set to null?
    return Status::OK();
  }

  if (need_to_adjust_memory_) {
    mutex_lock l(manage_mu_);
    // Start to adjust the resource limit as required.
    for (const auto& it : gpu_resource_management_info_) {
      gpu_usage_adjustment_->AdjustMemLimit(it.first,
          it.second.mem_limit_, options.device_mgr,
          options.device_set);
    }
    need_to_adjust_memory_ = false;
  }

  DoSleepOrSuspend(options.sess_duration_us);

  return Status::OK();
}

uint64 GPUResourceManagement::GetExecutorQueuedOpNum(const void* executor_ptr) {
  auto ec = executor_queued_op_num_.find(executor_ptr);
  if (ec != executor_queued_op_num_.end()) {
    return ec->second;
  }
  return 0;
}

void GPUResourceManagement::SetExecutorQueuedOpNum(const void* executor_ptr, uint64 queued_op_num) {
  auto ec = executor_queued_op_num_.find(executor_ptr);
  if (ec != executor_queued_op_num_.end()) {
    if (queued_op_num > ec->second) {
      ec->second = queued_op_num;
    }
  } else {
    executor_queued_op_num_.emplace(executor_ptr, queued_op_num);
  }
}


#if GOOGLE_CUDA
// We register the GPUResourceManagement as a POST_SESSION_RUN action
// during the initialization phase of the program.
REGISTER_SESSION_RUN_ACTION(SessionRunActionRegistry::POST_SESSION_RUN,
                            2, GPUResourceManagement);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
