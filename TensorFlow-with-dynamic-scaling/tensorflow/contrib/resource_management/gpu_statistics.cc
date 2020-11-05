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

#include "tensorflow/contrib/resource_management/gpu_statistics.h"

const char* gpu_statistics_file_flag = "GPU_STATUS_FILE";
const int gpu_statistics_interval = 10;  // second
const int max_record_interval = 100;  // second
const int duration_statistics_num_sess = 10;
const double duration_statistics_threshold = 0.01;
const int duration_huge_change_threshold = 10;

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

}  // namespace

bool GPUVMemAllocatorStatus::CheckChanged(
    const GPUVMemAllocatorStatus& target) const {
  if (target.deviceMemUsedMax != deviceMemUsedMax ||
      target.deviceMemUsedMin != deviceMemUsedMin ||
      target.deviceMemPoolSize != deviceMemPoolSize ||
      target.deviceMemStable != deviceMemStable ||
      target.hostMemUsedMax != hostMemUsedMax ||
      target.hostMemUsedMin != hostMemUsedMin ||
      target.hostMemPoolSize != hostMemPoolSize) {
    return true;
  }
  return false;
}

void GPUVMemAllocatorStatus::Update() {
  absl::optional<AllocatorStats> host_stats;
  absl::optional<AllocatorStats> device_stats;
  host_stats = vmem_allocator->HostAllocator()->GetStats();
  Allocator* device_allocator = vmem_allocator->DeviceAllocator();
  device_stats = device_allocator->GetStats();
  deviceMemUsedMax = device_stats->peak_bytes_in_use;
  deviceMemUsedMin = device_stats->bytes_in_use;
  hostMemUsedMax = host_stats->peak_bytes_in_use;
  hostMemUsedMin = host_stats->bytes_in_use;
  hostMemPoolSize = *host_stats->bytes_limit;

  BFCAllocator * device_bfc_allocator =
      dynamic_cast<BFCAllocator *>(device_allocator);
  if (device_bfc_allocator == nullptr) {
    LOG(ERROR) << "Cannot get the BFCAllocator of the GPU device.";
    return;
  }
  GPUAdjustableAllocator* adj = new GPUAdjustableAllocator();
  adj->GetMemPoolStats(device_bfc_allocator,
      &deviceMemPoolSize, &deviceMemStable);
}

GPUStatistics::GPUStatistics()
    : gpu_statistics_last_write_(0),
      need_to_dump_statistics_(false) {
  Status status = ReadStringFromEnvVar(gpu_statistics_file_flag,
                                       "",
                                       &gpu_statistics_file_);
  if (!status.ok() || gpu_statistics_file_.empty()) {
    // LOG(WARNING) << "Error getting the " << gpu_statistics_file_flag << ": "
    //              << status.error_message()
    //              << ", GPUStatistics has been disabled.";
    // GPUStatistics has been disabled.
    need_to_dump_statistics_ = false;
    return;
  }

  // GPUStatistics only work when gpu_vmem is on
  // (gpu_vmem has been enabled by default).

  CHECK_EQ(allocator_status_lists_.size(), 0);
  need_to_dump_statistics_ = true;
  LOG(INFO) << "GPUStatistics monitor start.";
}

GPUStatistics::~GPUStatistics() {
  for (auto it = graph_durations_histry_.begin();
      it != graph_durations_histry_.end(); ) {
    delete it->second;
    graph_durations_histry_.erase(it++);
  }
  graph_durations_histry_.clear();
}

bool GPUStatistics::ShouldCheckGPUStatistics() const {
  return gpu_statistics_last_write_ == 0 ||
      time(0) - gpu_statistics_last_write_ >= gpu_statistics_interval;
}

void GPUStatistics::InitAllocatorStatus(Device* device) {
  if (!IsGpuDevice(device->device_type())) {
    return;
  }
  AllocatorAttributes alloc_attrs;
  Allocator* allocator = device->GetAllocator(alloc_attrs);
  GPUVMemAllocator * vmem_allocator =
      dynamic_cast<GPUVMemAllocator *>(allocator);
  if (vmem_allocator == nullptr) {
    // LOG(INFO) << "Cannot get the GPUVMemAllocator of device:"
    //           << device->attributes().physical_device_desc();
    return;
  }
  GPUVMemAllocatorStatus status;
  status.gpu_pci_bus_id =
      GetPciBusId(device->attributes().physical_device_desc());
  status.vmem_allocator = vmem_allocator;
  allocator_status_lists_.emplace_back(status);
}

bool GPUStatistics::CheckGPUVMemAllocatorStatistics(
    const std::unique_ptr<const tensorflow::DeviceMgr>* device_mgr,
    const std::unique_ptr<DeviceSet>* device_set) {
  bool need_update = false;

  if (allocator_status_lists_.size() == 0) {
    // Init the allocator_status_lists_
    if (device_mgr != nullptr) {
      for (const auto& d : (*device_mgr)->ListDevices()) {
        InitAllocatorStatus(d);
      }
    } else if (device_set != nullptr) {
      for (const auto& d : (*device_set)->devices()) {
        InitAllocatorStatus(d);
      }
    } else {
      LOG(ERROR) << "Failed to get the device_list from "
                 << "the SessionRunActionOptions";
      return false;
    }
  }

  for (auto& a : allocator_status_lists_) {
    GPUVMemAllocatorStatus last_status(a);
    a.Update();
    if (!a.CheckChanged(last_status)) {
      continue;
    }
    need_update = true;
    if (a.hostMemUsedMax == 0) {
      a.swapReason = std::string("");
    } else {
      a.swapReason =
          a.deviceMemUsedMax + a.hostMemUsedMax > a.deviceMemPoolSize ?
              std::string("FRAGMENT") : std::string("OOM");
    }
  }
  return need_update;
}

bool GPUStatistics::RecordSessionRunDuration(const uint64 graph_id,
    const uint64 sess_duration_us) {
  DurationHistry* histry_list = nullptr;
  // Read lock.
  histry_mu_.lock_shared();
  auto tar_graph_histry = graph_durations_histry_.find(graph_id);
  bool not_found = tar_graph_histry == graph_durations_histry_.end();
  if (!not_found) {
    histry_list = tar_graph_histry->second;
  }
  histry_mu_.unlock_shared();

  if (not_found) {
    DurationHistry* d_l = new DurationHistry();
    DurationInfo d_info{time(0), sess_duration_us};
    d_l->histry_.emplace_back(d_info);
    // Write lock.
    histry_mu_.lock();
    graph_durations_histry_.emplace(graph_id, d_l);
    histry_mu_.unlock();
    return false;
  }

  uint64 last = 0;
  histry_list->histry_mu_.lock();
  if (histry_list->histry_.size() > 0) {
    last = histry_list->histry_.front().duration_;
  }
  DurationInfo d_info{time(0), sess_duration_us};
  histry_list->histry_.emplace_back(d_info);
  if (histry_list->histry_.size() >= duration_statistics_num_sess) {
    histry_list->histry_.pop_front();
  }
  histry_list->histry_mu_.unlock();

  if (sess_duration_us > last * duration_huge_change_threshold ||
      last > sess_duration_us * duration_huge_change_threshold) {
    // Huge change.
    return true;
  }
  return false;
}

bool GPUStatistics::CheckSessionRunDuration(const uint64 graph_id,
    const uint64 sess_duration_us) {
  auto tar_sess_duration = sess_run_durations_.find(graph_id);
  if (tar_sess_duration == sess_run_durations_.end()) {
    DurationInfo du = {time(0), sess_duration_us};
    sess_run_durations_.emplace(graph_id, du);
    return true;
  }

  // Update duration histry for all graphs.
  std::map<const uint64, DurationHistry*> all_du_histry;
  // Read lock.
  histry_mu_.lock_shared();
  for (const auto& dur : graph_durations_histry_) {
    all_du_histry.emplace(dur.first, dur.second);
  }
  histry_mu_.unlock_shared();

  bool need_update_dur = false;
  for (const auto& dur_histry : all_du_histry) {
    uint64 sum = 0;
    uint64 g_id = dur_histry.first;
    DurationHistry* histry_list = dur_histry.second;
    histry_list->histry_mu_.lock_shared();
    for (const auto& h : histry_list->histry_) {
      sum += h.duration_;
    }
    uint64 d_size = histry_list->histry_.size();
    time_t recording_time = histry_list->histry_.back().recording_time_;
    histry_list->histry_mu_.unlock_shared();

    if (time(0) - recording_time < max_record_interval) {
      uint64 cur_duration = sum / d_size;
      uint64 last_duration = sess_run_durations_[g_id].duration_;
      uint64 diff = cur_duration > last_duration ?
          cur_duration - last_duration : last_duration - cur_duration;
      sess_run_durations_[g_id].duration_ = cur_duration;
      sess_run_durations_[g_id].recording_time_ = recording_time;
      if (diff >= duration_statistics_threshold * last_duration) {
      need_update_dur = true;
      }
    }
  }

  return need_update_dur;
}

void GPUStatistics::dumpGPUStatistics() {
  Json::Value dump_json;
  Json::Value gpu_info_json;
  for (const auto& a : allocator_status_lists_) {
    Json::Value device_json;
    device_json["deviceMemUsedMax"] = Json::Int64(a.deviceMemUsedMax);
    device_json["deviceMemUsedMin"] = Json::Int64(a.deviceMemUsedMin);
    device_json["deviceMemPoolSize"] = Json::Int64(a.deviceMemPoolSize);
    device_json["deviceMemStable"] = Json::Int64(a.deviceMemStable);
    device_json["hostMemUsedMax"] = Json::Int64(a.hostMemUsedMax);
    device_json["hostMemUsedMin"] = Json::Int64(a.hostMemUsedMin);
    device_json["hostMemPoolSize"] = Json::Int64(a.hostMemPoolSize);
    device_json["swapReason"] = a.swapReason;
    device_json["deviceMemUsedNvidia"] = Json::Int64(-1);
    gpu_info_json[a.gpu_pci_bus_id] = device_json;
  }
  dump_json["gpuUsageInfo"] = gpu_info_json;

  Json::Value sess_json;
  uint64 max_duration = 0;
  for (const auto& s : sess_run_durations_) {
    uint64 du = s.second.duration_;
    time_t rec = s.second.recording_time_;
    sess_json["graph_" + std::to_string(s.first)] = Json::UInt64(du);
    if (du > max_duration && time(0) - rec < max_record_interval) {
      max_duration = du;
    }
  }
  dump_json["miniBatchDuration"] = Json::UInt64(max_duration);
  dump_json["Durations"] = sess_json;

  Json::StreamWriterBuilder stream_writer;
  std::unique_ptr<Json::StreamWriter> writer(stream_writer.newStreamWriter());
  std::ofstream statistics_file;
  statistics_file.open(gpu_statistics_file_);
  writer->write(dump_json, &statistics_file);
  statistics_file.close();
  // LOG(INFO) << "gpu_statistics_file updated.";
}

Status GPUStatistics::RunAction(const SessionRunActionOptions& options) {
  if (!need_to_dump_statistics_) {
    return Status::OK();
  }
  bool huge_change = RecordSessionRunDuration(
      options.graph_id, options.sess_duration_us);

  if (!ShouldCheckGPUStatistics() && !huge_change) {
    return Status::OK();
  }

  {
    // Global lock.
    mutex_lock l(check_mu_);
    bool dur_flag = CheckSessionRunDuration(options.graph_id,
        options.sess_duration_us);
    bool stat_flag = CheckGPUVMemAllocatorStatistics(options.device_mgr,
        options.device_set);
    if (dur_flag || stat_flag) {
      dumpGPUStatistics();
    }

    gpu_statistics_last_write_ = time(0);
  }

  return Status::OK();
}

#if GOOGLE_CUDA
// We register the GPUStatistics as a POST_SESSION_RUN action
// during the initialization phase of the program.
REGISTER_SESSION_RUN_ACTION(SessionRunActionRegistry::POST_SESSION_RUN,
                            1, GPUStatistics);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
