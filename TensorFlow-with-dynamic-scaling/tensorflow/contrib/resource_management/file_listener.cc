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

#include <fstream>
#include <openssl/md5.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "tensorflow/contrib/resource_management/file_listener.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

void CalculateMd5(const unsigned char* buffer,
                  size_t buffer_size, unsigned char* md) {
  MD5(buffer, buffer_size, md);
}

bool CalculateMd5(const char* file_path, unsigned char* md) {
  unsigned char* buffer = nullptr;
  std::ifstream file;
  bool ret = false;

  if (file_path == NULL) {
    LOG(ERROR) << "file_path Must be non NULL";
    return false;
  }
  file.open(file_path, std::ios::ate);
  if (!file.is_open()) {
    // LOG(INFO) << "File " << file_path << " can't be opened!";
    return false;
  } else {
    std::streampos file_size = file.tellg();
    buffer = new unsigned char[file_size];
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(buffer), file_size);
    if (!file) {
      // LOG(INFO) << "Failed to read file " << file_path;
      ret = false;
    } else {
      CalculateMd5(buffer, file_size, md);
      ret = true;
    }
    file.close();
    delete[] buffer;
    return ret;
  }
}

bool IsTimeElapsed(timespec since, long long timeout_in_sec) {
  time_t now = time(0);
  return (now - since.tv_sec > timeout_in_sec);
}

constexpr static int kFileContentCheckTimeoutInSec = 5;
}  // namespace

FileInfo::FileInfo(const std::string& file_path)
    : path_(file_path) {
  // Init 'last_checked_time_' with current time.
  last_checked_time_ = {time(0), 0};
  // Init other info with '0', therefore, for the already existed
  // file, we can invoke the callbacks and load content of the file
  // at the very first time.
  last_modified_time_ = {0, 0};
  file_size_ = 0;
}

bool FileInfo::UpdateFileInfo(const struct stat& file_stat,
                              unsigned char* cur_md5) {
  last_modified_time_ = file_stat.st_mtim;
  file_size_ = file_stat.st_size;
  if (cur_md5 == nullptr) {
    unsigned char cal_md5[kMd5DigestLength];
    if (CalculateMd5(path_.c_str(), cal_md5)) {
      memcpy(md5_, cal_md5, kMd5DigestLength);
    }
  } else {
    memcpy(md5_, cur_md5, kMd5DigestLength);
  }
}

bool FileInfo::IsFileChanged() {
  struct stat file_stat;
  // Get the last modified time of the file.
  if (stat(path_.c_str(), &file_stat) == 0) {
    // Check if the file has been modified again since the last time
    // when file was modified. Since some file systems such as ext3
    // only have last modified time at the granularity of seconds
    // we'll need to check file content when the file hasn't been updated
    // for a very long time.
    if (IsTimeElapsed(last_checked_time_,
                      kFileContentCheckTimeoutInSec)) {
      last_checked_time_ = {time(0), 0};
      unsigned char cur_md5[kMd5DigestLength];
      if (CalculateMd5(path_.c_str(), cur_md5)) {
        if (memcmp(md5_, cur_md5, kMd5DigestLength) != 0) {
            UpdateFileInfo(file_stat, cur_md5);
            return true;
        }
      }
      // No need to log error msg here.
    }

    if (file_stat.st_mtim.tv_sec > last_modified_time_.tv_sec
      || file_stat.st_mtim.tv_nsec > last_modified_time_.tv_nsec
      || file_stat.st_size != file_size_) {
      UpdateFileInfo(file_stat, nullptr);
      return true;
    }
  }
  // No need to log error msg here.

  return false;
}

bool FileInfo::LoadFile(std::string* file_content) {
  std::ifstream file(path_);
  if (!file.is_open()) {
    VLOG(2) << "Failed to read file " << path_;
    return false;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  file_content->append(buffer.str());
  return true;
}

// static
FileListener* FileListener::GlobalFileListener() {
  static FileListener global_file_listener;
  return &global_file_listener;
}

void FileListener::FileMonitor() {
  while (!stop_running_) {
    lock_.lock();
    for (auto& listener : listeners_) {
      if (listener.second.file_info_.IsFileChanged()) {
        std::string file_content;
        if (!listener.second.file_info_.LoadFile(&file_content)) {
          // Failed to read this file, so deal with the next file.
          continue;
        }
        for (const auto& handler : listener.second.file_handlers_) {
          // Invoke each handler one by one (callback func)
          VLOG(2) << "File changed, invoke each handler";
          handler.func_(file_content);
        }
      }
    }
    lock_.unlock();

    usleep(kSleepIntervalInUs);
  }
}

void FileListener::StartMonitorThread() {
  stop_running_ = false;
  file_monitor_thread_ =
      new std::thread(&FileListener::FileMonitor, this);
  // LOG(INFO) << "Start file monitor thread.";
}

void FileListener::StopMonitorThread() {
  stop_running_ = true;
  if (file_monitor_thread_ != nullptr) {
    // LOG(INFO) << "Stop file monitor thread.";
    file_monitor_thread_->join();
    delete file_monitor_thread_;
    file_monitor_thread_ = nullptr;
  }
}

void FileListener::RegisterFileListener(const std::string& file_path,
                                        const std::string& handler_name,
                                        callback callback_func) {
  LOG(INFO) << "Register a file listener named " << handler_name
             << " on file " << file_path;
  FileInfo new_file(file_path);
  std::vector<CallbackFunc> new_handlers;
  InfoAndHandlers value = {new_file, new_handlers};
  CallbackFunc new_callback(handler_name, callback_func);

  mutex_lock l(lock_);
  auto res = listeners_.emplace(file_path, value);
  // if (res.first->second.file_handlers_.find(new_callback) !=
  //     // Guarantee that there are no duplicate handers.
  //     res.first->second.file_handlers_.end()) {
  //     LOG(INFO) << "There is already have a handler named "
  //               <<  new_callback.func_name_;
  //     return;
  // }
  res.first->second.file_handlers_.emplace_back(new_callback);

  if (file_monitor_thread_ == nullptr) {
    // Note we should start only one monitor thread
    StartMonitorThread();
  }
}

void FileListener::UnregisterFileListener(
    const std::string& file_path, const std::string& handler_name) {
  VLOG(2) << "Unregister a file listener named " << handler_name
          << " on file " << file_path;

  mutex_lock l(lock_);
  const auto& it = listeners_.find(file_path);
  if (it == listeners_.end()) {
    VLOG(2) << "Failed to find the file " << file_path
            << "in the listener list";
    return;
  }
  // There is an assumption that the users will not add the same
  // callback func multiple times.
  callback f = [](const std::string&) {};
  CallbackFunc tar_callback(handler_name, f);

  auto v_it = std::find(it->second.file_handlers_.begin(),
      it->second.file_handlers_.end(), tar_callback);
  if (v_it == it->second.file_handlers_.end()) {
    VLOG(2) << "Failed to find this listener on file " << file_path;
    return;
  }
  it->second.file_handlers_.erase(v_it);
  if (it->second.file_handlers_.size() == 0) {
    // Have removed all handlers corresponding to this file,
    // therefore remove this file from 'listeners_'.
    listeners_.erase(it);
  }

  if (listeners_.size() == 0) {
    // Have removed all listeners, therefore stop the monitor
    // thread. Note that the monitor thread will be created
    // again during next FileListener generation.
    StopMonitorThread();
  }
}

void FileListener::logAllFileListeners(int vlog_level) {
  mutex_lock l(lock_);
  for (const auto& listener : listeners_) {
    for (const auto& handler : listener.second.file_handlers_) {
      VLOG(vlog_level) << "Registered file listener named "
                       << handler.func_name_
                       << " on file path "
                       << listener.first;
    }
  }
}

}  // namespace tensorflow
