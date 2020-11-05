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

#ifndef TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_FILE_LISTENER_H_
#define TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_FILE_LISTENER_H_

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>
#include <vector>

#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// For describing a file and storing the status of this file.
class FileInfo {
 public:
  explicit FileInfo(const std::string& file_path);

  // For Checking if the target file has been modified
  // since last checking.
  bool IsFileChanged();

  // For loading all content of the file.
  bool LoadFile(std::string* file_content);

  const std::string FilePath() const { return path_; }

 private:
  bool UpdateFileInfo(const struct stat& file_stat,
                      unsigned char* cur_md5);

  constexpr static int kMd5DigestLength = 16;

  // File status.
  timespec last_modified_time_;
  timespec last_checked_time_;
  off_t file_size_;
  unsigned char md5_[kMd5DigestLength];
  // Compatible with both relative path and absolute path.
  const std::string path_;
};

// Note that the callback func MUST recive a 'const std::string&'
// parameter to load the file content.
using callback = std::function<void(const std::string&)>;

// Note that we use 'func_name_' to uniquely mark a
// callback func.
class CallbackFunc {
 public:
  CallbackFunc(const std::string& name, callback func)
      : func_(func), func_name_(name) {}

  bool operator==(const CallbackFunc& other) const {
    return this->func_name_ == other.func_name_;
  }

  callback func_;
  std::string func_name_;
};

// For triggering some handlers (callback func) when the specific
// file changed.
class FileListener {
 public:
  // Users can register multiple handlers on one file.
  using handlers = std::vector<CallbackFunc>;

  // Register a handler that will be triggered when the file named
  // file_name changed.
  // NOTE: the handler may be invoked MULTIPLE TIMES during
  // file modifying due to the modification is NOT atomic !!!
  void RegisterFileListener(const std::string& file_path,
                            const std::string& handler_name,
                            callback callback_func);

  // Unregister a handler.
  void UnregisterFileListener(const std::string& file_path,
                              const std::string& handler_name);

  // Returns the global registry of file listeners.
  static FileListener* GlobalFileListener();

  // Prints registered file listeners for debugging.
  void logAllFileListeners(int vlog_level);

 private:
  FileListener()
      : stop_running_(false), file_monitor_thread_(nullptr) {}

  ~FileListener() {
    StopMonitorThread();
  }

  // Start a new thread to execute the monitor func.
  void StartMonitorThread();

  // Stop the monitor thread.
  void StopMonitorThread();

  // The monitor func.
  void FileMonitor();

  // Default sleep time after each time that checking file status
  constexpr static size_t kSleepIntervalInUs = 1000000UL;   // 1s

  // In order to determine if we need to stop the running
  // of the monitor thread.
  std::atomic<bool> stop_running_;

  // There is at most one monitor thread running.
  std::thread* file_monitor_thread_;

  mutable mutex lock_;

  struct InfoAndHandlers {
    FileInfo file_info_;
    handlers file_handlers_;
  };

  std::map<std::string, InfoAndHandlers> listeners_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_RESOURCE_MANAGEMENT_FILE_LISTENER_H_
