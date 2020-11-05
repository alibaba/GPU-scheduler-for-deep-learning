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

#include "tensorflow/contrib/resource_management/file_listener.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static int count_1;
static int count_2;
static int count_3;
static int last_read_111;
static int last_read_211;
static int last_read_212;
static int last_read_221;
static int last_read_222;

#define REGISTER_FILE_LISTENER_WITH_LOG(file_path, callback_name, test_num, \
                               file_num, callback_num, add_time)            \
  FileListener::GlobalFileListener()->                                      \
      RegisterFileListener(file_path, callback_name,                        \
          [](const std::string& str) {                                      \
                std::cout << " I'm the callback_"                           \
                          << #callback_num                                  \
                          << " on MemoryManagement.ini : "                  \
                          << "file content: "                               \
                          << str << '\n';                                   \
                if (str != "") {                                            \
                  int i = std::stoi(str);                                   \
                  if (last_read_##test_num##file_num##callback_num != i) {  \
                    int j = add_time;                                       \
                    while (j > 0) {                                         \
                      count_##file_num += i;                                \
                      --j;                                                  \
                    }                                                       \
                    last_read_##test_num##file_num##callback_num = i;       \
                  }                                                         \
                }                                                           \
          })                                                                \

#define REGISTER_FILE_LISTENER(file_path, callback_name, test_num,          \
                               file_num, callback_num, add_time)            \
  FileListener::GlobalFileListener()->                                      \
      RegisterFileListener(file_path, callback_name,                        \
          [](const std::string& str) {                                      \
                if (str != "") {                                            \
                  int i = std::stoi(str);                                   \
                  if (last_read_##test_num##file_num##callback_num != i) {  \
                    int j = add_time;                                       \
                    while (j > 0) {                                         \
                      count_##file_num += i;                                \
                      --j;                                                  \
                    }                                                       \
                    last_read_##test_num##file_num##callback_num = i;       \
                  }                                                         \
                }                                                           \
          })                                                                \


TEST(FileListener, RegisterFileListener) {
  // FileListener::GlobalFileListener()->logAllFileListeners(2);
  std::ofstream file_1;
  const char* file_1_name = "MemoryManagement.ini";

  REGISTER_FILE_LISTENER("MemoryManagement.ini", "Memory_Manage", 1, 1, 1, 1);

  // FileListener::GlobalFileListener()->logAllFileListeners(2);

  sleep(5);

  file_1.open(file_1_name);
  file_1 << "100" << std::endl;
  file_1.close();
  sleep(5);
  EXPECT_EQ(100, count_1);

  file_1.open(file_1_name);
  file_1 << "200" << std::endl;
  file_1.close();
  sleep(5);
  EXPECT_EQ(300, count_1);

  FileListener::GlobalFileListener()->
    UnregisterFileListener("MemoryManagement.ini", "Memory_Manage");

  // FileListener::GlobalFileListener()->logAllFileListeners(2);

  remove(file_1_name);
}

TEST(FileListener, RegisterMultipleFileListener) {
  count_1 = 0;
  count_2 = 0;
  count_3 = 0;

  const char* file_1_name = "MemoryManagement.ini";
  const char* file_2_name = "GPUManagement.ini";
  const char* file_3_name = "CPUManagement.ini";

  REGISTER_FILE_LISTENER(file_1_name, "Memory_Manage_1", 2, 1, 1, 1);
  std::ofstream file_1;
  file_1.open(file_1_name);
  file_1 << "100" << std::endl;
  file_1.close();
  sleep(2);
  EXPECT_EQ(100, count_1);

  REGISTER_FILE_LISTENER(file_1_name, "Memory_Manage_2", 2, 1, 2, 2);

  REGISTER_FILE_LISTENER(file_2_name, "GPU_Manage_1", 2, 2, 1, 1);

  REGISTER_FILE_LISTENER(file_2_name, "GPU_Manage_2", 2, 2, 2, 2);

  // FileListener::GlobalFileListener()->logAllFileListeners(2);

  file_1.open(file_1_name);
  file_1 << "200" << std::endl;
  file_1.close();
  sleep(2);
  EXPECT_EQ(700, count_1);

  file_1.open(file_1_name);
  file_1 << "300" << std::endl;
  file_1.close();
  sleep(2);
  EXPECT_EQ(1600, count_1);

  std::ofstream file_2;
  file_2.open(file_2_name);
  file_2 << "2" << std::endl;
  file_2.close();
  sleep(2);
  EXPECT_EQ(6, count_2);

  FileListener::GlobalFileListener()->
    UnregisterFileListener(file_1_name, "Memory_Manage_1");

  file_1.open(file_1_name);
  file_1 << "400" << std::endl;
  file_1.close();
  sleep(2);
  EXPECT_EQ(2400, count_1);

  last_read_211 = 0;
  REGISTER_FILE_LISTENER(file_1_name, "Memory_Manage_1", 2, 1, 1, 1);

  file_1.open(file_1_name);
  file_1 << "500" << std::endl;
  file_1.close();
  sleep(2);
  EXPECT_EQ(3900, count_1);

  FileListener::GlobalFileListener()->
    UnregisterFileListener(file_1_name, "Memory_Manage_1");
  FileListener::GlobalFileListener()->
    UnregisterFileListener(file_1_name, "Memory_Manage_2");

  file_1.open(file_1_name);
  file_1 << "600" << std::endl;
  file_1.close();
  sleep(2);
  EXPECT_EQ(3900, count_1);

  last_read_212 = 0;
  REGISTER_FILE_LISTENER(file_1_name, "Memory_Manage_2", 2, 1, 2, 2);

  sleep(2);
  EXPECT_EQ(5100, count_1);

  file_1.open(file_1_name);
  file_1 << "700" << std::endl;
  file_1.close();
  sleep(2);
  EXPECT_EQ(6500, count_1);

  file_2.open(file_2_name);
  file_2 << "3" << std::endl;
  file_2.close();
  sleep(2);
  EXPECT_EQ(15, count_2);

  FileListener::GlobalFileListener()->
    UnregisterFileListener(file_1_name, "Memory_Manage_2");
  FileListener::GlobalFileListener()->
    UnregisterFileListener(file_2_name, "GPU_Manage_2");
  FileListener::GlobalFileListener()->
    UnregisterFileListener(file_2_name, "GPU_Manage_1");
  // FileListener::GlobalFileListener()->logAllFileListeners(2);

  remove(file_1_name);
  remove(file_2_name);
}

}  // namespace tensorflow
