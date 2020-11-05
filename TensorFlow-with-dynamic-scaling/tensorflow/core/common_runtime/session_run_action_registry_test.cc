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

#include "tensorflow/core/common_runtime/session_run_action_registry.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class TestSessionRunActionA : public SessionRunAction {
 public:
  static int count_A;
  Status RunAction(const SessionRunActionOptions& options) override {
    ++count_A;
    return Status::OK();
  }
};

class TestSessionRunActionB : public SessionRunAction {
 public:
  static int count_B;
  Status RunAction(const SessionRunActionOptions& options) override {
    count_B += 2;
    return Status::OK();
  }
};

int TestSessionRunActionA::count_A = 0;
int TestSessionRunActionB::count_B = 0;

REGISTER_SESSION_RUN_ACTION(SessionRunActionRegistry::PRE_SESSION_RUN,
                            1, TestSessionRunActionA);
REGISTER_SESSION_RUN_ACTION(SessionRunActionRegistry::POST_SESSION_RUN,
                            1, TestSessionRunActionA);
REGISTER_SESSION_RUN_ACTION(SessionRunActionRegistry::POST_SESSION_RUN,
                            2, TestSessionRunActionB);

TEST(SessionRunActionRegistry, SessionRunAction) {
  EXPECT_EQ(0, TestSessionRunActionA::count_A);
  EXPECT_EQ(0, TestSessionRunActionB::count_B);
  SessionRunActionOptions options;
  EXPECT_EQ(Status::OK(),
            SessionRunActionRegistry::Global()->RunGrouping(
                SessionRunActionRegistry::POST_SESSION_RUN, options));
  EXPECT_EQ(1, TestSessionRunActionA::count_A);

  EXPECT_EQ(Status::OK(),
            SessionRunActionRegistry::Global()->RunGrouping(
                SessionRunActionRegistry::PRE_SESSION_RUN, options));
  EXPECT_EQ(2, TestSessionRunActionA::count_A);
  EXPECT_EQ(2, TestSessionRunActionB::count_B);
}

}  // namespace tensorflow
