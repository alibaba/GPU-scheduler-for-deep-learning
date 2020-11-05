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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SESSION_RUN_ACTION_REGISTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SESSION_RUN_ACTION_REGISTRY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// All the parameters used by an SessionRunAction are packaged in
// this struct.
struct SessionRunActionOptions {
  // The DeviceMgr contains each device in 'devices'.
  const std::unique_ptr<const tensorflow::DeviceMgr>*
      device_mgr = nullptr;   // Not owned.

  // The DeviceSet contains the devices known to the system.
  const std::unique_ptr<DeviceSet>* device_set = nullptr;  // Not owned.

  // To store the pointer to the relevant session.
  const void* sess_ptr = nullptr;  // Not owned.

  // To store the unique id of the graph
  // which is used in this session run.
  uint64 graph_id = 0;

  // To store the time duration (in microsecond) of this session run.
  uint64 sess_duration_us = 0;

  // TODO(shiru): Add other parameters needed by SessionRunAction here.
};

// For running some actions before or after each SessionRun
class SessionRunAction {
 public:
  virtual ~SessionRunAction() {}
  virtual Status RunAction(const SessionRunActionOptions& options) = 0;
  void set_name(const string& name) { name_ = name; }
  std::string name() const { return name_; }

 private:
  // The name of the action, which is the same as the inherited
  // class name.
  string name_;
};

// The key is a 'phase' number. Phases are executed in increasing
// order. Within each phase the order of actions is undefined.
typedef std::map<int, std::vector<std::unique_ptr<SessionRunAction>>>
    SessionRunActions;

// A global SessionRunActionRegistry is used to hold all actions.
class SessionRunActionRegistry {
 public:
  // Groups of actions are run in a predetermined order.
  enum Grouping {
    PRE_SESSION_RUN,         // before each RunStep.
    POST_SESSION_RUN,        // right after each RunStep.
  };

  // Add an action to the registry. The 'phase' indicates
  // the running order of the actions inside a group.
  // Note that the actions may be run in parallel, therefore you MUST
  // make sure that all your actions are thread safe!
  // Note that we do NOT guarantee that all your actions will be run as
  // one atomic transaction!
  void Register(Grouping grouping, int phase, SessionRunAction* action);

  // Run all action in grouping, ordered by phase, with the same options.
  Status RunGrouping(Grouping grouping,
                      const SessionRunActionOptions& options);

  // Returns the global registry of session run actions.
  static SessionRunActionRegistry* Global();

  // Prints registered actions for debugging.
  void LogGrouping(Grouping grouping, int vlog_level);
  void LogAllGroupings(int vlog_level);

  // Return the instance of an action.
  SessionRunAction* GetAction(Grouping grouping,
                              int phase,
                              const std::string& action_name);

  // Record whether this graph is a trainable graph.
  void RecordTrainableGraph(const uint64 graph_hash, Graph* g);

  // Return whether should we run actions on this graph.
  bool ShouldRunAction(const uint64 graph_hash);

 private:
  std::map<Grouping, SessionRunActions> groups_;

  // For recording whether this graph is a trainable graph.
  std::unordered_set<uint64> is_trainable_graph_;
};

namespace session_run_action_registration {

class SessionRunActionRegistration {
 public:
  SessionRunActionRegistration(SessionRunActionRegistry::Grouping grouping,
                                int phase,
                                SessionRunAction* action,
                                std::string action_name) {
    action->set_name(action_name);
    SessionRunActionRegistry::Global()->Register(grouping, phase, action);
  }
};

}  // namespace session_run_action_registration

#define REGISTER_SESSION_RUN_ACTION(grouping, phase, action) \
  REGISTER_ACTION_UNIQ_HELPER(__COUNTER__, grouping, phase, action)

#define REGISTER_ACTION_UNIQ_HELPER(ctr, grouping, phase, action) \
  REGISTER_ACTION_UNIQ(ctr, grouping, phase, action)

#define REGISTER_ACTION_UNIQ(ctr, grouping, phase, action)             \
  static ::tensorflow::session_run_action_registration::               \
      SessionRunActionRegistration register_session_run_action_##ctr(  \
          grouping, phase, new action(),                               \
          #action)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SESSION_RUN_ACTION_REGISTRY_H_
