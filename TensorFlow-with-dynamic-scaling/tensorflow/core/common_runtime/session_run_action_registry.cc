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
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

bool IsTrainableGraph(Graph* g) {
  for (auto n : g->nodes()) {
    if (n->name().find("gradients") != std::string::npos) {
      return true;
    }
  }
  return false;
}

}  // namespace

// static
SessionRunActionRegistry* SessionRunActionRegistry::Global() {
  static SessionRunActionRegistry global_action_registry;
  return &global_action_registry;
}

void SessionRunActionRegistry::Register(
    Grouping grouping, int phase, SessionRunAction* action) {
  VLOG(2) << "Register session run action " << action->name();
  groups_[grouping][phase].emplace_back(action);
}

SessionRunAction* SessionRunActionRegistry::GetAction(
    Grouping grouping, int phase, const std::string& action_name) {
  auto tar_group = groups_.find(grouping);
  if (tar_group == groups_.end()) {
    return nullptr;
  }
  auto tar_phase = tar_group->second.find(phase);
  if (tar_phase == tar_group->second.end()) {
    return nullptr;
  }
  for (auto& tar_action : tar_phase->second) {
    if (tar_action->name() == action_name) {
      return tar_action.get();
    }
  }
  return nullptr;
}

void SessionRunActionRegistry::RecordTrainableGraph(
    const uint64 graph_hash, Graph* g) {
  if (IsTrainableGraph(g)) {
    // No need to use a mutex to protect this func since
    // the MasterSession::StartStep is thread safe.
    is_trainable_graph_.emplace(graph_hash);
  }
}

bool SessionRunActionRegistry::ShouldRunAction(
    const uint64 graph_hash) {
  return is_trainable_graph_.find(graph_hash) != is_trainable_graph_.end();
}

Status SessionRunActionRegistry::RunGrouping(
    Grouping grouping, const SessionRunActionOptions& options) {
  auto group = groups_.find(grouping);
  if (group != groups_.end()) {
    for (auto& phase : group->second) {
      for (auto& action : phase.second) {
        Status s = action->RunAction(options);
        if (!s.ok()) return s;
      }
    }
  }
  return Status::OK();
}

void SessionRunActionRegistry::LogGrouping(Grouping grouping,
                                           int vlog_level) {
  auto group = groups_.find(grouping);
  if (group == groups_.end()) {
    return;
  }
  for (auto& phase : group->second) {
    for (auto& action : phase.second) {
      VLOG(vlog_level) << "Registered session run action grouping "
                        << grouping
                        << " phase " << phase.first
                        << " action " << action->name();
    }
  }
}

void SessionRunActionRegistry::LogAllGroupings(int vlog_level) {
  for (auto group = groups_.begin(); group != groups_.end(); ++group) {
    LogGrouping(group->first, vlog_level);
  }
}

}  // namespace tensorflow
