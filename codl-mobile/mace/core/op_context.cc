// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/core/op_context.h"

namespace mace {

OpContext::OpContext(Workspace *ws, Device *device)
    : device_(device), cpu_device_(nullptr),
      ws_(ws), future_(nullptr),
      part_run_config_(nullptr),
      partition_configer_(nullptr),
      op_runtime_mode_(OpRuntimeMode::NONE),
      dura_collector_(nullptr) {}

OpContext::~OpContext() = default;

int OpContext::op_idx() const {
  return op_idx_;
}

void OpContext::set_op_idx(const int i) {
  op_idx_ = i;
}

void OpContext::set_device(Device *device) {
  device_ = device;
}

Device* OpContext::device() const {
  return device_;
}

void OpContext::set_cpu_device(Device *device) {
  cpu_device_ = device;
}

Device* OpContext::cpu_device() const {
  return cpu_device_;
}

Workspace* OpContext::workspace() const {
  return ws_;
}

void OpContext::set_future(StatsFuture *future) {
  future_ = future;
}

StatsFuture *OpContext::future() const {
  return future_;
}

void OpContext::set_part_run_config(PartitionRunConfig *config) {
  part_run_config_ = config;
}

PartitionRunConfig *OpContext::part_run_config() const {
  return part_run_config_;
}

void OpContext::set_partition_configer(PartitionConfiger *configer) {
  partition_configer_ = configer;
}

PartitionConfiger *OpContext::partition_configer() const {
  return partition_configer_;
}

void OpContext::set_op_runtime_mode(OpRuntimeMode mode) {
  op_runtime_mode_ = mode;
}

OpRuntimeMode OpContext::op_runtime_mode() const {
  return op_runtime_mode_;
}

void OpContext::set_dura_collector(DurationCollector<double> *dura_collector) {
  dura_collector_ = dura_collector;
}

DurationCollector<double> *OpContext::dura_collector() const {
  return dura_collector_;
}

}  // namespace mace
