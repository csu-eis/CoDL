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

#ifndef MACE_CORE_OP_CONTEXT_H_
#define MACE_CORE_OP_CONTEXT_H_

#include "mace/core/device.h"
#include "mace/core/workspace.h"
#include "mace/core/future.h"
#include "mace/core/partition_configer.h"
#include "mace/utils/op_delay_tool.h"

namespace mace {

enum class OpRuntimeMode {
  NONE = 0,
  WARMING = 1,
  TUNING = 2,
  RUNNING = 3,
};

class OpContext {
 public:
  OpContext(Workspace *ws, Device *device);
  ~OpContext();
  int op_idx() const;
  void set_op_idx(const int i);
  void set_device(Device *device);
  Device *device() const;
  void set_cpu_device(Device *device);
  Device* cpu_device() const;
  Workspace *workspace() const;

  void set_future(StatsFuture *future);
  StatsFuture *future() const;

  void set_part_run_config(PartitionRunConfig *config);
  PartitionRunConfig *part_run_config() const;

  void set_partition_configer(PartitionConfiger *configer);
  PartitionConfiger *partition_configer() const;

  void set_op_runtime_mode(OpRuntimeMode mode);
  OpRuntimeMode op_runtime_mode() const;

  void set_dura_collector(DurationCollector<double> *dura_collector);
  DurationCollector<double> *dura_collector() const;

 private:
  int op_idx_;
  Device *device_;
  Device *cpu_device_;
  Workspace *ws_;
  StatsFuture *future_;
  PartitionRunConfig *part_run_config_;
  PartitionConfiger *partition_configer_;
  OpRuntimeMode op_runtime_mode_;
  DurationCollector<double> *dura_collector_;
  
  // metadata
};

}  // namespace mace
#endif  // MACE_CORE_OP_CONTEXT_H_
