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

#ifndef MACE_CORE_NET_H_
#define MACE_CORE_NET_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

#include "mace/core/operator.h"
#include "mace/core/operator_chain.h"
#include "mace/core/part_ratio_predictor.h"

namespace mace {

class RunMetadata;
class Workspace;
class MemoryOptimizer;

class NetOpStatInfo {
public:
  inline std::vector<int64_t> &macs_info() {
    return macs_info_;
  }

  inline std::vector<int64_t> &flops_info() {
    return flops_info_;
  }

private:
  std::vector<int64_t> macs_info_;
  std::vector<int64_t> flops_info_;
};

class NetBase {
 public:
  NetBase() noexcept = default;
  virtual ~NetBase() = default;

  virtual MaceStatus Init() = 0;

  virtual MaceStatus Run(RunMetadata *run_metadata = nullptr,
                         PartitionRunConfig *part_run_config = nullptr,
                         MaceRuntimeStatistics *runtime_stat = nullptr) = 0;

  virtual int GetPartLayerCount() = 0;

  virtual MaceStatus GetPartRatioPredictInputData(float **in_data_ptr) = 0;

  virtual MaceStatus UpdatePartRatio(const float *values) = 0;

 protected:
  MACE_DISABLE_COPY_AND_ASSIGN(NetBase);
};

class SerialNet : public NetBase {
 public:
  SerialNet(const OpRegistryBase *op_registry,
            const NetDef *net_def,
            Workspace *ws,
            Device *target_device,
            MemoryOptimizer * mem_optimizer);

  ~SerialNet();

  MaceStatus Init() override;

  MaceStatus Run(RunMetadata *run_metadata = nullptr,
                 PartitionRunConfig *part_run_config = nullptr,
                 MaceRuntimeStatistics *runtime_stat = nullptr) override;

  int GetPartLayerCount() override {
    return partition_configer_.value_size();
  }

  MaceStatus GetPartRatioPredictInputData(float **in_data_ptr) override;

  MaceStatus UpdatePartRatio(const float *values) override {
    partition_configer_.Update(nullptr, values);
    return MaceStatus::MACE_SUCCESS;
  }

  void SetPartRatioPredictorRegistry(PartRatioPredictorRegistryBase *registry) {
    MACE_CHECK(pr_predictor_registry_ == nullptr);
    pr_predictor_registry_.reset(registry);
  }

 protected:
  MaceStatus PredictPartRatio(OpContext *context);

  MaceStatus ShowOpFeatures(const std::string &op_type);

  MaceStatus BuildOpChain();

  bool is_first_time_run_;
  bool run_conv2d_only_enabled_;
  Workspace *ws_;
  Device *target_device_;
  std::unique_ptr<Device> cpu_device_;  // CPU is base device.
  std::vector<std::unique_ptr<Operation> > operators_;
  std::vector<std::shared_ptr<OperatorChain>> op_chains_;
  PartitionConfiger partition_configer_;
  std::unique_ptr<PartRatioPredictorRegistryBase> pr_predictor_registry_;

  MACE_DISABLE_COPY_AND_ASSIGN(SerialNet);
};

}  // namespace mace

#endif  // MACE_CORE_NET_H_
