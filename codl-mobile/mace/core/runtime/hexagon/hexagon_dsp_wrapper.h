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

#ifndef MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DSP_WRAPPER_H_
#define MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DSP_WRAPPER_H_

#include <map>
#include <string>
#include <vector>

#include "mace/core/runtime/hexagon/hexagon_control_wrapper.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"
#include "third_party/nnlib/hexagon_nn.h"

namespace mace {

class HexagonDSPWrapper : public HexagonControlWrapper {
 public:
  HexagonDSPWrapper();
  ~HexagonDSPWrapper();

  int GetVersion() override;
  bool Config() override;
  bool Init() override;
  bool Finalize() override;
  bool SetupGraph(const NetDef &net_def,
                  const unsigned char *model_data,
                  const index_t model_data_size) override;
  bool ExecuteGraph(const Tensor &input_tensor,
                    Tensor *output_tensor) override;
  bool ExecuteGraphNew(const std::map<std::string, Tensor*> &input_tensors,
                       std::map<std::string, Tensor*> *output_tensors) override;
  bool TeardownGraph() override;
  void PrintLog() override;
  void PrintGraph() override;
  void GetPerfInfo() override;
  void ResetPerfInfo() override;
  void PrintMemStats() override;
  void SetDebugLevel(int level) override;

  static bool SetPower(HexagonNNCornerType corner,
                       bool dcvs_enable,
                       int latency);
  static bool RequestUnsignedPD();

 private:
  uint64_t GetLastExecuteCycles();
  void PrintOpNodeInfo(hexagon_nn_op_node &node);

  bool log_execute_time_;
  bool is_setup_;
  std::vector<InOutInfo> input_info_;
  std::vector<InOutInfo> output_info_;
  MACE_DISABLE_COPY_AND_ASSIGN(HexagonDSPWrapper);
};
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DSP_WRAPPER_H_