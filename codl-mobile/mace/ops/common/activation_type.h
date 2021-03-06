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

#ifndef MACE_OPS_COMMON_ACTIVATION_TYPE_H_
#define MACE_OPS_COMMON_ACTIVATION_TYPE_H_

#include <string>

namespace mace {
namespace ops {

enum ActivationType {
  NOOP = 0,
  RELU = 1,
  RELUX = 2,
  PRELU = 3,
  TANH = 4,
  SIGMOID = 5,
  LEAKYRELU = 6,
};

inline std::string ActivationTypeToString(const ActivationType activation) {
  switch (activation) {
    case NOOP:
      return "NOOP";
    case RELU:
      return "RELU";
    case RELUX:
      return "RELUX";
    case TANH:
      return "TANH";
    case SIGMOID:
      return "SIGMOID";
    case LEAKYRELU:
      return "LEAKYRELU";
    default:
      return "UNKNOWN";
  }
}

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_ACTIVATION_TYPE_H_
