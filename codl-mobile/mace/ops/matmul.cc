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

#include "mace/ops/matmul.h"

namespace mace {
namespace ops {

void RegisterMatMul(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::CPU, float16_t);
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_GPU_OP(op_registry, "MatMul", MatMulOp);
#endif
}

}  // namespace ops
}  // namespace mace
