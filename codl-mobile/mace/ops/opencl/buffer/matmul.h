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
#ifndef MACE_OPS_OPENCL_BUFFER_MATMUL_H_
#define MACE_OPS_OPENCL_BUFFER_MATMUL_H_

#include "mace/ops/opencl/matmul.h"

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

class MatMulKernel : public OpenCLMatMulKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *A,
      const Tensor *B,
      Tensor *C,
      bool transpose_a,
      bool transpose_b) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
};

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_MATMUL_H_
