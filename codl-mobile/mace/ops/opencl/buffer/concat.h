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
#ifndef MACE_OPS_OPENCL_BUFFER_CONCAT_H_
#define MACE_OPS_OPENCL_BUFFER_CONCAT_H_

#include "mace/ops/opencl/concat.h"

#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {
namespace concat {
MaceStatus Concat2(OpContext *context,
                   cl::Kernel *kernel,
                   const Tensor *input0,
                   const Tensor *input1,
                   std::vector<index_t> *prev_input_shape,
                   Tensor *output,
                   uint32_t *kwg_size);

MaceStatus ConcatN(OpContext *context,
                   cl::Kernel *kernel,
                   const std::vector<const Tensor *> &input_list,
                   Tensor *output,
                   uint32_t *kwg_size);
}  // namespace concat

class ConcatKernel : public OpenCLConcatKernel {
 public:
  ConcatKernel() {}
  MaceStatus Compute(
      OpContext *context,
      const std::vector<const Tensor *> &input_list,
      const int32_t axis,
      Tensor *output) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_CONCAT_H_
