// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_ARM_FP32_DECONV_2D_2X2_H_
#define MACE_OPS_ARM_FP32_DECONV_2D_2X2_H_

#include <vector>
#include <memory>

#include "mace/public/mace.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/core/op_context.h"
#include "mace/ops/arm/fp32/deconv_2d.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

class Deconv2dK2x2S1 : public Deconv2dBase {
 public:
  Deconv2dK2x2S1(const std::vector<int> &paddings,
                 const std::vector<int> &output_paddings,
                 const Padding padding_type,
                 const FrameworkType framework_type)
      : Deconv2dBase({1, 1}, {1, 1}, paddings, output_paddings, padding_type, framework_type) {}
  virtual ~Deconv2dK2x2S1() {}

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *output_shape,
      Tensor *output) override;
};

class Deconv2dK2x2S2 : public Deconv2dBase {
 public:
  Deconv2dK2x2S2(const std::vector<int> &paddings,
                 const std::vector<int> &output_paddings,
                 const Padding padding_type,
                 const FrameworkType framework_type)
      : Deconv2dBase({2, 2}, {1, 1}, paddings, output_paddings, padding_type, framework_type) {}
  virtual ~Deconv2dK2x2S2() {}

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *output_shape,
      Tensor *output) override;
};

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FP32_DECONV_2D_2X2_H_
