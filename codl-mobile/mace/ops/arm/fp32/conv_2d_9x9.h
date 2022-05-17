
#ifndef MACE_OPS_ARM_FP32_CONV_2D_9X9_H_
#define MACE_OPS_ARM_FP32_CONV_2D_9X9_H_

#include <vector>
#include "mace/public/mace.h"
#include "mace/core/tensor.h"
#include "mace/core/op_context.h"
#include "mace/ops/arm/fp32/conv_2d.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

class Conv2dK9x9S1 : public Conv2dBase {
 public:
  Conv2dK9x9S1(const std::vector<int> &paddings, const Padding padding_type)
      : Conv2dBase({1, 1}, {1, 1}, paddings, padding_type) {}
  virtual ~Conv2dK9x9S1() {}

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      Tensor *output) override;
};

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FP32_CONV_2D_9X9_H_
