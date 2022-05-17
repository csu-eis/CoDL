
#ifndef MACE_OPS_OPENCL_GEMM_H_
#define MACE_OPS_OPENCL_GEMM_H_

#include <vector>

#include "mace/public/mace.h"
#include "mace/utils/math.h"
#include "mace/ops/common/activation_type.h"

namespace mace {

class OpContext;
class Tensor;

namespace ops {

class OpenCLGemmKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *lhs,
      const Tensor *rhs,
      const Tensor *bias,
      const ActivationType activation,
      const float relux_max_limit,
      const float leakyrelu_coefficient,
      Tensor *output) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLGemmKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_GEMM_H_
