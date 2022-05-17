
#ifndef MACE_OPS_OPENCL_TRANSPOSE_H_
#define MACE_OPS_OPENCL_TRANSPOSE_H_

#include <vector>

#include "mace/core/tensor.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/public/mace.h"
#include "mace/utils/math.h"

namespace mace {
class OpContext;

namespace ops {
class OpenCLTransposeKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const std::vector<int> &dims,
      Tensor *output) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLTransposeKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_TRANSPOSE_H_
