
#ifndef MACE_OPS_OPENCL_RESHAPE_H_
#define MACE_OPS_OPENCL_RESHAPE_H_

#include <vector>

#include "mace/core/tensor.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/public/mace.h"
#include "mace/utils/math.h"

namespace mace {
class OpContext;

namespace ops {
class OpenCLReshapeKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *shape,
      Tensor *output) = 0;

  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const std::vector<index_t> &shape,
      Tensor *output) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLReshapeKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_RESHAPE_H_
