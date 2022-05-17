
#ifndef MACE_OPS_OPENCL_SCATTER_ND_H_
#define MACE_OPS_OPENCL_SCATTER_ND_H_

#include <vector>

#include "mace/core/tensor.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/public/mace.h"
#include "mace/utils/math.h"

namespace mace {
class OpContext;

namespace ops {
class OpenCLScatterNDKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *indices,
      const Tensor *updates,
      Tensor *output) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLScatterNDKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_SCATTER_ND_H_
