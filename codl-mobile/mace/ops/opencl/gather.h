
#ifndef MACE_OPS_OPENCL_GATHER_H_
#define MACE_OPS_OPENCL_GATHER_H_

#include <vector>

#include "mace/core/tensor.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/public/mace.h"
#include "mace/utils/math.h"

namespace mace {
class OpContext;

namespace ops {
class OpenCLGatherKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *params,
      const Tensor *indices,
      int axis,
      Tensor *output) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLGatherKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_GATHER_H_
