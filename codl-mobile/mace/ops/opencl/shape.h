
#ifndef MACE_OPS_OPENCL_SHAPE_H_
#define MACE_OPS_OPENCL_SHAPE_H_

#include <vector>

#include "mace/core/tensor.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/public/mace.h"
#include "mace/utils/math.h"

namespace mace {
class OpContext;

namespace ops {
class OpenCLShapeKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      Tensor *output) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLShapeKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_SHAPE_H_
