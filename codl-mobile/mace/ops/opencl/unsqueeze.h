
#ifndef MACE_OPS_OPENCL_UNSQUEEZE_H_
#define MACE_OPS_OPENCL_UNSQUEEZE_H_

#include <vector>

#include "mace/public/mace.h"
#include "mace/utils/math.h"

namespace mace {

class OpContext;
class Tensor;

namespace ops {
class OpenCLUnsqueezeKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      Tensor *output) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLUnsqueezeKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_UNSQUEEZE_H_
