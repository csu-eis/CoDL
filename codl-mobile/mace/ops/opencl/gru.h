
#ifndef MACE_OPS_OPENCL_GRU_H_
#define MACE_OPS_OPENCL_GRU_H_

#include <vector>

#include "mace/core/tensor.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/ops/common/gru_type.h"
#include "mace/public/mace.h"
#include "mace/utils/math.h"

namespace mace {
class OpContext;

namespace ops {
class OpenCLGruKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *X,
      const Tensor *W,
      const Tensor *R,
      const Tensor *B,
      const Tensor *sequence_lens_tensor,
      const Tensor *initial_h_tensor,
      const GruDirection direction,
      const int hidden_size,
      Tensor *Y,
      Tensor *Y_h) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLGruKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_GRU_H_
