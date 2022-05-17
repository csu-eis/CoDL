
#ifndef MACE_OPS_OPENCL_CONV_3D_H_
#define MACE_OPS_OPENCL_CONV_3D_H_

#include <vector>

#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/conv_pool_3d_util.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {
class OpContext;

namespace ops {
class OpenCLConv3dKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *bias,
      const int *strides,
      const Padding &padding_type,
      const std::vector<int> &padding_data,
      const int *dilations,
      const ActivationType activation,
      const float relux_max_limit,
      const float leakyrelu_coefficient,
      const int winograd_blk_size,
      Tensor *output) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLConv3dKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_CONV_3D_H_
