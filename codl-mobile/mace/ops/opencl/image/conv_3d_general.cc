
#include "mace/core/op_context.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/ops/opencl/helper.h"
#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/conv_pool_3d_util.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus Conv3d(OpContext *context,
                  cl::Kernel *kernel,
                  const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  const int *stride,
                  const int *padding,
                  const int *dilations,
                  const ActivationType activation,
                  const float relux_max_limit,
                  const float leakyrelu_coefficient,
                  std::vector<index_t> *prev_input_shape,
                  Tensor *output,
                  uint32_t *kwg_size) {
  MACE_UNUSED(context);
  MACE_UNUSED(kernel);
  MACE_UNUSED(input);
  MACE_UNUSED(filter);
  MACE_UNUSED(bias);
  MACE_UNUSED(stride);
  MACE_UNUSED(padding);
  MACE_UNUSED(dilations);
  MACE_UNUSED(activation);
  MACE_UNUSED(relux_max_limit);
  MACE_UNUSED(leakyrelu_coefficient);
  MACE_UNUSED(prev_input_shape);
  MACE_UNUSED(output);
  MACE_UNUSED(kwg_size);
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
