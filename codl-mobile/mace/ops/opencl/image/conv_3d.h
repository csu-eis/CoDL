
#ifndef MACE_OPS_OPENCL_IMAGE_CONV_3D_H_
#define MACE_OPS_OPENCL_IMAGE_CONV_3D_H_

#include "mace/ops/opencl/conv_3d.h"

#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

extern MaceStatus Conv3d(OpContext *context,
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
                         uint32_t *kwg_size);

class Conv3dKernel : public OpenCLConv3dKernel {
 public:
  MaceStatus Compute(
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
      const int wino_blk_size,
      Tensor *output) override;

 private:
  cl::Kernel kernels_[3];
  uint32_t kwg_size_[3];
  std::vector<index_t> input_shape_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_CONV_3D_H_
