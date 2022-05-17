
#ifndef MACE_OPS_OPENCL_BUFFER_DECONV_2D_H_
#define MACE_OPS_OPENCL_BUFFER_DECONV_2D_H_
#include "mace/ops/opencl/deconv_2d.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

class Deconv2dKernel : public OpenCLDeconv2dKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *bias,
      const int *strides,
      const int *padding_data,
      const ActivationType activation,
      const float relux_max_limit,
      const float leakyrelu_coefficient,
      const std::vector<index_t> &output_shape,
      Tensor *output) override;

  MaceStatus ResizeOutputTensor(
      const std::vector<index_t> &output_shape,
      Tensor *output) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_DECONV_2D_H_
