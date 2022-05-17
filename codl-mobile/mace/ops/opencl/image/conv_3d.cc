
#include "mace/ops/opencl/image/conv_3d.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus Conv3dKernel::Compute(
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
    Tensor *output) {
  MACE_UNUSED(padding_type);
  MACE_UNUSED(padding_data);
  MACE_UNUSED(wino_blk_size);
  index_t kernel_h = filter->dim(2);
  index_t kernel_w = filter->dim(3);
  if (dilations[0] > 1 && (strides[0] > 1 || kernel_h == 1)) {
    LOG(WARNING) << "OpenCL conv3d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides[0] << "x" << strides[1]
                 << ",dilations " << dilations[0] << "x" << dilations[1]
                 << " is not implemented yet.";
    MACE_NOT_IMPLEMENTED;
  }

  // Reshape output
  std::vector<index_t> output_shape(5);
  std::vector<int> paddings(3, 0);
  Calc3dNCHWOutputSize(input->shape().data(), filter->shape().data(),
                       paddings.data(), dilations, strides,
                       RoundType::FLOOR, output_shape.data());
  VLOG(1) << "output_shape " << VectorToString(output_shape);

  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  std::function<MaceStatus()> conv_func;

  conv_func = [&]() -> MaceStatus {
    return Conv3d(context,
                  &kernels_[0],
                  input,
                  filter,
                  bias,
                  strides,
                  paddings.data(),
                  dilations,
                  activation,
                  relux_max_limit,
                  leakyrelu_coefficient,
                  &input_shape_,
                  output,
                  &kwg_size_[0]);
  };

  return conv_func();
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
