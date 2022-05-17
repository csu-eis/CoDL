
#include "mace/ops/opencl/image/constant_of_shape.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus ConstantOfShapeKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const float value,
    Tensor *output) {
  MACE_UNUSED(context);
  MACE_UNUSED(value);

  MACE_CHECK(input->dim_size() == 1 && input->dim(0) == 1);

  std::vector<index_t> output_shape(1, input->data<int32_t>()[0]);
  //for (size_t i = 0; i < output_shape.size(); i ++) {
  //  output_shape[i] = static_cast<index_t>(value);
  //}
  VLOG(1) << "output_shape " << VectorToString<index_t>(output_shape);

  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
