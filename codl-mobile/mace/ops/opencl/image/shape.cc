
#include "mace/ops/opencl/image/shape.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus ShapeKernel::Compute(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {
  MACE_UNUSED(context);
  std::vector<index_t> output_shape;
  if (input->dim_size() > 0) {
    output_shape = {input->dim_size()};
  } else {
    MACE_NOT_IMPLEMENTED;
  }

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
