
#include "mace/ops/opencl/image/unsqueeze.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus UnsqueezeKernel::Compute(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {
  MACE_UNUSED(context);
  MACE_CHECK(!axis_.empty(), "Unsqueeze op should have axis values.");
  std::vector<index_t> output_shape = input->shape();
  for (size_t i = 0; i < axis_.size(); ++i) {
    MACE_CHECK(axis_[i] >= 0, "axis's value should be non-negative.");
    output_shape.insert(output_shape.begin() + axis_[i], 1);
  }

  VLOG(2) << "output_shape " << VectorToString<index_t>(output_shape);

  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(output_shape,
                              OpenCLBufferType::IN_OUT_CHANNEL,
                              &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, image_shape));

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
