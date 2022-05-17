
#include "mace/ops/opencl/image/transpose.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus TransposeKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const std::vector<int> &dims,
    Tensor *output) {
  MACE_UNUSED(context);
  const std::vector<index_t> input_shape = input->shape();
  std::vector<index_t> output_shape;
  for (size_t i = 0; i < dims.size(); ++i) {
    output_shape.push_back(input_shape[dims[i]]);
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
