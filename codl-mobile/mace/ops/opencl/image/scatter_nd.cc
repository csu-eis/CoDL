
#include "mace/ops/opencl/image/scatter_nd.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus ScatterNDKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *indices,
    const Tensor *updates,
    Tensor *output) {
  MACE_UNUSED(context);
  MACE_UNUSED(indices);
  MACE_UNUSED(updates);
  std::vector<index_t> output_shape(input->shape());

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
