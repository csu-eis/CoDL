
#include "mace/ops/opencl/image/gather.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus GatherKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *params,
    const Tensor *indices,
    int axis,
    Tensor *output) {
  MACE_UNUSED(context);
  MACE_UNUSED(input);

  std::vector<index_t> output_shape;
  if (axis < 0) {
    axis += params->dim_size();
  }
  MACE_CHECK(axis >= 0 && axis < params->dim_size(),
             "axis is out of bound: ", axis);
  output_shape.insert(output_shape.end(), params->shape().begin(),
                      params->shape().begin() + axis);
  output_shape.insert(output_shape.end(), indices->shape().begin(),
                      indices->shape().end());
  output_shape.insert(output_shape.end(),
                      params->shape().begin() + (axis + 1),
                      params->shape().end());
  
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
