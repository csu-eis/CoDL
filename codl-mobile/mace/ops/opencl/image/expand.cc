
#include "mace/ops/opencl/image/expand.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus ExpandKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *shape,
    Tensor *output) {
  VLOG(1) << "input_shape " << VectorToString<index_t>(input->shape());
  //shape->DebugPrint();

  Tensor::MappingGuard shape_guard(shape);
  std::vector<index_t> output_shape =
      std::vector<index_t>(shape->data<float>(),
                           shape->data<float>() + shape->size());

  VLOG(1) << "output_shape " << VectorToString<index_t>(output_shape);

  return Compute(context, input, output_shape, output);
}

MaceStatus ExpandKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const std::vector<index_t> &shape,
    Tensor *output) {
  MACE_UNUSED(context);
  MACE_UNUSED(input);
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(shape, image_shape));
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
