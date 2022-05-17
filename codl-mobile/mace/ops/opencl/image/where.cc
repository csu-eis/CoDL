
#include "mace/ops/opencl/image/where.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus WhereKernel::Compute(
    OpContext *context,
    const Tensor *condition,
    const Tensor *X,
    const Tensor *Y,
    Tensor *output) {
  MACE_UNUSED(context);

  int max_case = 0;
  std::vector<index_t> max_shape;
  if (condition->dim_size() > X->dim_size()) {
    max_case = 1;
    max_shape = condition->shape();
  } else {
    max_case = 2;
    max_shape = X->shape();
  }

  if (Y->dim_size() > static_cast<index_t>(max_shape.size())) {
    max_case = 3;
    max_shape = Y->shape();
  }

  VLOG(1) << "max_case " << max_case << ", max_shape " << VectorToString<index_t>(max_shape);

  // NOTE(fucheng): We use Y by default.
  max_case = 3;
  max_shape = Y->shape();
  
#if 0
  std::vector<index_t> output_shape(max_shape[0]);
  Tensor::MappingGuard guard(Y);
  for (index_t i = 0; i < max_shape[0]; i ++) {
    switch (max_case) {
      case 1:
        output_shape[i] = static_cast<index_t>(condition->data<float>()[i]);
        break;
      case 2:
        output_shape[i] = static_cast<index_t>(X->data<float>()[i]);
        break;
      case 3:
        output_shape[i] = static_cast<index_t>(Y->data<int32_t>()[i]);
        break;
    }
  }
#endif

  std::vector<index_t> output_shape = max_shape;

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
