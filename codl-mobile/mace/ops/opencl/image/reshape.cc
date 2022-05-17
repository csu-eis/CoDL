
#include "mace/ops/opencl/image/reshape.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template<typename T>
void ReshapeKernel::CalcOutputShape(const Tensor *input,
                                    const Tensor *shape,
                                    std::vector<index_t> &output_shape) {
  const index_t num_dims = static_cast<index_t>(output_shape.size());
  //VLOG(1) << "num_dims " << num_dims;
  Tensor::MappingGuard shape_guard(shape);
  // NOTE(fucheng): Must support both int32_t and float.
  const T *shape_data = shape->data<T>();

  int unknown_idx = -1;
  index_t product = 1;
  index_t n = 0;

  for (int i = 0; i < num_dims; ++i) {
    VLOG(1) << "i " << i << ", shape_data " << shape_data[i];
    if (shape_data[i] == -1) {
      MACE_CHECK(unknown_idx == -1, "Only one input size may be -1");
      unknown_idx = i;
      output_shape[i] = 1;
    } else {
      MACE_CHECK(shape_data[i] >= 0, "Shape must be non-negative: ",
                 shape_data[i]);
      // NOTE(fucheng): I observe that NAS model contains
      //                a very large shape value.
      if (shape_data[i] == 0 || shape_data[i] >= 1030880001) {
        MACE_CHECK(i < input->dim_size(),
                   "dims:0 out of input dims' range.");
        n = input->dim(i);
      } else {
        n = shape_data[i];
      }
      output_shape[i] = n;
      product *= n;
    }
  }

  VLOG(1) << "input_shape " << VectorToString<index_t>(input->shape())
          << ", output_shape " << VectorToString<index_t>(output_shape);

  if (unknown_idx != -1) {
    MACE_CHECK(product != 0)
        << "Cannot infer shape if there is zero shape size.";
    const index_t missing = input->size() / product;
#if 0
    LOG(INFO) << "input_size " << input->size()
              << ", product " << product
              << ", missing " << missing;
#endif
    MACE_CHECK(missing * product == input->size())
        << "Input size not match reshaped tensor size";
    output_shape[unknown_idx] = missing;
  }
}

MaceStatus ReshapeKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *shape,
    Tensor *output) {
  MACE_UNUSED(context);
  const index_t num_dims = shape->dim_size() == 0 ? 0 : shape->dim(0);
  MACE_CHECK(num_dims > 0);
  std::vector<index_t> output_shape(num_dims);
  if (shape->dtype() == DataType::DT_FLOAT ||
      shape->dtype() == DataType::DT_HALF) {
    CalcOutputShape<float>(input, shape, output_shape);
  } else if (shape->dtype() == DataType::DT_INT32) {
    CalcOutputShape<int32_t>(input, shape, output_shape);
  }

  return Compute(context, input, output_shape, output);
}

MaceStatus ReshapeKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const std::vector<index_t> &shape,
    Tensor *output) {
  MACE_UNUSED(context);
  MACE_UNUSED(input);
  VLOG(1) << "shape " << VectorToString<index_t>(shape);
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
