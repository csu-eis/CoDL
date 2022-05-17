
#include "mace/ops/opencl/image/gru.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus GruKernel::Compute(
    OpContext *context,
    const Tensor *X,
      const Tensor *W,
      const Tensor *R,
      const Tensor *B,
      const Tensor *sequence_lens_tensor,
      const Tensor *initial_h_tensor,
      const GruDirection direction,
      const int hidden_size,
      Tensor *Y,
      Tensor *Y_h) {
  MACE_UNUSED(context);
  MACE_UNUSED(W);
  MACE_UNUSED(R);
  MACE_UNUSED(B);
  MACE_UNUSED(initial_h_tensor);

  const index_t batch_size = X->dim(1);
  const index_t num_directions
      = (direction == GruDirection::GD_BIDIRECTIONAL) ? 2 : 1;
  index_t seq_length = batch_size;
  if (sequence_lens_tensor != nullptr) {
    Tensor::MappingGuard guard(sequence_lens_tensor);
    seq_length = sequence_lens_tensor->data<int32_t>()[0];
  }
  
  std::vector<index_t> Y_shape;
  Y_shape.push_back(seq_length);
  Y_shape.push_back(num_directions);
  Y_shape.push_back(batch_size);
  Y_shape.push_back(hidden_size);
  
  std::vector<index_t> Y_h_shape;
  Y_h_shape.push_back(num_directions);
  Y_h_shape.push_back(batch_size);
  Y_h_shape.push_back(hidden_size);

  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(Y_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &output_image_shape);
  MACE_RETURN_IF_ERROR(Y->ResizeImage(Y_shape, output_image_shape));
  OpenCLUtil::CalImage2DShape(Y_h_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &output_image_shape);
  MACE_RETURN_IF_ERROR(Y_h->ResizeImage(Y_h_shape, output_image_shape));

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
