
#include "mace/ops/opencl/image/slice.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus SliceKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const std::vector<int> &axes,
    const std::vector<int> &starts,
    const std::vector<int> &ends,
    Tensor *output) {
  MACE_UNUSED(context);

  const index_t rank = input->dim_size();
  MACE_CHECK(rank >= 1) << "The input dim size should >= 1";
  const index_t input_dim = input->dim(rank - 1);
  MACE_CHECK(starts.size() == 1 && ends.size() == 1 && axes.size() == 1,
             "only support slicing at one axis.");
  MACE_CHECK(axes[0] == -1 || axes[0] == rank - 1,
             "only support slicing at the last axis.");
  MACE_CHECK(starts[0] < input_dim && starts[0] >= 0
             && ends[0] >= 0 && ends[0] <= input_dim)
      << "The starts and ends caused over range error.";
  const index_t output_dim = ends[0] - starts[0];
  MACE_CHECK(output_dim >= 0, "output_dim should >= 0");

  std::vector<index_t> output_shape = input->shape();
  output_shape[rank - 1] = output_dim;

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
