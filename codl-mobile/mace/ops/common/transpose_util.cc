
#include "mace/ops/common/transpose_util.h"

namespace mace {
namespace ops {

std::vector<index_t> TransposeUtil::TransposeShape(
    const std::vector<index_t> &input_shape,
    const std::vector<int> &dst_dims) {
  std::vector<index_t> output_shape;
  for (size_t i = 0; i < dst_dims.size(); ++i) {
    output_shape.push_back(input_shape[dst_dims[i]]);
  }
  return output_shape;
}

void TransposeUtil::TransposeTensorShape(
    Tensor *input, const std::vector<int> &dst_dims) {
  const std::vector<index_t> &src_shape = input->shape();
  const std::vector<index_t> shape = TransposeUtil::TransposeShape(src_shape, dst_dims);
  input->Reshape(shape);
}

}  // namespace ops
}  // namespace mace
