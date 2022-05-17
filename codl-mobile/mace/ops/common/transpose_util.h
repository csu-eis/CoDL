
#ifndef MACE_OPS_COMMON_TRANSPOSE_UTIL_H_
#define MACE_OPS_COMMON_TRANSPOSE_UTIL_H_

#include <vector>
#include "mace/core/types.h"
#include "mace/core/tensor.h"

#define DST_DIMS_UNCHANGED     std::vector<int>{0, 1, 2, 3}
#define DST_DIMS_IMAGE_TO_NCHW DST_DIMS_NHWC_TO_NCHW
#define DST_DIMS_IMAGE_TO_OIHW std::vector<int>{1, 0, 2, 3}
#define DST_DIMS_NCHW_TO_IMAGE DST_DIMS_NCHW_TO_NHWC
#define DST_DIMS_NHWC_TO_IMAGE std::vector<int>{3, 2, 0, 1}
#define DST_DIMS_OIHW_TO_IMAGE DST_DIMS_IMAGE_TO_OIHW

namespace mace {
namespace ops {

class TransposeUtil {
public:
  static std::vector<index_t> TransposeShape(
      const std::vector<index_t> &input_shape,
      const std::vector<int> &dst_dims);

  static void TransposeTensorShape(
      Tensor *input, const std::vector<int> &dst_dims);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_TRANSPOSE_UTIL_H_
