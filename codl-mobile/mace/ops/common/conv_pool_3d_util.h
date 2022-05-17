
#ifndef MACE_OPS_COMMON_CONV_POOL_3D_UTIL_H_
#define MACE_OPS_COMMON_CONV_POOL_3D_UTIL_H_

#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {

void Calc3dOutputSize(const index_t *input_shape,   // NHWC
                      const index_t *filter_shape,  // OIHW
                      const int *padding_size,
                      const int *dilations,
                      const int *strides,
                      const RoundType round_type,
                      index_t *output_shape);

void Calc3dNCHWOutputSize(const index_t *input_shape,   // NCHW
                          const index_t *filter_shape,  // OIHW
                          const int *padding_size,
                          const int *dilations,
                          const int *strides,
                          const RoundType round_type,
                          index_t *output_shape);

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_CONV_POOL_3D_UTIL_H_
