
#ifndef MACE_OPS_CONV_POOL_3D_BASE_H_
#define MACE_OPS_CONV_POOL_3D_BASE_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {

class ConvPool3dOpBase : public Operation {
 public:
  explicit ConvPool3dOpBase(OpConstructContext *context)
      : Operation(context),
        strides_(Operation::GetRepeatedArgs<int>("strides", {1, 1, 1})),
        padding_type_(static_cast<Padding>(Operation::GetOptionalArg<int>(
            "padding", static_cast<int>(SAME)))),
        paddings_(Operation::GetRepeatedArgs<int>("padding_values", {0, 0, 0})),
        dilations_(Operation::GetRepeatedArgs<int>("dilations", {1, 1, 1})) {}

 protected:
  std::vector<int> strides_;
  Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CONV_POOL_3D_BASE_H_
