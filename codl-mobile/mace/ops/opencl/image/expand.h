
#ifndef MACE_OPS_OPENCL_IMAGE_EXPAND_H_
#define MACE_OPS_OPENCL_IMAGE_EXPAND_H_

#include "mace/ops/opencl/expand.h"

#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class ExpandKernel : public OpenCLExpandKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *shape,
      Tensor *output) override;

  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const std::vector<index_t> &shape,
      Tensor *output) override;

 private:
  //cl::Kernel kernel_;
  //uint32_t kwg_size_;
  //std::vector<index_t> input_shape_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_EXPAND_H_
