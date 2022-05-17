
#ifndef MACE_OPS_OPENCL_IMAGE_WHERE_H_
#define MACE_OPS_OPENCL_IMAGE_WHERE_H_

#include "mace/ops/opencl/where.h"

#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class WhereKernel : public OpenCLWhereKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *condition,
      const Tensor *X,
      const Tensor *Y,
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

#endif  // MACE_OPS_OPENCL_IMAGE_WHERE_H_
