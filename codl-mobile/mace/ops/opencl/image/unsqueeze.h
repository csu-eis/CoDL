
#ifndef MACE_OPS_OPENCL_IMAGE_UNSQUEEZE_H_
#define MACE_OPS_OPENCL_IMAGE_UNSQUEEZE_H_

#include "mace/ops/opencl/unsqueeze.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <set>
#include <string>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class UnsqueezeKernel : public OpenCLUnsqueezeKernel {
 public:
  explicit UnsqueezeKernel(const std::vector<int> axis) : axis_(axis) {}
  
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      Tensor *output) override;

 private:
  std::vector<int> axis_;
  //cl::Kernel kernel_;
  //uint32_t kwg_size_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_UNSQUEEZE_H_
