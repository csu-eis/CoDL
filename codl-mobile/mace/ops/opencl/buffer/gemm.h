
#ifndef MACE_OPS_OPENCL_BUFFER_GEMM_H_
#define MACE_OPS_OPENCL_BUFFER_GEMM_H_

#include "mace/ops/opencl/gemm.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {
namespace gemm {

extern MaceStatus Gemm(OpContext *context,
                       cl::Kernel *kernel,
                       const Tensor *lhs,
                       const Tensor *rhs,
                       const Tensor *bias,
                       const int m,
                       const int n,
                       const int k,
                       const ActivationType activation,
                       const float relux_max_limit,
                       const float leakyrelu_coefficient,
                       const bool input_changed,
                       Tensor *output,
                       StatsFuture *future);

}  // namespace gemm

class GemmKernel : public OpenCLGemmKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *lhs,
      const Tensor *rhs,
      const Tensor *bias,
      const ActivationType activation,
      const float relux_max_limit,
      const float leakyrelu_coefficient,
      Tensor *output) override;

 private:
  MaceStatus ComputeInternal(
      OpContext *context,
      const Tensor *lhs,
      const Tensor *rhs,
      const Tensor *bias,
      const index_t m,
      const index_t n,
      const index_t k,
      const ActivationType activation,
      const float relux_max_limit,
      const float leakyrelu_coefficient,
      Tensor *output);
  
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_GEMM_H_

