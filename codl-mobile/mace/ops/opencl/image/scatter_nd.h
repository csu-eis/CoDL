
#ifndef MACE_OPS_OPENCL_IMAGE_SCATTER_ND_H_
#define MACE_OPS_OPENCL_IMAGE_SCATTER_ND_H_

#include "mace/ops/opencl/scatter_nd.h"

#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class ScatterNDKernel : public OpenCLScatterNDKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *indices,
      const Tensor *updates,
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

#endif  // MACE_OPS_OPENCL_IMAGE_SCATTER_ND_H_
