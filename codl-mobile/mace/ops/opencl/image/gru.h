
#ifndef MACE_OPS_OPENCL_IMAGE_GRU_H_
#define MACE_OPS_OPENCL_IMAGE_GRU_H_

#include "mace/ops/opencl/gru.h"

#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class GruKernel : public OpenCLGruKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *X,
      const Tensor *W,
      const Tensor *R,
      const Tensor *B,
      const Tensor *sequence_lens_tensor,
      const Tensor *initial_h_tensor,
      const GruDirection direction,
      const int hidden_size,
      Tensor *Y,
      Tensor *Y_h) override;

 private:
  //cl::Kernel kernel_;
  //uint32_t kwg_size_;
  //std::vector<index_t> input_shape_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_GRU_H_
