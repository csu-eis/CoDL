
#include <memory>

#include "mace/core/operator.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/scatter_nd.h"
//#include "mace/ops/opencl/buffer/scatter_nd.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<DeviceType D, class T>
class ScatterNDOp;

template<typename T>
class ScatterNDOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit ScatterNDOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    const Tensor *indices = this->Input(1);
    const Tensor *updates = this->Input(2);
    Tensor *output = this->Output(0);

    MACE_UNUSED(indices);
    MACE_UNUSED(updates);

    MACE_RETURN_IF_ERROR(output->Resize(input->shape()));

    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<>
class ScatterNDOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit ScatterNDOp(OpConstructContext *context)
      : Operation(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ScatterNDKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *indices = this->Input(1);
    const Tensor *updates = this->Input(2);
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, input, indices, updates, output);
  }

 private:
  std::unique_ptr<OpenCLScatterNDKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterScatterND(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "ScatterND", ScatterNDOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_GPU_OP(op_registry, "ScatterND", ScatterNDOp);
}

}  // namespace ops
}  // namespace mace
