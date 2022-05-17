
#include <memory>

#include "mace/core/operator.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/where.h"
//#include "mace/ops/opencl/buffer/where.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<DeviceType D, class T>
class WhereOp;

template<typename T>
class WhereOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit WhereOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *contition = this->Input(0);
    const Tensor *X = this->Input(1);
    const Tensor *Y = this->Input(2);
    Tensor *output = this->Output(0);

    MACE_UNUSED(contition);
    MACE_UNUSED(X);

    MACE_CHECK(Y->dim_size() == 1);
    const index_t dim_size = Y->dim(0);
    std::vector<index_t> output_shape = Y->shape();

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    Tensor::MappingGuard guard(Y);
    for (index_t i = 0; i < dim_size; i ++) {
      output->mutable_data<T>()[i] = static_cast<T>(Y->data<int32_t>()[i]);
    }
    //output->DebugPrint();

    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<>
class WhereOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit WhereOp(OpConstructContext *context)
      : Operation(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::WhereKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *condition = this->Input(0);
    const Tensor *X = this->Input(1);
    const Tensor *Y = this->Input(2);
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, condition, X, Y, output);
  }

 private:
  std::unique_ptr<OpenCLWhereKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterWhere(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Where", WhereOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_GPU_OP(op_registry, "Where", WhereOp);
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Where")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                MACE_UNUSED(context);
                //return {DeviceType::CPU};
                return {DeviceType::CPU, DeviceType::GPU};
              }));
}

}  // namespace ops
}  // namespace mace
