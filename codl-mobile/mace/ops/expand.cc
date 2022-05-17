
#include <memory>

#include "mace/core/operator.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/expand.h"
//#include "mace/ops/opencl/buffer/expand.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<DeviceType D, class T>
class ExpandOp;

template<typename T>
class ExpandOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit ExpandOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    const Tensor *shape = this->Input(1);
    Tensor *output = this->Output(0);

    MACE_UNUSED(input);

    Tensor::MappingGuard shape_guard(shape);
    std::vector<index_t> output_shape =
        std::vector<index_t>(shape->data<T>(),
                             shape->data<T>() + shape->size());

    VLOG(1) << "output_shape " << VectorToString<index_t>(output_shape);

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<>
class ExpandOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit ExpandOp(OpConstructContext *context)
      : Operation(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ExpandKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *shape = this->Input(1);
    Tensor *output = this->Output(0);
    if (!this->debug_def().name().compare("Expand_57") ||
        !this->debug_def().name().compare("Expand_83") ||
        !this->debug_def().name().compare("Expand_109") ||
        !this->debug_def().name().compare("Expand_135")) {
      //const std::vector<index_t> output_shape{1, 1, 576};  // CHW
      const std::vector<index_t> output_shape{1, 576, 1};  // HWC
      return kernel_->Compute(context, input, output_shape, output);
    } else {
      return kernel_->Compute(context, input, shape, output);
    }
  }

 private:
  std::unique_ptr<OpenCLExpandKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterExpand(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Expand", ExpandOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_GPU_OP(op_registry, "Expand", ExpandOp);
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Expand")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                MACE_UNUSED(context);
                //return {DeviceType::CPU};
                return {DeviceType::CPU, DeviceType::GPU};
              }));
}

}  // namespace ops
}  // namespace mace
