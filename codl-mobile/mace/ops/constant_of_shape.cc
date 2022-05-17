
#include <memory>

#include "mace/core/operator.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/constant_of_shape.h"
//#include "mace/ops/opencl/buffer/constant_of_shape.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<DeviceType D, class T>
class ConstantOfShapeOp;

template<typename T>
class ConstantOfShapeOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit ConstantOfShapeOp(OpConstructContext *context)
      : Operation(context),
        value_(Operation::GetOptionalArg<T>("value", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    std::vector<index_t> output_shape(input->dim_size());
    for (index_t i = 0; i < input->dim_size(); i ++) {
      output_shape[i] = input->data<T>()[i];
    }
    VLOG(1) << "output_shape " << VectorToString<index_t>(output_shape);

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  T value_;
};

#ifdef MACE_ENABLE_OPENCL
template<>
class ConstantOfShapeOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit ConstantOfShapeOp(OpConstructContext *context)
      : Operation(context),
        value_(Operation::GetOptionalArg<float>("value", 0)) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ConstantOfShapeKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, input, value_, output);
  }

 private:
  float value_;
  std::unique_ptr<OpenCLConstantOfShapeKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterConstantOfShape(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "ConstantOfShape", ConstantOfShapeOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_OP(op_registry, "ConstantOfShape", ConstantOfShapeOp,
                   DeviceType::CPU, int32_t);
  MACE_REGISTER_GPU_OP(op_registry, "ConstantOfShape", ConstantOfShapeOp);
}

}  // namespace ops
}  // namespace mace
