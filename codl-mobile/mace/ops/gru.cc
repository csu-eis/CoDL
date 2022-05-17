
#include <memory>

#include "mace/core/operator.h"
#include "mace/ops/common/gru_type.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/gru.h"
//#include "mace/ops/opencl/buffer/gru.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<DeviceType D, class T>
class GruOp;

template<typename T>
class GruOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit GruOp(OpConstructContext *context)
      : Operation(context),
        direction_(static_cast<GruDirection>(
            Operation::GetOptionalArg<int>("direction", 0))),
        hidden_size_(Operation::GetOptionalArg<int>("hidden_size", 0)),
        linear_before_reset_(Operation::GetOptionalArg<int>(
            "linear_before_reset", 0)) {}

  MaceStatus Run(OpContext *context) override {
    VLOG(1) << "direction " << static_cast<int>(direction_)
            << ", hidden_size " << hidden_size_
            << ", linear_before_reset " << linear_before_reset_;
    MACE_CHECK(hidden_size_ > 0, "hidden size must be > 0");
    MACE_UNUSED(context);
    const Tensor *X = this->Input(0);
    const Tensor *W = this->Input(1);
    const Tensor *R = this->Input(2);
    const Tensor *B = this->Input(3);
    const Tensor *sequence_lens_tensor = this->Input(4);
    const Tensor *initial_h_tensor = this->Input(5);
    Tensor *Y = this->Output(0);
    Tensor *Y_h = this->Output(1);

    MACE_UNUSED(W);
    MACE_UNUSED(R);
    MACE_UNUSED(B);
    MACE_UNUSED(initial_h_tensor);

    const index_t batch_size = X->dim(1);
    const index_t num_directions
        = (direction_ == GruDirection::GD_BIDIRECTIONAL) ? 2 : 1;
    index_t seq_length = batch_size;
    if (sequence_lens_tensor != nullptr) {
      Tensor::MappingGuard guard(sequence_lens_tensor);
      seq_length = sequence_lens_tensor->data<int32_t>()[0];
    }

    std::vector<index_t> Y_shape;
    Y_shape.push_back(seq_length);
    Y_shape.push_back(num_directions);
    Y_shape.push_back(batch_size);
    Y_shape.push_back(hidden_size_);
    MACE_RETURN_IF_ERROR(Y->Resize(Y_shape));
    
    std::vector<index_t> Y_h_shape;
    Y_h_shape.push_back(num_directions);
    Y_h_shape.push_back(batch_size);
    Y_h_shape.push_back(hidden_size_);
    MACE_RETURN_IF_ERROR(Y_h->Resize(Y_h_shape));

    VLOG(1) << "X_shape " << VectorToString<index_t>(X->shape())
            << ", Y_shape " << VectorToString<index_t>(Y_shape)
            << ", Y_h_shape " << VectorToString<index_t>(Y_h_shape);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  GruDirection direction_;
  int hidden_size_;
  int linear_before_reset_;
};

#ifdef MACE_ENABLE_OPENCL
template<>
class GruOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit GruOp(OpConstructContext *context)
      : Operation(context),
        direction_(static_cast<GruDirection>(
            Operation::GetOptionalArg<int>("direction", 0))),
        hidden_size_(Operation::GetOptionalArg<int>("hidden_size", 0)),
        linear_before_reset_(Operation::GetOptionalArg<int>(
            "linear_before_reset", 0)) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::GruKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *X = this->Input(0);
    const Tensor *W = this->Input(1);
    const Tensor *R = this->Input(2);
    const Tensor *B = this->Input(3);
    const Tensor *sequence_lens_tensor = this->Input(4);
    const Tensor *initial_h_tensor = this->Input(5);
    Tensor *Y = this->Output(0);
    Tensor *Y_h = this->Output(1);
    return kernel_->Compute(context, X, W, R, B,
                            sequence_lens_tensor, initial_h_tensor,
                            direction_, hidden_size_,
                            Y, Y_h);
  }

 private:
  GruDirection direction_;
  int hidden_size_;
  int linear_before_reset_;
  std::unique_ptr<OpenCLGruKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterGru(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Gru", GruOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_GPU_OP(op_registry, "Gru", GruOp);
}

}  // namespace ops
}  // namespace mace
