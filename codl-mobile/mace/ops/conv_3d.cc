
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/operator.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/conv_pool_3d_base.h"
#include "mace/ops/common/conv_pool_3d_util.h"
#include "mace/utils/memory.h"
#include "mace/utils/math.h"

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/bias_add.h"
#include "mace/ops/arm/fp32/activation.h"
#else
#include "mace/ops/ref/activation.h"
#include "mace/ops/ref/bias_add.h"
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
//#include "mace/ops/opencl/buffer/conv_2d.h"
#include "mace/ops/opencl/image/conv_3d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<DeviceType D, class T>
class Conv3dOp;

template<>
class Conv3dOp<DeviceType::CPU, float> : public ConvPool3dOpBase {
 public:
  explicit Conv3dOp(OpConstructContext *context)
      : ConvPool3dOpBase(context),
        activation_delegator_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation", "NOOP")),
            Operation::GetOptionalArg<float>("max_limit", 0.0f),
            Operation::GetOptionalArg<float>("leakyrelu_coefficient", 0.0f)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    MACE_UNUSED(bias);

    // TODO(fucheng): Calculate output shape.
    if (paddings_.size() < 3) {
      for (size_t i = 0; i < (3 - paddings_.size()); i ++) {
        paddings_.push_back(0);
      }
    }
    std::vector<index_t> output_shape(input->dim_size(), 0);
    Calc3dNCHWOutputSize(input->shape().data(), filter->shape().data(),
                         paddings_.data(), dilations_.data(), strides_.data(),
                         RoundType::FLOOR, output_shape.data());
    VLOG(1) << "input_shape " << VectorToString<index_t>(input->shape())
            << ", filter_shape " << VectorToString<index_t>(filter->shape())
            << ", strides " << VectorToString<int>(strides_)
            << ", paddings " << VectorToString<int>(paddings_)
            << ", dilations " << VectorToString<int>(dilations_)
            << ", strides " << VectorToString<int>(strides_)
            << ", output_shape " << VectorToString<index_t>(output_shape);
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    return MaceStatus::MACE_SUCCESS;
  }

 private:
#ifdef MACE_ENABLE_NEON
  //std::unique_ptr<arm::fp32::Conv2dBase> conv2d_delegator_;
  //arm::fp32::BiasAdd bias_add_delegator_;
  arm::fp32::Activation activation_delegator_;
#else
  //std::unique_ptr<ref::Conv2d<float>> ref_conv2d_delegator_;
  //ref::BiasAdd bias_add_delegator_;
  ref::Activation activation_delegator_;
#endif  // MACE_ENABLE_NEON

 private:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef MACE_ENABLE_OPENCL
template<>
class Conv3dOp<DeviceType::GPU, float> : public ConvPool3dOpBase {
 public:
  explicit Conv3dOp(OpConstructContext *context)
      : ConvPool3dOpBase(context),
        activation_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation",
                                                   "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(Operation::GetOptionalArg<float>(
              "leakyrelu_coefficient", 0.0f)),
        wino_block_size_(Operation::GetOptionalArg<int>("wino_block_size", 0)) {
    MemoryType mem_type;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_ = make_unique<opencl::image::Conv3dKernel>();
    } else {
      //mem_type = MemoryType::GPU_BUFFER;
      //kernel_ = make_unique<opencl::buffer::Conv2dKernel>();
      MACE_NOT_IMPLEMENTED;
    }
#if 0
    // Transform filter tensor to target format
    if ((wino_block_size_ == 2 || wino_block_size_ == 4) &&
        (kernel_->CheckUseWinograd(
          context->device()->gpu_runtime()->opencl_runtime(),
          context->workspace()->GetTensor(
              operator_def_->input(1))->shape(),
          std::vector<index_t>(operator_def_->output_shape(0).dims().begin(),
                               operator_def_->output_shape(0).dims().end()),
          strides_.data(),
          dilations_.data(),
          &wino_block_size_))) {
      MACE_CHECK(TransformFilter(
          context, operator_def_.get(), 1,
          OpenCLBufferType::WINOGRAD_FILTER, mem_type, wino_block_size_)
                     == MaceStatus::MACE_SUCCESS);
    } else {
      wino_block_size_ = 0;
      MACE_CHECK(TransformFilter(
          context, operator_def_.get(), 1,
          OpenCLBufferType::CONV2D_FILTER, mem_type)
                     == MaceStatus::MACE_SUCCESS);
    }
    if (operator_def_->input_size() > 2) {
      MACE_CHECK(TransformFilter(
          context, operator_def_.get(), 2, OpenCLBufferType::ARGUMENT, mem_type)
                     == MaceStatus::MACE_SUCCESS);
    }
#endif
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);
    if (!this->debug_def().name().compare("Conv_14")) {
      const std::vector<index_t> output_shape{1, 1, 1, 80, 32};
      std::vector<size_t> output_image_shape;
      OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                                  &output_image_shape);
      MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));
      return MaceStatus::MACE_SUCCESS;
    } else {
      return kernel_->Compute(context, input, filter, bias,
                              strides_.data(), padding_type_, paddings_,
                              dilations_.data(), activation_, relux_max_limit_,
                              leakyrelu_coefficient_, wino_block_size_, output);
    }
  }

 private:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
  std::unique_ptr<OpenCLConv3dKernel> kernel_;
  int wino_block_size_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_OPENCL

void RegisterConv3D(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Conv3D", Conv3dOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_GPU_OP(op_registry, "Conv3D", Conv3dOp);
}

}  // namespace ops
}  // namespace mace
