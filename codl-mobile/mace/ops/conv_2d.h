
#ifndef MACE_OPS_CONV_2D_H_
#define MACE_OPS_CONV_2D_H_

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif
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
#include "mace/ops/conv_pool_2d_base.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/common/transpose.h"
#include "mace/ops/conv_2d_part_plan.h"
#include "mace/utils/memory.h"
#include "mace/utils/math.h"
#include "mace/utils/op_delay_tool.h"

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/conv_2d.h"
#include "mace/ops/arm/fp32/conv_2d_1x1.h"
#include "mace/ops/arm/fp32/conv_2d_3x3.h"
#include "mace/ops/arm/fp32/conv_2d_3x3_winograd.h"
#include "mace/ops/arm/fp32/conv_2d_5x5.h"
#include "mace/ops/arm/fp32/conv_2d_7x7.h"
#include "mace/ops/arm/fp32/conv_2d_9x9.h"
#include "mace/ops/arm/fp32/conv_2d_1xn.h"
#include "mace/ops/arm/fp32/conv_general.h"
#include "mace/ops/arm/fp32/bias_add.h"
#include "mace/ops/arm/fp32/activation.h"
#else
#include "mace/ops/ref/activation.h"
#include "mace/ops/ref/bias_add.h"
#endif  // MACE_ENABLE_NEON

#include "mace/ops/ref/conv_2d.h"

#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/common/gemmlowp_util.h"
#include "mace/ops/arm/q8/quantization_util.h"
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/buffer/conv_2d.h"
#include "mace/ops/opencl/buffer/pad.h"
#include "mace/ops/opencl/image/conv_2d.h"
#include "mace/ops/opencl/image/pad.h"
#endif  // MACE_ENABLE_OPENCL

// NOTE(fucheng): This definition is to enable original Conv2D GPU of MACE.
//#define CODL_ENABLE_MACE_CONV2D_GPU

namespace mace {
namespace ops {

#ifdef MACE_ENABLE_NEON
typedef arm::fp32::Conv2dBase Conv2dDelegator;
typedef arm::fp32::BiasAdd    BiasAddDelegator;
typedef arm::fp32::Activation ActivationDelegator;
#else  // MACE_ENABLE_NEON
typedef ref::Conv2d<float> Conv2dDelegator;
typedef ref::BiasAdd       BiasAddDelegator;
typedef ref::Activation    ActivationDelegator;
#endif  // MACE_ENABLE_NEON

#ifndef CODL_ENABLE_MACE_CONV2D_GPU
#ifdef MACE_ENABLE_OPENCL

constexpr int kNumTempTensorsConv2D = 10;

template<DeviceType D, class T>
class Conv2dOp;

template<>
class Conv2dOp<DeviceType::GPU, float> : public ConvPool2dOpBase {
 public:
  explicit Conv2dOp(OpConstructContext *context)
      : ConvPool2dOpBase(context),
        activation_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation", "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(Operation::GetOptionalArg<float>(
            "leakyrelu_coefficient", 0.0f)),
        wino_block_size_(Operation::GetOptionalArg<int>("wino_block_size", 0)),
        part_plan_(nullptr),
        activation_delegator_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation", "NOOP")),
            Operation::GetOptionalArg<float>("max_limit", 0.0f),
            Operation::GetOptionalArg<float>("leakyrelu_coefficient", 0.0f))
  
  {
    InitCpu(context);
    InitGpu(context);
  }

  ~Conv2dOp() {}

  MaceStatus MakePartitionPlan(OpContext *context) override;

  MaceStatus PrepareTemporaryTensors(OpContext *context) override;

  MaceStatus EnqueueInputDataTransform(
      OpContext *context, StatsFuture *future = nullptr) override;

  MaceStatus EnqueueMap(
      OpContext *context, StatsFuture *future = nullptr) override;

  MaceStatus EnqueueGpuCompute(
      OpContext *context, StatsFuture *future = nullptr) override;

  MaceStatus EnqueueUnmap(OpContext *context,
                          cl::UserEvent **event = nullptr,
                          StatsFuture *future = nullptr) override;

  MaceStatus EnqueueOutputDataTransform(
      OpContext *context, StatsFuture *future = nullptr) override;

  MaceStatus RunCpuCompute(OpContext *context) override;

 private:
  void InitCpu(OpConstructContext *context) override;

  void InitGpu(OpConstructContext *context) override;
  
#ifdef MACE_ENABLE_NEON
  Conv2dDelegator* CreateNEONConv2dDelegator(
    const Tensor *input,
    const Tensor *filter,
    const std::vector<int> strides,
    const std::vector<int> paddings,
    const Padding padding_type,
    const std::vector<int> dilations);
#endif  // MACE_ENABLE_NEON

  Conv2dDelegator* CreateCpuConv2dDelegator(
      const Tensor *input,
      const Tensor *filter) {
    const std::vector<int> paddings;
    const Padding padding_type = Padding::VALID;
#ifdef MACE_ENABLE_NEON
    return CreateNEONConv2dDelegator(input, filter, strides_,
                                     paddings, padding_type, dilations_);
#else
    return new Conv2dDelegator(strides_, dilations_, paddings, padding_type);
#endif
  }

  const Tensor* RunPadInput(OpContext *context);

  MaceStatus TransformWeightGpuToCpu(OpContext *context,
                                     const Tensor *src,
                                     Tensor *dst) override;

  MaceStatus TransformBiasGpuToCpu(OpContext *context,
                                   const Tensor *src,
                                   Tensor *dst) override;

  MaceStatus RunCpu(OpContext *context,
                    const Tensor *input,
                    const Tensor *filter,
                    const Tensor *bias,
                    Tensor *output) {
    if (conv2d_delegator_ == nullptr) {
      conv2d_delegator_.reset(CreateCpuConv2dDelegator(input, filter));
    }
    conv2d_delegator_->Compute(context, input, filter, output);
    bias_add_delegator_.Compute(context, output, bias, output);
    activation_delegator_.Compute(context, output, output);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus RunGpu(OpContext *context) {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    return kernel_->Compute(context, input, filter, bias,
                            strides_.data(), padding_type_, paddings_,
                            dilations_.data(), activation_, relux_max_limit_,
                            leakyrelu_coefficient_, wino_block_size_, output);
  }
  
  MaceStatus RunGpu(OpContext *context,
                    const Tensor *input,
                    const Tensor *filter,
                    const Tensor *bias,
                    Tensor *output) {
    const Padding padding_type = Padding::VALID;
    const std::vector<int> paddings;
    
    return kernel_->Compute(context, input, filter, bias,
                            strides_.data(), padding_type, paddings,
                            dilations_.data(), activation_, relux_max_limit_,
                            leakyrelu_coefficient_, wino_block_size_, output);
  }

  MaceStatus RunCpuGpuImage(OpContext *context) override;
  
  MaceStatus RunCpuGpuImageV2(OpContext *context) override;

  MaceStatus RunCpuGpuBuffer(OpContext *context) override;

 private:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
  int wino_block_size_;

  // Partition plan.
  std::shared_ptr<ConvPool2dPartPlan> part_plan_;

  // Pad input OpenCL kernel.
  std::unique_ptr<OpenCLPadKernel> pad_kernel_;
  // CPU delegator.
  std::unique_ptr<Conv2dDelegator> conv2d_delegator_;
  BiasAddDelegator                 bias_add_delegator_;
  ActivationDelegator              activation_delegator_;
  // OpenCL kernel.
  std::unique_ptr<OpenCLConv2dKernel> kernel_;

  enum _TempTensorIndex {
    GPU_PADDED_INPUT = 0,
    GPU_INPUT  = 1,
    GPU_FILTER = 2,
    GPU_OUTPUT = 3,
    CPU_INPUT  = 4,
    CPU_FILTER_V1 = 5,
    CPU_FILTER_V2 = 6,
    CPU_BIAS_V1 = 7,
    CPU_BIAS_V2 = 8,
    CPU_OUTPUT = 9,
    LAST_TENSOR_INDEX
  };

  index_t temp_tensor_indices_[kNumTempTensorsConv2D];

 private:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

typedef Conv2dOp<DeviceType::GPU, float> Conv2dGpuFloatOp;

#endif  // MACE_ENABLE_OPENCL
#endif  // CODL_ENABLE_MACE_CONV2D_GPU

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CONV_2D_H_
