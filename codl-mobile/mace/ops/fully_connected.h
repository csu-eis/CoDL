// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_OPS_FULLY_CONNECTED_H_
#define MACE_OPS_FULLY_CONNECTED_H_

#include <memory>
#include <string>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/operator.h"
#include "mace/core/co_operator.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/fully_connected_part_plan.h"

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/gemv.h"
#include "mace/ops/arm/fp32/activation.h"

#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/arm/q8/gemv.h"
#endif

#else  // MACE_ENABLE_NEON
#include "mace/ops/ref/gemv.h"
#include "mace/ops/ref/activation.h"
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/fully_connected.h"
#include "mace/ops/opencl/buffer/fully_connected.h"
#endif  // MACE_ENABLE_OPENCL

#include "mace/utils/memory.h"

namespace mace {
namespace ops {

class FullyConnectedOpBase : public CoOperation {
 public:
  explicit FullyConnectedOpBase(OpConstructContext *context)
      : CoOperation(context),
        activation_(ops::StringToActivationType(
            CoOperation::GetOptionalArg<std::string>("activation",
                                                   "NOOP"))),
        relux_max_limit_(CoOperation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(CoOperation::GetOptionalArg<float>(
            "leakyrelu_coefficient", 0.0f)) {}
 protected:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;

  MACE_OP_INPUT_TAGS(INPUT, WEIGHT, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

template<DeviceType D, class T>
class FullyConnectedOp;

template<>
class FullyConnectedOp<DeviceType::CPU, float> : public FullyConnectedOpBase {
 public:
  explicit FullyConnectedOp(OpConstructContext *context)
      : FullyConnectedOpBase(context),
        activation_delegator_(activation_,
                              relux_max_limit_,
                              leakyrelu_coefficient_) {}

 private:
  MaceStatus RunCpuGpuImage(OpContext *context) override {
    return RunCpu(context);
  }

  MaceStatus RunCpuGpuBuffer(OpContext *context) override {
    return RunCpu(context);
  }

  MaceStatus RunCpu(OpContext *context) {
    const Tensor *input = this->Input(INPUT);
    const Tensor *weight = this->Input(WEIGHT);  // OIHW
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    MACE_CHECK(
        input->dim(1) == weight->dim(1) && input->dim(2) == weight->dim(2) &&
            input->dim(3) == weight->dim(3),
        "The shape of Input: ", MakeString(input->shape()),
        "The shape of Weight: ", MakeString(weight->shape()),
        " don't match.");
    if (bias) {
      MACE_CHECK(weight->dim(0) == bias->dim(0),
                 "The shape of Weight: ", MakeString(weight->shape()),
                 " and shape of Bias: ", bias->dim(0),
                 " don't match.");
    }
    std::vector<index_t> output_shape = {input->dim(0), weight->dim(0), 1, 1};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    const index_t batch = output->dim(0);
    const index_t input_size = weight->dim(1) * weight->dim(2) * weight->dim(3);
    const index_t output_size = weight->dim(0);

    gemv_.Compute(context,
                  weight,
                  input,
                  bias,
                  batch,
                  output_size,
                  input_size,
                  false,
                  true,
                  output);

    activation_delegator_.Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
#ifdef MACE_ENABLE_NEON
  arm::fp32::Gemv gemv_;
  arm::fp32::Activation activation_delegator_;
#else
  ref::Gemv<float> gemv_;
  ref::Activation activation_delegator_;
#endif  // MACE_ENABLE_NEON
};

#ifdef MACE_ENABLE_QUANTIZE
template<>
class FullyConnectedOp<DeviceType::CPU, uint8_t>
    : public FullyConnectedOpBase {
 public:
  explicit FullyConnectedOp(OpConstructContext *context)
      : FullyConnectedOpBase(context) {}

 private:
  MaceStatus RunCpuGpuImage(OpContext *context) override {
    return RunCpu(context);
  }

  MaceStatus RunCpuGpuBuffer(OpContext *context) override {
    return RunCpu(context);
  }

  MaceStatus RunCpu(OpContext *context) {
    const Tensor *input = this->Input(INPUT);
    const Tensor *weight = this->Input(WEIGHT);  // OIHW
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    MACE_CHECK(
        input->dim(1) == weight->dim(1) && input->dim(2) == weight->dim(2) &&
            input->dim(3) == weight->dim(3),
        "The shape of Input: ", MakeString(input->shape()),
        "The shape of Weight: ", MakeString(weight->shape()),
        " don't match.");
    if (bias) {
      MACE_CHECK(weight->dim(0) == bias->dim(0),
                 "The shape of Weight: ", MakeString(weight->shape()),
                 " and shape of Bias: ", bias->dim(0),
                 " don't match.");
    }
    auto gemm_context = context->device()->cpu_runtime()->GetGemmlowpContext();
    MACE_CHECK_NOTNULL(gemm_context);

    std::vector<index_t> output_shape = {input->dim(0), 1, 1, weight->dim(0)};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    const int batch = static_cast<int>(output->dim(0));
    const int input_size =
        static_cast<int>(weight->dim(1) * weight->dim(2) * weight->dim(3));
    const int output_size = static_cast<int>(weight->dim(0));
    gemv_.Compute(context,
                  weight,
                  input,
                  bias,
                  batch,
                  output_size,
                  input_size,
                  false,
                  true,
                  output);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
#ifdef MACE_ENABLE_NEON
  ::mace::ops::arm::q8::Gemv<uint8_t> gemv_;
#else
  ref::Gemv<uint8_t> gemv_;
#endif  // MACE_ENABLE_NEON
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL

constexpr int kNumTempTensorsFullyConnected = 9;

template<>
class FullyConnectedOp<DeviceType::GPU, float> : public FullyConnectedOpBase {
 public:
  explicit FullyConnectedOp(OpConstructContext *context)
      : FullyConnectedOpBase(context),
        activation_delegator_(activation_,
                              relux_max_limit_,
                              leakyrelu_coefficient_) {
    InitCpu(context);
    InitGpu(context);
  }

  MaceStatus MakePartitionPlan(OpContext *context) override {
    MACE_UNUSED(context);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus PrepareTemporaryTensors(OpContext *context) override {
    MACE_UNUSED(context);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus EnqueueInputDataTransform(
      OpContext *context, StatsFuture *future = nullptr) override {
    MACE_UNUSED(context);
    MACE_UNUSED(future);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus EnqueueMap(
      OpContext *context, StatsFuture *future = nullptr) override {
    MACE_UNUSED(context);
    MACE_UNUSED(future);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus EnqueueGpuCompute(
      OpContext *context, StatsFuture *future = nullptr) override {
    MACE_UNUSED(context);
    MACE_UNUSED(future);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus EnqueueUnmap(OpContext *context,
                          cl::UserEvent **event = nullptr,
                          StatsFuture *future = nullptr) override {
    MACE_UNUSED(context);
    MACE_UNUSED(event);
    MACE_UNUSED(future);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus EnqueueOutputDataTransform(
      OpContext *context, StatsFuture *future = nullptr) override {
    MACE_UNUSED(context);
    MACE_UNUSED(future);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus RunCpuCompute(OpContext *context) override {
    MACE_UNUSED(context);
    return MaceStatus::MACE_SUCCESS;
  }
  
 private:
  void InitCpu(OpConstructContext *context) override {
    MACE_UNUSED(context);
    MACE_CHECK(kNumTempTensorsFullyConnected == LAST_TENSOR_INDEX);
    for (size_t i = 0; i < kNumTempTensorsFullyConnected; i ++) {
      temp_tensor_indices_[i] = -1;
    }
  }

  void InitGpu(OpConstructContext *context) override {
    mem_type_ = MemoryType::CPU_BUFFER;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      mem_type_ = MemoryType::GPU_IMAGE;
      kernel_ = make_unique<opencl::image::FullyConnectedKernel>();
      // Transform filter tensor to target format
      MACE_CHECK(TransformFilter(
          context,
          operator_def_.get(),
          1,
          OpenCLBufferType::WEIGHT_WIDTH,
          mem_type_) == MaceStatus::MACE_SUCCESS);
    } else {
      mem_type_ = MemoryType::GPU_BUFFER;
      kernel_ = make_unique<opencl::buffer::FullyConnectedKernel>();
      // Transform filter tensor to target format
      MACE_CHECK(TransformFilter(
          context,
          operator_def_.get(),
          1,
          OpenCLBufferType::CONV2D_FILTER,
          mem_type_) == MaceStatus::MACE_SUCCESS);
    }

    if (operator_def_->input_size() > 2) {
      MACE_CHECK(TransformFilter(
          context, operator_def_.get(), 2, OpenCLBufferType::ARGUMENT, mem_type_)
                     == MaceStatus::MACE_SUCCESS);
    }
  }

  MaceStatus TransformWeightGpuToCpu(OpContext *context,
                                     const Tensor *src,
                                     Tensor *dst) override;

  MaceStatus TransformBiasGpuToCpu(OpContext *context,
                                   const Tensor *src,
                                   Tensor *dst) override;

  MaceStatus RunCpu(OpContext *context,
                    const Tensor *input,
                    const Tensor *weight,
                    const Tensor *bias,
                    Tensor *output) {
    LOG(INFO) << "CPU weight shape " << VectorToString<index_t>(weight->shape());

    MACE_CHECK(
        input->dim(1) == weight->dim(1) && input->dim(2) == weight->dim(2) &&
            input->dim(3) == weight->dim(3),
        "The shape of Input: ", MakeString(input->shape()),
        "The shape of Weight: ", MakeString(weight->shape()),
        " don't match.");
    if (bias) {
      MACE_CHECK(weight->dim(0) == bias->dim(0),
                 "The shape of Weight: ", MakeString(weight->shape()),
                 " and shape of Bias: ", bias->dim(0),
                 " don't match.");
    }
    std::vector<index_t> output_shape = {input->dim(0), weight->dim(0), 1, 1};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    const index_t batch = output->dim(0);
    const index_t input_size = weight->dim(1) * weight->dim(2) * weight->dim(3);
    const index_t output_size = weight->dim(0);

    gemv_.Compute(context,
                  weight,
                  input,
                  bias,
                  batch,
                  output_size,
                  input_size,
                  false,
                  true,
                  output);

    activation_delegator_.Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus RunGpu(OpContext *context,
                    const Tensor *input,
                    const Tensor *weight,
                    const Tensor *bias,
                    Tensor *output) {
    MACE_CHECK(
        input->dim(1) == weight->dim(2) && input->dim(2) == weight->dim(3) &&
            input->dim(3) == weight->dim(1),
        "The shape of Input: ", MakeString(input->shape()),
        "The shape of Weight: ", MakeString(weight->shape()),
        " don't match.");

#if 0
    LOG(INFO) << "GPU weight shape " << VectorToString<index_t>(weight->shape());
#endif

    return kernel_->Compute(
        context, input, weight, bias, activation_, relux_max_limit_,
        leakyrelu_coefficient_, output);
  }

  MaceStatus RunGpu(OpContext *context) {
    const Tensor *input = this->Input(INPUT);
    const Tensor *weight = this->Input(WEIGHT);  // OIHW
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);
    return RunGpu(context, input, weight, bias, output);
  }

  MaceStatus MakePartitionPlan(OpContext *context,
                               const Tensor *input,
                               const Tensor *weight,
                               PartitionDim partition_dim,
                               float partition_ratio);

  MaceStatus RunCpuGpuImage(OpContext *context) override;
  
  MaceStatus RunCpuGpuImageV2(OpContext *context) override {
    MACE_UNUSED(context);
    return MaceStatus::MACE_SUCCESS;
  }
  
  MaceStatus RunCpuGpuBuffer(OpContext *context) override;

 private:
  enum _TempTensorIndex {
    GPU_INPUT   = 0,
    GPU_WEIGHT  = 1,
    GPU_OUTPUT  = 2,
    CPU_INPUT   = 3,
    CPU_WEIGHT1 = 4,
    CPU_WEIGHT2 = 5,
    CPU_BIAS1   = 6,
    CPU_BIAS2   = 7,
    CPU_OUTPUT  = 8,
    LAST_TENSOR_INDEX
  };

  index_t temp_tensor_indices_[kNumTempTensorsFullyConnected];

  std::shared_ptr<FullyConnectedPartPlan> part_plan_;

  // CPU delegators.
#ifdef MACE_ENABLE_NEON
  arm::fp32::Gemv gemv_;
  arm::fp32::Activation activation_delegator_;
#else
  ref::Gemv<float> gemv_;
  ref::Activation activation_delegator_;
#endif  // MACE_ENABLE_NEON
  
  // GPU kernel.
  std::unique_ptr<OpenCLFullyConnectedKernel> kernel_;
};

typedef FullyConnectedOp<DeviceType::GPU, float> FullyConnectedGpuFloatOp;

#endif  // MACE_ENABLE_OPENCL

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_FULLY_CONNECTED_H_
