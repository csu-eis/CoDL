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

#ifndef MACE_OPS_DECONV_2D_H_
#define MACE_OPS_DECONV_2D_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/operator.h"
#include "mace/core/co_operator.h"
#include "mace/core/types.h"
#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/deconv_2d_part_plan.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/utils/memory.h"
#include "mace/utils/math.h"

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#include "mace/ops/arm/fp32/deconv_2d_2x2.h"
#include "mace/ops/arm/fp32/deconv_2d_3x3.h"
#include "mace/ops/arm/fp32/deconv_2d_4x4.h"
#include "mace/ops/arm/fp32/deconv_2d_general.h"
#include "mace/ops/arm/fp32/bias_add.h"
#include "mace/ops/arm/fp32/activation.h"
#else
#include "mace/ops/ref/bias_add.h"
#include "mace/ops/ref/activation.h"
#include "mace/ops/ref/deconv_2d.h"
#endif

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/deconv_2d.h"
#include "mace/ops/opencl/buffer/deconv_2d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

class Deconv2dOpBase : public CoOperation {
 public:
  explicit Deconv2dOpBase(OpConstructContext *context)
      : CoOperation(context),
        strides_(CoOperation::GetRepeatedArgs<int>("strides")),
        padding_type_(static_cast<Padding>(CoOperation::GetOptionalArg<int>(
            "padding", static_cast<int>(SAME)))),
        paddings_(CoOperation::GetRepeatedArgs<int>("padding_values")),
        output_paddings_(CoOperation::GetRepeatedArgs<int>(
            "output_padding_values", {0, 0})),
        group_(CoOperation::GetOptionalArg<int>("group", 1)),
        model_type_(static_cast<FrameworkType>(
                        CoOperation::GetOptionalArg<int>("framework_type", 0))),
        activation_(ops::StringToActivationType(
            CoOperation::GetOptionalArg<std::string>("activation", "NOOP"))),
        relux_max_limit_(
            CoOperation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(
            CoOperation::GetOptionalArg<float>("leakyrelu_coefficient", 0.0f)) {}

 protected:
  std::vector<int> strides_;  // [stride_h, stride_w]
  const Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<int> output_paddings_;
  const int group_;
  const FrameworkType model_type_;
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
};

template<DeviceType D, class T>
class Deconv2dOp;

template<>
class Deconv2dOp<DeviceType::CPU, float> : public Deconv2dOpBase {
 public:
  explicit Deconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context),
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
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = nullptr;
    const Tensor *output_shape_tensor = nullptr;
    if (model_type_ == TENSORFLOW) {
      output_shape_tensor =
          this->InputSize() >= 3 ? this->Input(2) : nullptr;
      bias = this->InputSize() >= 4 ? this->Input(3) : nullptr;
    } else {
      bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    }
    Tensor *output = this->Output(0);

    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

#ifdef MACE_ENABLE_NEON
    const index_t kernel_h = filter->dim(2);
    const index_t kernel_w = filter->dim(3);

    bool use_neon_2x2_s1 = kernel_h == kernel_w && kernel_h == 2 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_2x2_s2 = kernel_h == kernel_w && kernel_h == 2 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    bool use_neon_3x3_s1 = kernel_h == kernel_w && kernel_h == 3 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_3x3_s2 = kernel_h == kernel_w && kernel_h == 3 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    bool use_neon_4x4_s1 = kernel_h == kernel_w && kernel_h == 4 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_4x4_s2 = kernel_h == kernel_w && kernel_h == 4 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    if (deconv2d_delegator_ == nullptr) {
      if (use_neon_2x2_s1) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK2x2S1>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else if (use_neon_2x2_s2) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK2x2S2>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else if (use_neon_3x3_s1) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK3x3S1>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else if (use_neon_3x3_s2) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK3x3S2>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else if (use_neon_4x4_s1) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK4x4S1>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else if (use_neon_4x4_s2) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK4x4S2>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else {
        deconv2d_delegator_ =
            make_unique<arm::fp32::Deconv2dGeneral>(strides_,
                                                    std::vector<int>{1, 1},
                                                    paddings_,
                                                    output_paddings_,
                                                    padding_type_,
                                                    model_type_);
      }
    }
    deconv2d_delegator_->Compute(context,
                                 input,
                                 filter,
                                 output_shape_tensor,
                                 output);
#else
    if (deconv2d_delegator_ == nullptr) {
      deconv2d_delegator_ = make_unique<ref::Deconv2d<float>>(strides_,
                                                              std::vector<int>{
                                                                  1, 1},
                                                              paddings_,
                                                              output_paddings_,
                                                              padding_type_,
                                                              model_type_);
    }
    deconv2d_delegator_->Compute(context,
                                 input,
                                 filter,
                                 output_shape_tensor,
                                 output);

#endif  // MACE_ENABLE_NEON

    bias_add_delegator_.Compute(context, output, bias, output);
    activation_delegator_.Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
#ifdef MACE_ENABLE_NEON
  std::unique_ptr<arm::fp32::Deconv2dBase> deconv2d_delegator_;
  arm::fp32::BiasAdd bias_add_delegator_;
  arm::fp32::Activation activation_delegator_;
#else
  ref::BiasAdd bias_add_delegator_;
  ref::Activation activation_delegator_;
  std::unique_ptr<ref::Deconv2d<float>> deconv2d_delegator_;
#endif  // MACE_ENABLE_NEON
};

#ifdef MACE_ENABLE_OPENCL

constexpr int kNumTempTensorsDeconv2d = 9;

template<>
class Deconv2dOp<DeviceType::GPU, float> : public Deconv2dOpBase {
 public:
  explicit Deconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context),
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
    MACE_CHECK(kNumTempTensorsDeconv2d == LAST_TENSOR_INDEX);
    for (size_t i = 0; i < kNumTempTensorsDeconv2d; i ++) {
      temp_tensor_indices_[i] = -1;
    }
  }

  void InitGpu(OpConstructContext *context) override {
    mem_type_ = MemoryType::GPU_IMAGE;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::Deconv2dKernel>();
    } else {
      mem_type_ = MemoryType::GPU_BUFFER;
      kernel_ = make_unique<opencl::buffer::Deconv2dKernel>();
    }
    MACE_CHECK(TransformFilter(
        context, operator_def_.get(), 1,
        OpenCLBufferType::CONV2D_FILTER, mem_type_)
                   == MaceStatus::MACE_SUCCESS);
    if (model_type_ == FrameworkType::TENSORFLOW) {
      if (operator_def_->input_size() >= 4) {
        MACE_CHECK(TransformFilter(
            context,
            operator_def_.get(),
            3,
            OpenCLBufferType::ARGUMENT,
            mem_type_) == MaceStatus::MACE_SUCCESS);
      }
    } else {
      if (operator_def_->input_size() >= 3) {
        MACE_CHECK(TransformFilter(
            context, operator_def_.get(), 2,
            OpenCLBufferType::ARGUMENT, mem_type_) == MaceStatus::MACE_SUCCESS);
      }
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
                    const Tensor *filter,
                    const Tensor *bias,
                    const Tensor *output_shape_tensor,
                    Tensor *output) {
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

#ifdef MACE_ENABLE_NEON
    const index_t kernel_h = filter->dim(2);
    const index_t kernel_w = filter->dim(3);

    bool use_neon_2x2_s1 = kernel_h == kernel_w && kernel_h == 2 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_2x2_s2 = kernel_h == kernel_w && kernel_h == 2 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    bool use_neon_3x3_s1 = kernel_h == kernel_w && kernel_h == 3 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_3x3_s2 = kernel_h == kernel_w && kernel_h == 3 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    bool use_neon_4x4_s1 = kernel_h == kernel_w && kernel_h == 4 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_4x4_s2 = kernel_h == kernel_w && kernel_h == 4 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    if (deconv2d_delegator_ == nullptr) {
      if (use_neon_2x2_s1) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK2x2S1>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else if (use_neon_2x2_s2) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK2x2S2>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else if (use_neon_3x3_s1) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK3x3S1>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else if (use_neon_3x3_s2) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK3x3S2>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else if (use_neon_4x4_s1) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK4x4S1>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else if (use_neon_4x4_s2) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK4x4S2>(
            paddings_, output_paddings_, padding_type_, model_type_);
      } else {
        deconv2d_delegator_ =
            make_unique<arm::fp32::Deconv2dGeneral>(strides_,
                                                    std::vector<int>{1, 1},
                                                    paddings_,
                                                    output_paddings_,
                                                    padding_type_,
                                                    model_type_);
      }
    }
    deconv2d_delegator_->Compute(context,
                                 input,
                                 filter,
                                 output_shape_tensor,
                                 output);
#else
    if (deconv2d_delegator_ == nullptr) {
      deconv2d_delegator_ = make_unique<ref::Deconv2d<float>>(strides_,
                                                              std::vector<int>{
                                                                  1, 1},
                                                              paddings_,
                                                              output_paddings_,
                                                              padding_type_,
                                                              model_type_);
    }
    deconv2d_delegator_->Compute(context,
                                 input,
                                 filter,
                                 output_shape_tensor,
                                 output);

#endif  // MACE_ENABLE_NEON

    bias_add_delegator_.Compute(context, output, bias, output);
    activation_delegator_.Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }
  
  MaceStatus RunGpu(OpContext *context) {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = nullptr;
    const Tensor *output_shape_tensor = nullptr;
    if (model_type_ == TENSORFLOW) {
      output_shape_tensor =
          this->InputSize() >= 3 ? this->Input(2) : nullptr;
      bias = this->InputSize() >= 4 ? this->Input(3) : nullptr;
    } else {
      bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    }
    Tensor *output = this->Output(0);

    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<index_t> out_shape;
    if (output_shape_tensor) {
      Tensor::MappingGuard out_shape_guard(output_shape_tensor);
      MACE_CHECK(output_shape_tensor->size() == 4,
                 "output shape should be 4-dims");
      out_shape =
          std::vector<index_t>(output_shape_tensor->data<int32_t>(),
                               output_shape_tensor->data<int32_t>() + 4);
    }
    std::vector<int> in_paddings;
    std::vector<int> out_paddings;

    CalDeconvOutputShapeAndPadSize(input->shape(),
                                   filter->shape(),
                                   strides_,
                                   padding_type_,
                                   paddings_,
                                   output_paddings_,
                                   1,
                                   &out_shape,
                                   &in_paddings,
                                   &out_paddings,
                                   nullptr,
                                   model_type_,
                                   DataFormat::NHWC);

    return kernel_->Compute(context, input, filter, bias,
                            strides_.data(), in_paddings.data(), activation_,
                            relux_max_limit_, leakyrelu_coefficient_,
                            out_shape, output);
  }

  MaceStatus RunGpu(OpContext *context,
                    const Tensor *input,
                    const Tensor *filter,
                    const Tensor *bias,
                    Tensor *output) {
    std::vector<int> in_paddings = {0, 0};
    return kernel_->Compute(context, input, filter, bias,
                            strides_.data(), in_paddings.data(), activation_,
                            relux_max_limit_, leakyrelu_coefficient_,
                            output->shape(), output);
  }

  MaceStatus MakePartitionPlan(OpContext *context,
                               const Tensor *output_shape_tensor,
                               const Tensor *input,
                               const Tensor *filter,
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
    GPU_FILTER  = 1,
    GPU_OUTPUT  = 2,
    CPU_INPUT   = 3,
    CPU_FILTER1 = 4,
    CPU_FILTER2 = 5,
    CPU_BIAS1   = 6,
    CPU_BIAS2   = 7,
    CPU_OUTPUT  = 8,
    LAST_TENSOR_INDEX
  };

  index_t temp_tensor_indices_[kNumTempTensorsDeconv2d];

  std::shared_ptr<Deconv2dPartPlan> part_plan_;

#ifdef MACE_ENABLE_NEON
  std::unique_ptr<arm::fp32::Deconv2dBase> deconv2d_delegator_;
  arm::fp32::BiasAdd bias_add_delegator_;
  arm::fp32::Activation activation_delegator_;
#else
  ref::BiasAdd bias_add_delegator_;
  ref::Activation activation_delegator_;
  std::unique_ptr<ref::Deconv2d<float>> deconv2d_delegator_;
#endif  // MACE_ENABLE_NEON
  
  std::unique_ptr<OpenCLDeconv2dKernel> kernel_;
};

typedef Deconv2dOp<DeviceType::GPU, float> Deconv2dGpuFloatOp;

#endif  // MACE_ENABLE_OPENCL

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DECONV_2D_H_
