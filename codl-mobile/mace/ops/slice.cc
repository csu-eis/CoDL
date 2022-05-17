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

#include <functional>
#include <memory>

#include "mace/core/operator.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/slice.h"
//#include "mace/ops/opencl/buffer/shape.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

class SliceOpBase : public Operation {
 public:
  explicit SliceOpBase(OpConstructContext *context)
      : Operation(context),
        axes_(Operation::GetRepeatedArgs<int>("axes")),
        starts_(Operation::GetRepeatedArgs<int>("starts")),
        ends_(Operation::GetRepeatedArgs<int>("ends")) {}

  void LoadArgs() {
    const Tensor *starts_tensor = this->Input(1);
    const Tensor *ends_tensor = this->Input(2);
    const Tensor *axes_tensor = this->Input(3);
    if (starts_tensor != nullptr && starts_.size() == 0 &&
        ends_tensor != nullptr && ends_.size() == 0 &&
        axes_tensor != nullptr && axes_.size() == 0) {
      Tensor::MappingGuard starts_guard(starts_tensor);
      Tensor::MappingGuard ends_guard(ends_tensor);
      Tensor::MappingGuard axes_guard(axes_tensor);
      starts_.push_back(starts_tensor->data<int32_t>()[0]);
      ends_.push_back(ends_tensor->data<int32_t>()[0]);
      axes_.push_back(axes_tensor->data<int32_t>()[0]);
    }
  }

 protected:
  std::vector<int> axes_;
  std::vector<int> starts_;
  std::vector<int> ends_;
};

template <DeviceType D, typename T>
class SliceOp;

template <typename T>
class SliceOp<DeviceType::CPU, T> : public SliceOpBase {
 public:
  explicit SliceOp(OpConstructContext *context) : SliceOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    LoadArgs();
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    const index_t rank = input->dim_size();
    MACE_CHECK(rank >= 1) << "The input dim size should >= 1";
    const index_t input_dim = input->dim(rank - 1);
    const std::vector<T> input_shape_value(
        input->data<T>(), input->data<T>() + input->shape()[0]);
    VLOG(1) << "input_shape " << VectorToString<index_t>(input->shape())
            << ", input_shape_value " << VectorToString<T>(input_shape_value)
            << ", axes " << VectorToString<int>(axes_)
            << ", starts " << VectorToString<int>(starts_)
            << ", ends " << VectorToString<int>(ends_);
    MACE_CHECK(starts_.size() == 1 && ends_.size() == 1 && axes_.size() == 1,
               "only support slicing at one axis.");
    MACE_CHECK(axes_[0] == -1 || axes_[0] == rank - 1,
               "only support slicing at the last axis.");
    MACE_CHECK(starts_[0] < input_dim && starts_[0] >= 0
                   && ends_[0] >= 0
                   && ends_[0] <= input_dim)
      << "The starts and ends caused over range error.";
    const index_t offset = starts_[0];
    const index_t output_dim = ends_[0] - starts_[0];
    MACE_CHECK(output_dim >= 0, "output_dim should >= 0");

    const index_t frames =
        std::accumulate(input->shape().begin(), input->shape().end() - 1, 1,
                        std::multiplies<index_t>());

    std::vector<index_t> output_shape = input->shape();
    output_shape[rank - 1] = output_dim;
    VLOG(1) << "output_shape " << VectorToString<index_t>(output_shape);
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    for (index_t i = 0; i < frames; ++i) {
      const T *input_base =
          input_data + i * input_dim + offset;
      T *output_base =
          output_data + i * output_dim;
      memcpy(output_base, input_base, output_dim * sizeof(T));
    }

    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<>
class SliceOp<DeviceType::GPU, float> : public SliceOpBase {
 public:
  explicit SliceOp(OpConstructContext *context) : SliceOpBase(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::SliceKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    LoadArgs();
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, input, axes_, starts_, ends_, output);
  }

 private:
  std::unique_ptr<OpenCLSliceKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterSlice(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Slice", SliceOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_GPU_OP(op_registry, "Slice", SliceOp);
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Slice")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                MACE_UNUSED(context);
                //return {DeviceType::CPU};
                return {DeviceType::CPU, DeviceType::GPU};
              }));
}

}  // namespace ops
}  // namespace mace
