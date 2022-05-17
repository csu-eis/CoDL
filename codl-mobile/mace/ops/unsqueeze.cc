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

#include <unordered_set>
#include <vector>

#include "mace/core/operator.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/unsqueeze.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template <DeviceType D, class T>
class UnsqueezeOp : public Operation {
 public:
  explicit UnsqueezeOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetRepeatedArgs<int>("axis", {})) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(0);
    MACE_CHECK(!axis_.empty(), "Unsqueeze op should have axis values.");
    std::vector<index_t> output_shape = input->shape();
    for (size_t i = 0; i < axis_.size(); ++i) {
      MACE_CHECK(axis_[i] >= 0, "axis's value should be non-negative.");
      output_shape.insert(output_shape.begin() + axis_[i], 1);
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    const index_t data_size =
        std::accumulate(input->shape().begin(), input->shape().end(), 1,
                        std::multiplies<index_t>());
    memcpy(output_data, input_data, data_size * sizeof(T));
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::vector<int> axis_;

 private:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef MACE_ENABLE_OPENCL
template<>
class UnsqueezeOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit UnsqueezeOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetRepeatedArgs<int>("axis", {})) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::UnsqueezeKernel>(axis_);
    } else {
      MACE_NOT_IMPLEMENTED;
      //kernel_ = make_unique<opencl::buffer::UnsqueezeKernel>(axis_);
    }
  }
  
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, input, output);
  }

 private:
  std::vector<int> axis_;
  std::unique_ptr<OpenCLUnsqueezeKernel> kernel_;

 private:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_OPENCL

void RegisterUnsqueeze(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Unsqueeze", UnsqueezeOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_OP(op_registry, "Unsqueeze", UnsqueezeOp,
                   DeviceType::CPU, int32_t);
  MACE_REGISTER_GPU_OP(op_registry, "Unsqueeze", UnsqueezeOp);
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Unsqueeze")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                MACE_UNUSED(context);
                //return {DeviceType::CPU};
                return {DeviceType::CPU, DeviceType::GPU};
              }));
}

}  // namespace ops
}  // namespace mace
