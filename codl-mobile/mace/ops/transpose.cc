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

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cmath>
#include <vector>

#include "mace/core/operator.h"
#include "mace/ops/common/transpose.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/transpose.h"
//#include "mace/ops/opencl/buffer/transpose.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class TransposeOp;

template<DeviceType D>
class TransposeOp<D, float> : public Operation {
 public:
  explicit TransposeOp(OpConstructContext *context)
      : Operation(context),
        dims_(Operation::GetRepeatedArgs<int>("dims")) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    const std::vector<index_t> &input_shape = input->shape();
    MACE_CHECK((input_shape.size() == 4 && dims_.size() == 4) ||
                   (input_shape.size() == 3 && dims_.size() == 3) ||
                   (input_shape.size() == 2 && dims_.size() == 2),
               "rank should be 2, 3 or 4");
    std::vector<index_t> output_shape;
    for (size_t i = 0; i < dims_.size(); ++i) {
      output_shape.push_back(input_shape[dims_[i]]);
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

#if 0
    LOG(INFO) << "input_shape " << VectorToString<index_t>(input->shape())
              << ", output_shape " << VectorToString<index_t>(output->shape());
#endif

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();

    return Transpose(&context->device()->cpu_runtime()->thread_pool(),
                     input_data, input->shape(), dims_, output_data);
  }

 private:
  std::vector<int> dims_;
};

#ifdef MACE_ENABLE_OPENCL
template<>
class TransposeOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit TransposeOp(OpConstructContext *context)
      : Operation(context),
        dims_(Operation::GetRepeatedArgs<int>("dims")) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::TransposeKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, input, dims_, output);
  }

 private:
  std::vector<int> dims_;
  std::unique_ptr<OpenCLTransposeKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterTranspose(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Transpose", TransposeOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_GPU_OP(op_registry, "Transpose", TransposeOp);
}

}  // namespace ops
}  // namespace mace
