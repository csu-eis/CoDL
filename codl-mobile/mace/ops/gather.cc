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

#include <algorithm>

#include "mace/core/operator.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/gather.h"
//#include "mace/ops/opencl/buffer/gather.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template <DeviceType D, class T>
class GatherOp : public Operation {
 public:
  explicit GatherOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetOptionalArg<int>("axis", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *params = this->Input(PARAMS);
    const Tensor *indices = this->Input(INDICES);
    Tensor *output = this->Output(OUTPUT);
    std::vector<index_t> output_shape;
    if (axis_ < 0) {
      axis_ += params->dim_size();
    }
    MACE_CHECK(axis_ >= 0 && axis_ < params->dim_size(),
               "axis is out of bound: ", axis_);
    output_shape.insert(output_shape.end(), params->shape().begin(),
                        params->shape().begin() + axis_);
    output_shape.insert(output_shape.end(), indices->shape().begin(),
                        indices->shape().end());
    output_shape.insert(output_shape.end(),
                        params->shape().begin() + (axis_ + 1),
                        params->shape().end());
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

#if 0
    LOG(INFO) << "params_shape " << VectorToString<index_t>(params->shape())
              << ", indices_shape " << VectorToString<index_t>(indices->shape())
              << ", output_shape " << VectorToString<index_t>(output->shape());
#endif

    Tensor::MappingGuard indices_guard(indices);
    Tensor::MappingGuard params_guard(params);
    Tensor::MappingGuard output_guard(output);
    const int32_t *indices_data = indices->data<int32_t>();
    const T *params_data = params->data<T>();
    T *output_data = output->mutable_data<T>();

    const index_t axis_dim_size = params->dim(axis_);
    const index_t lhs_size = std::accumulate(params->shape().begin(),
                                       params->shape().begin() + axis_, 1,
                                       std::multiplies<index_t>());
    const index_t rhs_size =
        std::accumulate(params->shape().begin() + (axis_ + 1),
                        params->shape().end(), 1, std::multiplies<index_t>());
    const index_t index_size = indices->size();

#if 0
    LOG(INFO) << "axis_dim_size " << axis_dim_size << ", lhs_size " << lhs_size
              << ", rhs_size " << rhs_size << ", index_size " << index_size;
#endif

    for (index_t l = 0; l < lhs_size; ++l) {
      for (index_t idx = 0; idx < index_size; ++idx) {
#if 0
        MACE_CHECK(indices_data[idx] < axis_dim_size, "idx out of bound: ",
                   indices_data[idx]);
#else
        if (indices_data[idx] >= axis_dim_size) {
          continue;
        }
#endif

#if 0
        LOG(INFO) << "l " << l << ", idx " << idx;
        memcpy(
            output_data + ((l * index_size) + idx) * rhs_size,
            params_data + ((l * axis_dim_size) + indices_data[idx]) * rhs_size,
            sizeof(T) * rhs_size);
#else
        T *dst_data = output_data + ((l * index_size) + idx) * rhs_size;
        const T *src_data = params_data + ((l * axis_dim_size) + indices_data[idx]) * rhs_size;
        memcpy(dst_data, src_data, rhs_size);
#endif
      }
    }

    output->SetScale(params->scale());
    output->SetZeroPoint(params->zero_point());

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int axis_;
  MACE_OP_INPUT_TAGS(PARAMS, INDICES);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef MACE_ENABLE_OPENCL
template<>
class GatherOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit GatherOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetOptionalArg<int>("axis", 0)) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::GatherKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *params = this->Input(PARAMS);
    const Tensor *indices = this->Input(INDICES);
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, input, params, indices, axis_, output);
  }

 private:
  int axis_;
  std::unique_ptr<OpenCLGatherKernel> kernel_;

 private:
  MACE_OP_INPUT_TAGS(PARAMS, INDICES);
};
#endif  // MACE_ENABLE_OPENCL

void RegisterGather(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Gather", GatherOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "Gather", GatherOp,
                   DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE
#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
  MACE_REGISTER_OP(op_registry, "Gather", GatherOp,
                   DeviceType::CPU, float16_t);
#endif

  MACE_REGISTER_GPU_OP(op_registry, "Gather", GatherOp);
}

}  // namespace ops
}  // namespace mace
