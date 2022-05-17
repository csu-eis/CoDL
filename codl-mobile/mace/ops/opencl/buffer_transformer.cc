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

#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/common/transpose.h"
#include "mace/ops/common/transpose_util.h"

namespace mace {
namespace ops {

MaceStatus OpenCLBufferTransformer::Transform(
    OpContext *context,
    const Tensor *input,
    const OpenCLBufferType type,
    const MemoryType out_mem_type,
    const int wino_blk_size,
    Tensor *output) {
  int64_t t0;
  Workspace *ws = context->workspace();
  DataType dt = output->dtype();
  MemoryType in_mem_type = input->memory_type();
  VLOG(2) << "Transform " << in_mem_type << " to " << out_mem_type;
  if (out_mem_type == MemoryType::GPU_IMAGE ||
      out_mem_type == MemoryType::GPU_BUFFER) {
    if (in_mem_type != MemoryType::CPU_BUFFER) {
      return kernel_->Compute(
          context, input, type, wino_blk_size, output);
    } else {
      // convert to the GPU Buffer with the input's data type.
      // 1. CPU buffer to GPU Buffer
      Tensor *internal_tensor = ws->CreateTensor(
          InternalTransformedName(input->name()),
          context->device()->allocator(), input->dtype());
      VLOG(2) << "Transform CPU Buffer " << input->name()
              << " to GPU Buffer " << internal_tensor->name()
              << " with data type " << dt;
      internal_tensor->Resize(input->shape());
      
      t0 = NowMicros();
      Tensor::MappingGuard guard(internal_tensor);
      const double map_duration = (NowMicros() - t0) / 1000.0;
      
      t0 = NowMicros();
      const uint8_t *input_ptr = input->data<uint8_t>();
      uint8_t *internal_ptr = internal_tensor->mutable_data<uint8_t>();
      memcpy(internal_ptr, input_ptr, input->raw_size());
      const double memcpy_duration = (NowMicros() - t0) / 1000.0;
      // 2. convert the internal GPU Buffer to output.
      t0 = NowMicros();
      const MaceStatus status = kernel_->Compute(
          context, internal_tensor, type, wino_blk_size, output);
      const double enq_duration = (NowMicros() - t0) / 1000.0;
      LOG(INFO) << "map_duration " << map_duration << " ms"
                << ", memcpy_duration " << memcpy_duration << " ms"
                << ", enq_duration " << enq_duration << " ms";
      return status;
    }
  } else if (out_mem_type == MemoryType::CPU_BUFFER) {
    // 1. convert to the GPU Buffer with the output's data type.
    if (in_mem_type == MemoryType::GPU_IMAGE) {
      Tensor internal_tensor(context->device()->allocator(),
                             dt,
                             false,
                             InternalTransformedName(input->name()));
      MACE_RETURN_IF_ERROR(kernel_->Compute(
          context, input, type, wino_blk_size, &internal_tensor));

      // 2. convert the internal GPU Buffer to output.
      VLOG(2) << "Transform GPU Buffer " << internal_tensor.name()
              << " to CPU Buffer " << output->name()
              << " with data type " << dt;
      Tensor::MappingGuard guard(&internal_tensor);
      
      const float *internal_ptr = internal_tensor.data<float>();
      output->Resize(internal_tensor.shape());
      Tensor::MappingGuard output_guard(output);
      float *output_ptr = output->mutable_data<float>();
      memcpy(output_ptr, internal_ptr, internal_tensor.size() * sizeof(float));
    } else {
      output->Resize(input->shape());
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const float *input_ptr = input->data<float>();
      float *output_ptr = output->mutable_data<float>();
      memcpy(output_ptr, input_ptr, input->size() * sizeof(float));
    }

    return MaceStatus::MACE_SUCCESS;
  } else {
    LOG(FATAL) << "Unexpected error: " << out_mem_type;
    return MaceStatus::MACE_SUCCESS;
  }
}

MaceStatus OpenCLBufferTransformer::TransformNoInternalTensor(
    OpContext *context,
    const Tensor *input,
    const OpenCLBufferType type,
    const MemoryType out_mem_type,
    const int wino_blk_size,
    Tensor *output) {
  MemoryType in_mem_type = input->memory_type();
  if (out_mem_type == MemoryType::GPU_IMAGE ||
      out_mem_type == MemoryType::GPU_BUFFER) {
    if (in_mem_type != MemoryType::CPU_BUFFER) {
      return kernel_->Compute(
          context, input, type, wino_blk_size, output);
    } else {
      return kernel_->Compute(
          context, input, type, wino_blk_size, output);
    }
  } else if (out_mem_type == MemoryType::CPU_BUFFER) {
    output->Resize(input->shape());

    MACE_RETURN_IF_ERROR(kernel_->Compute(
        context, input, type, wino_blk_size, output));

    return MaceStatus::MACE_SUCCESS;
  } else {
    LOG(FATAL) << "Unexpected error: " << out_mem_type;
    return MaceStatus::MACE_SUCCESS;
  }
}

enum CopyType {
  COPY_TYPE_MEMCPY,
  COPY_TYPE_TRANSPOSE
};

MaceStatus OpenCLBufferTransformer::TransformAndTranspose(
    OpContext *context,
    const Tensor *input,
    const OpenCLBufferType type,
    const MemoryType out_mem_type,
    const int wino_blk_size,
    const std::vector<int> &dst_dims,
    Tensor *output,
    TransformDebugInfo *info) {
  Workspace *ws = context->workspace();
  DataType dt = output->dtype();
  MemoryType in_mem_type = input->memory_type();
  CopyType copy_type = COPY_TYPE_TRANSPOSE;
  if (out_mem_type == MemoryType::GPU_IMAGE ||
      out_mem_type == MemoryType::GPU_BUFFER) {
    MACE_CHECK(in_mem_type == MemoryType::CPU_BUFFER);
    if (in_mem_type != MemoryType::CPU_BUFFER) {
      return kernel_->Compute(
          context, input, type, wino_blk_size, output);
    } else {
      // convert to the GPU Buffer with the input's data type.
      // 1. CPU buffer to GPU Buffer
      Tensor *internal_tensor = ws->CreateTensor(
          InternalTransformedName(input->name()),
          context->device()->allocator(), input->dtype());
      VLOG(2) << "Transform CPU Buffer " << input->name()
              << " to GPU Buffer " << internal_tensor->name()
              << " with data type " << dt;
      std::vector<index_t> nhwc_shape = input->shape();
      if (input->data_format() == DataFormat::NCHW) {
        nhwc_shape = TransposeUtil::TransposeShape(input->shape(), DST_DIMS_NCHW_TO_NHWC);
      }
      internal_tensor->Resize(nhwc_shape);

      StatsFuture *src_future = context->future();
      StatsFuture future;
      if (info != nullptr) {
        context->set_future(&future);
      }

      const float *input_ptr = input->data<float>();
      Tensor::MappingGuard guard(internal_tensor);
      float *internal_ptr = internal_tensor->mutable_data<float>();
      
      if (copy_type == COPY_TYPE_MEMCPY) {
        memcpy(internal_ptr, input_ptr, input->raw_size());
      } else if (copy_type == COPY_TYPE_TRANSPOSE) {
        Transpose(&context->device()->cpu_runtime()->thread_pool(),
                  input_ptr, input->shape(), dst_dims, internal_ptr);
      }

      // 2. convert the internal GPU Buffer to output.
      MaceStatus status = kernel_->Compute(
          context, internal_tensor, type, wino_blk_size, output);
      if (info != nullptr) {
        future.wait_fn(nullptr);
      }
      
      if (info != nullptr) {
        context->set_future(src_future);
      }

      return status;
    }
  } else if (out_mem_type == MemoryType::CPU_BUFFER) {
    // 1. convert to the GPU Buffer with the output's data type.
    Tensor internal_tensor(context->device()->allocator(),
                           dt,
                           false,
                           InternalTransformedName(input->name()));
    StatsFuture *src_future = context->future();
    StatsFuture future;
    if (info != nullptr) {
      context->set_future(&future);
    }

    MACE_RETURN_IF_ERROR(kernel_->Compute(
        context, input, type, wino_blk_size, &internal_tensor));
    if (info != nullptr) {
      future.wait_fn(nullptr);
    }

    // 2. convert the internal GPU Buffer to output.
    VLOG(2) << "Transform GPU Buffer " << internal_tensor.name()
            << " to CPU Buffer " << output->name()
            << " with data type " << dt;
    
    Tensor::MappingGuard guard(&internal_tensor);
    const float *internal_ptr = internal_tensor.data<float>();
    float *output_ptr = output->mutable_data<float>();

    // NOTE(fucheng): Transpose instead of memcpy.
    if (copy_type == COPY_TYPE_MEMCPY) {
      memcpy(output_ptr, internal_ptr, internal_tensor.size() * sizeof(float));
    } else if (copy_type == COPY_TYPE_TRANSPOSE) {
      Transpose(&context->device()->cpu_runtime()->thread_pool(),
                internal_ptr, internal_tensor.shape(), dst_dims, output_ptr);
    }
    if (info != nullptr) {
      context->set_future(src_future);
    }

    return MaceStatus::MACE_SUCCESS;
  } else {
    LOG(FATAL) << "Unexpected error: " << out_mem_type;
    return MaceStatus::MACE_SUCCESS;
  }
}

MaceStatus OpenCLBufferTransformer::TransformAndPartTranspose(
    OpContext *context,
    const Tensor *input,
    const OpenCLBufferType type,
    const MemoryType out_mem_type,
    const int wino_blk_size,
    const std::vector<int> &dst_dims,
    OdimRanges &odim_ranges,
    Tensor *output) {
  Workspace *ws = context->workspace();
  DataType dt = output->dtype();
  MemoryType in_mem_type = input->memory_type();
  if (out_mem_type == MemoryType::GPU_IMAGE ||
      out_mem_type == MemoryType::GPU_BUFFER) {
    if (in_mem_type != MemoryType::CPU_BUFFER) {
      return kernel_->Compute(
          context, input, type, wino_blk_size, output);
    } else {
      // convert to the GPU Buffer with the input's data type.
      // 1. CPU buffer to GPU Buffer
      Tensor *internal_tensor = ws->CreateTensor(
          InternalTransformedName(input->name()),
          context->device()->allocator(), input->dtype());
      VLOG(2) << "Transform CPU Buffer " << input->name()
              << " to GPU Buffer " << internal_tensor->name()
              << " with data type " << dt;
      std::vector<index_t> nhwc_shape = input->shape();
      if (input->data_format() == DataFormat::NCHW) {
        nhwc_shape = TransposeUtil::TransposeShape(input->shape(),
                                                   DST_DIMS_NCHW_TO_NHWC);
      }
      internal_tensor->Resize(nhwc_shape);
      const float *input_ptr = input->data<float>();
      Tensor::MappingGuard guard(internal_tensor);
      float *internal_ptr = internal_tensor->mutable_data<float>();
      PartTranspose(&context->device()->cpu_runtime()->thread_pool(),
                    input_ptr, input->shape(),
                    dst_dims, odim_ranges, internal_ptr);
      // 2. convert the internal GPU Buffer to output.
      return kernel_->Compute(
          context, internal_tensor, type, wino_blk_size, output);
    }
  } else if (out_mem_type == MemoryType::CPU_BUFFER) {
    // 1. convert to the GPU Buffer with the output's data type.
    Tensor internal_tensor(context->device()->allocator(),
                           dt,
                           false,
                           InternalTransformedName(input->name()));
    MACE_RETURN_IF_ERROR(kernel_->Compute(
        context, input, type, wino_blk_size, &internal_tensor));
    
    // 2. convert the internal GPU Buffer to output.
    VLOG(2) << "Transform GPU Buffer " << internal_tensor.name()
            << " to CPU Buffer " << output->name()
            << " with data type " << dt;
    
    Tensor::MappingGuard guard(&internal_tensor);
    const float *internal_ptr = internal_tensor.data<float>();
    float *output_ptr = output->mutable_data<float>();

    // NOTE(fucheng): Transpose instead of memcpy.
    PartTranspose(&context->device()->cpu_runtime()->thread_pool(),
                  internal_ptr, internal_tensor.shape(),
                  dst_dims, odim_ranges, output_ptr);

    return MaceStatus::MACE_SUCCESS;
  } else {
    LOG(FATAL) << "Unexpected error: " << out_mem_type;
    return MaceStatus::MACE_SUCCESS;
  }
}

#ifdef MACE_ENABLE_CODL

MaceStatus OpenCLPartBufferTransformer::Transform(
    OpContext *context,
    const Tensor *input,
    const OpenCLBufferType type,
    const MemoryType out_mem_type,
    const int wino_blk_size,
    const OdimRanges &odim_ranges,
    Tensor *output) {
  MemoryType in_mem_type = input->memory_type();
  if (out_mem_type == MemoryType::GPU_IMAGE) {
    if (in_mem_type != MemoryType::CPU_BUFFER) {
      return kernel_->Compute(
          context, input, type, wino_blk_size, odim_ranges, output);
    } else {
      MACE_NOT_IMPLEMENTED;
      return MaceStatus::MACE_RUNTIME_ERROR;
    }
  } else if (out_mem_type == MemoryType::GPU_BUFFER) {
    return kernel_->Compute(
        context, input, type, wino_blk_size, odim_ranges, output);
  } else {
    MACE_NOT_IMPLEMENTED;
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
}

#endif  // MACE_ENABLE_CODL

std::string TransformedFilterName(const std::string &name) {
  // TODO(liuqi): This may create a conflict.
  const char *postfix = "_mace_identity_transformed";
  return name + postfix;
}

MaceStatus TransformFilter(
    mace::OpConstructContext *context,
    OperatorDef *op_def,
    const int input_idx,
    const OpenCLBufferType buffer_type,
    const MemoryType mem_type,
    const int wino_blk_size) {
  OpContext op_context(context->workspace(), context->device());
  Workspace *ws = context->workspace();
  std::string input_name = op_def->input(input_idx);
  Tensor *input = ws->GetTensor(input_name);
  const DataType dt = input->dtype();
  VLOG(1) << "input_dt " << static_cast<int>(dt);
  std::string output_name = TransformedFilterName(input_name);
  Tensor *output =
      ws->CreateTensor(output_name, context->device()->allocator(), dt, true);

  // update the information
  op_def->set_input(input_idx, output_name);
  input->MarkUnused();
  return OpenCLBufferTransformer(input->memory_type(), mem_type).
      Transform(&op_context, input, buffer_type, mem_type, wino_blk_size,
                output);
}

}  // namespace ops
}  // namespace mace
