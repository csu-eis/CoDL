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

#ifndef MACE_OPS_OPENCL_BUFFER_TRANSFORMER_H_
#define MACE_OPS_OPENCL_BUFFER_TRANSFORMER_H_

#include <memory>
#include <string>
#include <vector>

#include "mace/core/operator.h"
#include "mace/ops/opencl/image/buffer_to_image.h"
#include "mace/ops/opencl/image/image_to_buffer.h"
#include "mace/ops/opencl/buffer/buffer_transform.h"
#include "mace/utils/memory.h"
#include "mace/utils/op_delay_tool.h"

namespace mace {
namespace ops {

class TransformDebugInfo {
public:
  TransformDebugInfo(const int num_items) : num_items_(num_items) {}

  void add_latency(double l) {
    values_.push_back(l);
  }

  void print() {
    LOG(INFO) << "Transform Debug Information";
    if (num_items_ > 0) {
      MACE_CHECK(values_.size() % num_items_ == 0);
      int num_data_lines = values_.size() / num_items_;
      for (int i = 0; i < num_data_lines; i ++) {
        std::vector<double> line_values;
        for (int j = 0; j < num_items_; j ++) {
          line_values.push_back(values_[i * num_items_ + j]);
        }
        LOG(INFO) << VectorToString<double>(line_values);
      }
    } else {
      LOG(INFO) << VectorToString<double>(values_);
    }
  }

  void print_avg(const int si) {
    if (num_items_ > 0) {
      LOG(INFO) << "Transform Debug Information";
      MACE_CHECK(values_.size() % num_items_ == 0);
      const int num_data_lines = values_.size() / num_items_;
      std::vector<double> avg_data(num_items_, 0);
      for (int i = si; i < num_data_lines; i ++) {
        for (int j = 0; j < num_items_; j ++) {
          avg_data[j] += values_[i * num_items_ + j];
        }
      }
      for (int i = 0; i < num_items_; i ++) {
        avg_data[i] /= (num_data_lines - si + 1);
      }

      LOG(INFO) << VectorToString<double>(avg_data);
    }
  }

private:
  int num_items_;
  std::vector<double> values_;
};

// Only used for GPU Operation (BufferTransform)
class OpenCLBufferTransformer {
 public:
  OpenCLBufferTransformer(const MemoryType in_mem_type,
                          const MemoryType out_mem_type) {
    if (out_mem_type == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::BufferToImage>();
    } else if (in_mem_type == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ImageToBuffer>();
    } else {
      kernel_ = make_unique<opencl::buffer::BufferTransform>();
    }
  }

  MaceStatus Transform(OpContext *context,
                       const Tensor *input,
                       const OpenCLBufferType type,
                       const MemoryType out_mem_type,
                       const int wino_blk_size,
                       Tensor *output);

  MaceStatus TransformNoInternalTensor(OpContext *context,
                                       const Tensor *input,
                                       const OpenCLBufferType type,
                                       const MemoryType out_mem_type,
                                       const int wino_blk_size,
                                       Tensor *output);

  MaceStatus TransformAndTranspose(
      OpContext *context,
      const Tensor *input,
      const OpenCLBufferType type,
      const MemoryType out_mem_type,
      const int wino_blk_size,
      const std::vector<int> &dst_dims,
      Tensor *output,
      TransformDebugInfo *info);

  MaceStatus TransformAndPartTranspose(
      OpContext *context,
      const Tensor *input,
      const OpenCLBufferType type,
      const MemoryType out_mem_type,
      const int wino_blk_size,
      const std::vector<int> &dst_dims,
      OdimRanges &odim_ranges,
      Tensor *output);

 private:
  std::string InternalTransformedName(const std::string &name) {
    const char *postfix = "_mace_identity_internal";
    return name + postfix;
  }

 private:
  std::unique_ptr<OpenCLBufferTransformKernel> kernel_;
  //Tensor *internal_tensor_;
};

#ifdef MACE_ENABLE_CODL

class OpenCLPartBufferTransformer {
public:
  OpenCLPartBufferTransformer(const MemoryType in_mem_type,
                              const MemoryType out_mem_type) {
    if (out_mem_type == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::PartBufferToImage>();
    } else if (in_mem_type == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::PartImageToBuffer>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Transform(OpContext *context,
                       const Tensor *input,
                       const OpenCLBufferType type,
                       const MemoryType out_mem_type,
                       const int wino_blk_size,
                       const OdimRanges &odim_ranges,
                       Tensor *output);

private:
  std::string InternalTransformedName(const std::string &name) {
    const char *postfix = "_mace_identity_internal";
    return name + postfix;
  }

private:
  std::unique_ptr<OpenCLPartBufferTransformKernel> kernel_;
  //Tensor *internal_tensor_;
};

#endif  // MACE_ENABLE_CODL

std::string TransformedFilterName(const std::string &name);

MaceStatus TransformFilter(
    mace::OpConstructContext *context,
    OperatorDef *op_def,
    const int input_idx,
    const OpenCLBufferType buffer_type,
    const MemoryType mem_type,
    const int wino_blk_size = 0);

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_TRANSFORMER_H_
