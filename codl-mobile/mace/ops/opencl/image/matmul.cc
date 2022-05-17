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

#include "mace/ops/opencl/image/matmul.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

void MatMulGlobalWS(const index_t batch,
                    const index_t height_blocks,
                    const index_t width_blocks,
                    uint32_t *gws) {
  gws[0] = static_cast<uint32_t>(width_blocks);
  gws[1] = static_cast<uint32_t>(height_blocks * batch);
}

std::vector<uint32_t> MatMulLocalWS(const uint32_t kwg_size) {
  const std::vector<uint32_t> lws = {kwg_size / 64, 64, 0};
  return lws;
}

MaceStatus MatMulKernel::Compute(
    OpContext *context,
    const Tensor *A,
    const Tensor *B,
    Tensor *C,
    bool transpose_a,
    bool transpose_b) {
#if 0
  MACE_CHECK(!transpose_a && !transpose_b,
             "GPU does not support transpose matmul");
#else
  MACE_CHECK(!transpose_a, "GPU does not support transpose A matmul");
#endif

  index_t rank = A->dim_size() < B->dim_size() ? A->dim_size() : B->dim_size();
  index_t height = A->dim(rank - 2);
  index_t K = A->dim(rank - 1);
  index_t width = transpose_b ? B->dim(rank - 2) : B->dim(rank - 1);
  index_t batch = std::accumulate(A->shape().begin(), A->shape().end() - 2, 1,
                                  std::multiplies<index_t>());

  std::vector<index_t> c_shape = A->shape();
  c_shape[rank - 2] = height;
  c_shape[rank - 1] = width;
  std::vector<size_t> c_image_shape;
#ifndef MACE_ENABLE_CODL
  // NOTE(fucheng): For winograd MM.
  std::vector<index_t> padded_c_shape = {batch, height, width, 1};
  OpenCLBufferType c_buffer_type = OpenCLBufferType::IN_OUT_HEIGHT;
#else
  // NOTE(fucheng): For MM in Transformer model.
  std::vector<index_t> padded_c_shape;
  if (rank == 2) {
    MACE_CHECK(batch == 1);
    padded_c_shape = {height, 1, 1, width};
  } else if (rank == 4) {
    padded_c_shape = {1, batch, height, width};
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  OpenCLBufferType c_buffer_type = OpenCLBufferType::IN_OUT_CHANNEL;
#endif

  OpenCLUtil::CalImage2DShape(padded_c_shape, c_buffer_type, &c_image_shape, 0);
#if 0
  LOG(INFO) << "Reisze C:"
            << " shape " << VectorToString<index_t>(c_shape)
            << ", padded_shape " << VectorToString<index_t>(padded_c_shape)
            << ", image_shape " << VectorToString<size_t>(c_image_shape);
#endif
  MACE_RETURN_IF_ERROR(C->ResizeImage(c_shape, c_image_shape));


  const index_t height_blocks = RoundUpDiv4(height);
  const index_t width_blocks = RoundUpDiv4(width);
#if 0
  const uint32_t gws[2] = {
      static_cast<uint32_t>(width_blocks),
      static_cast<uint32_t>(height_blocks * batch),
  };
#else
  uint32_t gws[2];
  MatMulGlobalWS(batch, height_blocks, width_blocks, gws);
#endif

#if 0
  const std::vector<uint32_t> gwsv = {gws[0], gws[1]};
  LOG(INFO) << "gws " << VectorToString<uint32_t>(gwsv);
#endif

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name;
    switch (rank) {
      case 4:
        if (!transpose_b) {
          kernel_name = "matmul_r4";
          kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
          built_options.emplace("-Dmatmul_r4=" + kernel_name);
        } else {
          kernel_name = "matmul_r4_tb";
          kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
          built_options.emplace("-Dmatmul_r4_tb=" + kernel_name);
        }
        break;
      case 2:
      default:
        kernel_name = "matmul_r2";
        kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
        built_options.emplace("-Dmatmul_r2=" + kernel_name);
    }
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("matmul", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
  MACE_SET_2D_GWS_ARGS(kernel_, gws);
  kernel_.setArg(idx++, *(A->opencl_image()));
  kernel_.setArg(idx++, *(B->opencl_image()));
  kernel_.setArg(idx++, static_cast<int>(height));
  kernel_.setArg(idx++, static_cast<int>(width));
  kernel_.setArg(idx++, static_cast<int>(K));
  kernel_.setArg(idx++, static_cast<int>(height_blocks));
  kernel_.setArg(idx++, static_cast<int>(RoundUpDiv4(K)));
  kernel_.setArg(idx++, *(C->opencl_image()));

#if 0
  const std::vector<uint32_t> lws = {kwg_size_ / 64, 64, 0};
#endif
  const std::vector<uint32_t> lws = MatMulLocalWS(kwg_size_);
  std::string tuning_key = Concat("matmul_opencl_kernel", batch, height, width);
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
