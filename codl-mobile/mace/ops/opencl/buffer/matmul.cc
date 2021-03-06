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

#include "mace/ops/opencl/buffer/matmul.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

MaceStatus MatMulKernel::Compute(
    OpContext *context,
    const Tensor *A,
    const Tensor *B,
    Tensor *C,
    bool transpose_a,
    bool transpose_b) {
  MACE_CHECK(!transpose_a, "GPU does not support transpose A matmul");

  index_t rank = A->dim_size() < B->dim_size() ? A->dim_size() : B->dim_size();
  index_t height = A->dim(rank - 2);
  index_t K = A->dim(rank - 1);
  index_t width = transpose_b ? B->dim(rank - 2) : B->dim(rank - 1);
  index_t batch = std::accumulate(A->shape().begin(), A->shape().end() - 2, 1,
                                  std::multiplies<index_t>());

  std::vector<index_t> c_shape = A->shape();
  c_shape[rank - 2] = height;
  c_shape[rank - 1] = width;
  std::vector<index_t> padded_c_shape = {1, batch, height, width};
  C->Resize(padded_c_shape);
#if 1
  LOG(INFO) << "Reisze C:"
            << " shape " << VectorToString<index_t>(c_shape)
            << ", padded_shape " << VectorToString<index_t>(padded_c_shape);
#endif

  const index_t height_blocks = RoundUpDiv4(height);
  const index_t width_blocks = RoundUpDiv4(width);
  const uint32_t gws[2] = {
      static_cast<uint32_t>(width_blocks),
      static_cast<uint32_t>(height_blocks * batch),
  };

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name;
    if (!transpose_b) {
      kernel_name = MACE_OBFUSCATE_SYMBOL("matmul");
      built_options.emplace("-Dmatmul=" + kernel_name);
    } else {
      kernel_name = MACE_OBFUSCATE_SYMBOL("matmul_transpose_b");
      built_options.emplace("-Dmatmul_transpose_b=" + kernel_name);
    }
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("matmul_buffer", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  uint32_t idx = 0;
  MACE_BUFF_OUT_OF_RANGE_SET_ARGS(kernel_, C->size());
  MACE_SET_2D_GWS_ARGS(kernel_, gws);
  kernel_.setArg(idx++, *(A->opencl_buffer()));
  kernel_.setArg(idx++, *(B->opencl_buffer()));
  kernel_.setArg(idx++, static_cast<int>(height));
  kernel_.setArg(idx++, static_cast<int>(width));
  kernel_.setArg(idx++, static_cast<int>(K));
  kernel_.setArg(idx++, static_cast<int>(height_blocks));
  kernel_.setArg(idx++, static_cast<int>(RoundUpDiv4(K)));
  kernel_.setArg(idx++, *(C->opencl_buffer()));

  const std::vector<uint32_t> lws = {kwg_size_ / 64, 64, 0};
  std::string tuning_key = Concat("matmul_opencl_kernel", batch, height, width);
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace
