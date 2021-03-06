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

#include "mace/ops/opencl/image/split.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus SplitKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const std::vector<Tensor *> &output_list) {
  const size_t outputs_count = output_list.size();
  index_t output_channels = 0;
  std::vector<index_t> output_shape;
  if (input->dim_size() == 4) {
    const index_t input_channels = input->dim(3);
    output_channels = input_channels / outputs_count;
    output_shape = {input->dim(0), input->dim(1), input->dim(2), output_channels};
  } else if (input->dim_size() == 3) {
    MACE_CHECK(axis_ == 1);
    const index_t input_dim = input->dim(axis_);
    const index_t output_dim = input_dim / outputs_count;
    output_shape = {input->dim(0), output_dim, input->dim(2)};
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  VLOG(1) << "input_shape " << VectorToString<index_t>(input->shape())
          << ", output_shape " << VectorToString<index_t>(output_shape);

  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(output_shape,
                              OpenCLBufferType::IN_OUT_CHANNEL,
                              &image_shape);
  for (size_t i = 0; i < outputs_count; ++i) {
    MACE_RETURN_IF_ERROR(
        output_list[i]->ResizeImage(output_shape, image_shape));
  }

  if (input->dim_size() != 4) {
    return MaceStatus::MACE_SUCCESS;
  }

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("split");
    built_options.emplace("-Dsplit=" + kernel_name);
    auto input_dt = input->dtype();
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(input_dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(input_dt));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("split",
                                              kernel_name,
                                              built_options,
                                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  const index_t channel_blk = RoundUpDiv4(output_channels);

  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blk), static_cast<uint32_t>(input->dim(2)),
      static_cast<uint32_t>(input->dim(0) * input->dim(1)),
  };
  MACE_OUT_OF_RANGE_INIT(kernel_);

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  cl::Event event;
  CallStats call_stats{INT64_MAX, 0};
  for (size_t i = 0; i < outputs_count; ++i) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, static_cast<int32_t>(channel_blk * i));
    kernel_.setArg(idx++, *(output_list[i]->opencl_image()));

    cl_int error;
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      error = runtime->command_queue().enqueueNDRangeKernel(
          kernel_, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
          cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
    } else {
      std::vector<uint32_t> roundup_gws(lws.size());
      for (size_t j = 0; j < 3; ++j) {
        roundup_gws[j] = RoundUp(gws[j], lws[j]);
      }

      error = runtime->command_queue().enqueueNDRangeKernel(
          kernel_, cl::NullRange,
          cl::NDRange(roundup_gws[0], roundup_gws[1], roundup_gws[2]),
          cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
    }
    MACE_CL_RET_STATUS(error);
    MACE_OUT_OF_RANGE_VALIDATION;
    if (context->future() != nullptr && runtime->is_profiling_enabled()) {
      event.wait();
      CallStats tmp_stats;
      runtime->GetCallStats(event, &tmp_stats);
      call_stats.start_micros =
          std::min<int64_t>(tmp_stats.start_micros, call_stats.start_micros);
      call_stats.end_micros += tmp_stats.end_micros - tmp_stats.start_micros;
    }
  }
  if (context->future() != nullptr) {
    context->future()->wait_fn = [call_stats](CallStats *stats) {
      if (stats != nullptr) {
        stats->start_micros = call_stats.start_micros;
        stats->end_micros = stats->start_micros + call_stats.end_micros;
      }
    };
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
