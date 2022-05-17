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

#include "mace/ops/opencl/buffer/fully_connected.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

MaceStatus FullyConnectedKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *weight,
    const Tensor *bias,
    const ActivationType activation,
    const float relux_max_limit,
    const float leakyrelu_coefficient,
    Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t output_size = output->dim(3);
  const index_t output_blocks = RoundUpDiv4(output_size);

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;
  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("fully_connected");
    built_options.emplace("-Dfully_connected=" + kernel_name);
    built_options.emplace("-DIN_DATA_TYPE=" + DtToCLDt(input->dtype()));
    built_options.emplace("-DOUT_DATA_TYPE=" + DtToCLDt(output->dtype()));
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    if (bias != nullptr) {
      built_options.emplace("-DBIAS");
    }
    switch (activation) {
      case NOOP:
        break;
      case RELU:
        built_options.emplace("-DUSE_RELU");
        break;
      case RELUX:
        built_options.emplace("-DUSE_RELUX");
        break;
      case TANH:
        built_options.emplace("-DUSE_TANH");
        break;
      case SIGMOID:
        built_options.emplace("-DUSE_SIGMOID");
        break;
      case LEAKYRELU:
        built_options.emplace("-DUSE_LEAKYRELU");
        break;
      default:
        LOG(FATAL) << "Unknown activation type: " << activation;
    }

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("fully_connected_buffer",
                                              kernel_name,
                                              built_options, &kernel_));
  }

  gws_ = {static_cast<uint32_t>(batch), static_cast<uint32_t>(output_blocks)};
  lws_ = {16, 4};

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    const index_t weight_chan_size = output_blocks * input->dim(3) * 4;
    uint32_t idx = 0;
    MACE_BUFF_OUT_OF_RANGE_SET_ARGS(kernel_, output->size());
    MACE_SET_2D_GWS_ARGS(kernel_, gws_);
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    kernel_.setArg(idx++, *(weight->opencl_buffer()));
    if (bias != nullptr) {
      kernel_.setArg(idx++, *(bias->opencl_buffer()));
    }
    kernel_.setArg(idx++, static_cast<int>(input->dim(1)));
    kernel_.setArg(idx++, static_cast<int>(input->dim(2)));
    kernel_.setArg(idx++, static_cast<int>(input->dim(3)));
    kernel_.setArg(idx++, static_cast<int>(weight_chan_size));
    kernel_.setArg(idx++, static_cast<int>(weight->dim(0)));
    kernel_.setArg(idx++, relux_max_limit);
    kernel_.setArg(idx++, leakyrelu_coefficient);
    kernel_.setArg(idx++, *(output->opencl_buffer()));

    input_shape_ = input->shape();
  }

  cl::Event event;
  cl_int error;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange, cl::NDRange(gws_[0], gws_[1]),
        cl::NDRange(lws_[0], lws_[1]), nullptr, &event);
  } else {
    std::vector<uint32_t> roundup_gws(lws_.size());
    for (size_t i = 0; i < lws_.size(); ++i) {
      roundup_gws[i] = RoundUp(gws_[i], lws_[i]);
    }
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange,
        cl::NDRange(roundup_gws[0], roundup_gws[1]),
        cl::NDRange(lws_[0], lws_[1]), nullptr, &event);
  }
  MACE_OUT_OF_RANGE_VALIDATION;
  MACE_CL_RET_STATUS(error);

  if (context->future() != nullptr) {
    context->future()->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus FullyConnectedKernel::ResizeOutputTensor(
    const Tensor *input,
    const Tensor *weight,
    Tensor *output) {
  std::vector<index_t> output_shape = {input->dim(0), 1, 1, weight->dim(0)};
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace
