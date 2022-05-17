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

#include "mace/ops/opencl/buffer/buffer_transform.h"

#include <vector>
#include <set>
#include <string>

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

MaceStatus TransformConv2DFilter(
    OpContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    Tensor *output) {
  const index_t out_chan = input->dim(0);
  const index_t in_chan = input->dim(1);
  const index_t filter_height = input->dim(2);
  const index_t filter_width = input->dim(3);

  std::vector<index_t> transformed_shape = {
      filter_height,
      filter_width,
      RoundUpDiv4(out_chan),
      RoundUp<index_t>(in_chan, 4),
      4,
  };

  uint32_t gws[3];
  gws[0] = static_cast<uint32_t>(transformed_shape[3]);
  gws[1] = static_cast<uint32_t>(transformed_shape[2]);
  gws[2] = static_cast<uint32_t>(filter_height * filter_width);
  MACE_RETURN_IF_ERROR(output->Resize(transformed_shape));
  output->Reshape(input->shape());

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION
  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_NON_UNIFORM_WG_CONFIG;
    MACE_OUT_OF_RANGE_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("transform_conv_filter");
    built_options.emplace("-Dtransform_conv_filter=" + kernel_name);
    std::string data_dt = DtToCLDt(input->dtype());
    built_options.emplace("-DIN_DATA_TYPE=" + data_dt);
    built_options.emplace("-DDATA_TYPE=" + data_dt);
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_transform",
                                              kernel_name,
                                              built_options,
                                              kernel));
  }
  MACE_OUT_OF_RANGE_INIT(*kernel);

  uint32_t idx = 0;
  MACE_BUFF_OUT_OF_RANGE_SET_ARGS(*kernel, output->UnderlyingBuffer()->size());
  MACE_SET_3D_GWS_ARGS(*kernel, gws);
  kernel->setArg(idx++, *(input->opencl_buffer()));
  MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
             "buffer offset not aligned");
  kernel->setArg(idx++,
                 static_cast<uint32_t>(input->buffer_offset() /
                     GetEnumTypeSize(input->dtype())));
  kernel->setArg(idx++, *(output->opencl_buffer()));
  kernel->setArg(idx++, static_cast<int32_t>(out_chan));
  kernel->setArg(idx++, static_cast<int32_t>(in_chan));
  kernel->setArg(idx++, static_cast<int32_t>(filter_height));
  kernel->setArg(idx++, static_cast<int32_t>(filter_width));
  kernel->setArg(idx++, static_cast<int32_t>(
      in_chan * filter_height * filter_width));

  std::string tuning_key =
      Concat("transform_conv_filter",
             transformed_shape[0],
             transformed_shape[1],
             transformed_shape[2],
             transformed_shape[3]);
  std::vector<uint32_t> lws = {4, 4, 4, 0};
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, *kernel, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus TransformDWConv2DFilter(
    OpContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    Tensor *output) {
  const index_t multiplier = input->dim(0);
  const index_t in_chan = input->dim(1);
  const index_t filter_height = input->dim(2);
  const index_t filter_width = input->dim(3);

  std::vector<index_t> transformed_shape = {
      multiplier, RoundUpDiv4(in_chan),
      filter_height, filter_width, 4,
  };
  uint32_t gws[3];
  gws[0] = static_cast<uint32_t>(filter_width);
  gws[1] = static_cast<uint32_t>(filter_height);
  gws[2] = static_cast<uint32_t>(transformed_shape[1]);
  MACE_RETURN_IF_ERROR(output->Resize(transformed_shape));
  output->Reshape(input->shape());

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION
  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("transform_dw_conv_filter");
    built_options.emplace("-Dtransform_dw_conv_filter=" + kernel_name);
    std::string data_dt = DtToCLDt(input->dtype());
    built_options.emplace("-DIN_DATA_TYPE=" + data_dt);
    built_options.emplace("-DDATA_TYPE=" + data_dt);
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_transform",
                                              kernel_name,
                                              built_options,
                                              kernel));
  }

  MACE_OUT_OF_RANGE_INIT(*kernel);

  uint32_t idx = 0;
  MACE_BUFF_OUT_OF_RANGE_SET_ARGS(*kernel, output->UnderlyingBuffer()->size());
  MACE_SET_3D_GWS_ARGS(*kernel, gws);
  kernel->setArg(idx++, *(input->opencl_buffer()));
  MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
             "buffer offset not aligned");
  kernel->setArg(idx++,
                 static_cast<uint32_t>(input->buffer_offset() /
                     GetEnumTypeSize(input->dtype())));
  kernel->setArg(idx++, *(output->opencl_buffer()));
  kernel->setArg(idx++, static_cast<int32_t>(in_chan));
  kernel->setArg(idx++, static_cast<int32_t>(filter_height * filter_width));

  std::string tuning_key =
      Concat("transform_conv_filter",
             transformed_shape[0],
             transformed_shape[1],
             transformed_shape[2],
             transformed_shape[3]);
  std::vector<uint32_t> lws = {4, 4, 4, 0};
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, *kernel, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus TransformArgument(
    OpContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    Tensor *output) {
  const index_t size = input->dim(0);

  std::vector<index_t> transformed_shape = {RoundUpDiv4(size), 4};

  LOG(INFO) << "input_shape " << VectorToString<index_t>(input->shape())
            << " transformed_shape " << VectorToString<index_t>(transformed_shape);

  uint32_t gws = static_cast<uint32_t>(transformed_shape[0]);
  MACE_RETURN_IF_ERROR(output->Resize(transformed_shape));
  output->Reshape(input->shape());

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION
  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("transform_arg");
    built_options.emplace("-Dtransform_arg=" + kernel_name);
    if (input->dtype() == DataType::DT_INT32) {
      return MaceStatus::MACE_SUCCESS;
    }
    std::string data_dt = DtToCLDt(input->dtype());
    built_options.emplace("-DIN_DATA_TYPE=" + data_dt);
    built_options.emplace("-DDATA_TYPE=" + data_dt);
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_transform",
                                              kernel_name,
                                              built_options,
                                              kernel));
  }
  MACE_OUT_OF_RANGE_INIT(*kernel);

  uint32_t idx = 0;
  MACE_BUFF_OUT_OF_RANGE_SET_ARGS(*kernel, output->UnderlyingBuffer()->size());
  kernel->setArg(idx++, gws);
  kernel->setArg(idx++, *(input->opencl_buffer()));
  MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
             "buffer offset not aligned");
  kernel->setArg(idx++,
                 static_cast<uint32_t>(input->buffer_offset() /
                     GetEnumTypeSize(input->dtype())));
  kernel->setArg(idx++, *(output->opencl_buffer()));
  kernel->setArg(idx++, static_cast<int32_t>(size));

  const uint32_t lws =
      static_cast<uint32_t>(RoundUpDiv4(runtime->GetDeviceMaxWorkGroupSize()));
  cl::Event event;
  cl_int error;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        *kernel, cl::NullRange, cl::NDRange(gws),
        cl::NDRange(lws), nullptr, &event);
  } else {
    uint32_t roundup_gws = RoundUp(gws, lws);
    error = runtime->command_queue().enqueueNDRangeKernel(
        *kernel, cl::NullRange, cl::NDRange(roundup_gws),
        cl::NDRange(lws), nullptr, &event);
  }
  MACE_CL_RET_STATUS(error);
  MACE_OUT_OF_RANGE_VALIDATION
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

MaceStatus TransformInOutChannel(
    OpContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    Tensor *output) {
  index_t channel;
  std::vector<index_t> transformed_shape;
  if (input->shape().size() == 4) {
    channel = input->dim(3);
    transformed_shape = {input->dim(0), input->dim(1),
                         input->dim(2), RoundUp<index_t>(input->dim(3), 4)};
  } else if (input->shape().size() == 2) {
    channel = input->dim(1);
    transformed_shape = {input->dim(0), 1, 1,
                         RoundUp<index_t>(input->dim(1), 4)};
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  MACE_RETURN_IF_ERROR(output->Resize(transformed_shape));
  output->Reshape(input->shape());

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION
  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("transform_in_out_channel");
    built_options.emplace("-Dtransform_in_out_channel=" + kernel_name);
    std::string data_dt = DtToCLDt(input->dtype());
    built_options.emplace("-DIN_DATA_TYPE=" + data_dt);
    built_options.emplace("-DDATA_TYPE=" + data_dt);
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_transform",
                                              kernel_name,
                                              built_options,
                                              kernel));
  }
  
  const uint32_t gws[2] = {
      static_cast<uint32_t>(transformed_shape[2] * RoundUpDiv4(channel)),
      static_cast<uint32_t>(transformed_shape[0] * transformed_shape[1])};

  MACE_OUT_OF_RANGE_INIT(*kernel);

  uint32_t idx = 0;
  MACE_BUFF_OUT_OF_RANGE_SET_ARGS(*kernel, output->UnderlyingBuffer()->size());
  MACE_SET_2D_GWS_ARGS(*kernel, gws);
  kernel->setArg(idx++, *(input->opencl_buffer()));
  kernel->setArg(idx++, static_cast<int32_t>(transformed_shape[1]));
  kernel->setArg(idx++, static_cast<int32_t>(transformed_shape[2]));
  kernel->setArg(idx++, static_cast<int32_t>(channel));
  kernel->setArg(idx++, *(output->opencl_buffer()));

  std::vector<uint32_t> lws = {4, 4, 0};
  std::string tuning_key = Concat("transform_in_out_channel",
      transformed_shape[0], transformed_shape[1],
      transformed_shape[2], channel);
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, *kernel, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BufferTransform::Compute(OpContext *context,
                                    const Tensor *input,
                                    const OpenCLBufferType type,
                                    const int wino_blk_size,
                                    Tensor *output) {
  MACE_UNUSED(wino_blk_size);
  switch (type) {
    case CONV2D_FILTER:
      return TransformConv2DFilter(context, &kernel_, input, output);
    case DW_CONV2D_FILTER:
      return TransformDWConv2DFilter(context, &kernel_, input, output);
    case ARGUMENT:
      return TransformArgument(context, &kernel_, input, output);
    case IN_OUT_CHANNEL:
      return TransformInOutChannel(context, &kernel_, input, output);
    default:
      return BufferTypeTransform(context, &kernel_, input, output);
  }
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace
