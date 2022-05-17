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

#include "mace/core/op_context.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

#define TRANFORM_DATA_TYPE

MaceStatus BufferTypeTransform(
    OpContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    Tensor *output) {
  // Resize V1
  MACE_RETURN_IF_ERROR(output->ResizeLike(input));
  // Resize V2
  //std::vector<index_t> transformed_shape = input->shape();
  //MACE_RETURN_IF_ERROR(output->Resize(transformed_shape));
  // Resize V3
  //std::vector<index_t> transformed_shape;
  //if (input->shape().size() == 4) {
  //  transformed_shape = {input->dim(0), input->dim(1),
  //                       input->dim(2), RoundUp<index_t>(input->dim(3), 4)};
  //} else if (input->shape().size() == 2) {
  //  transformed_shape = {input->dim(0), RoundUp<index_t>(input->dim(1), 4)};
  //} else {
  //  MACE_NOT_IMPLEMENTED;
  //}
  //MACE_RETURN_IF_ERROR(output->Resize(transformed_shape));
  //output->Reshape(input->shape());

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION

  const uint32_t gws =
      static_cast<uint32_t>(RoundUpDiv4(output->size()));
  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
#ifdef TRANFORM_DATA_TYPE_V2
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("transform_data_type_v2");
    built_options.emplace("-Dtransform_data_type_v2=" + kernel_name);
#else
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("transform_data_type");
    built_options.emplace("-Dtransform_data_type=" + kernel_name);
#endif
    built_options.emplace("-DIN_DATA_TYPE=" + DtToCLDt(input->dtype()));
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(output->dtype()));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_transform",
                                              kernel_name,
                                              built_options,
                                              kernel));
  }

  MACE_OUT_OF_RANGE_INIT(*kernel);
  uint32_t idx = 0;
  MACE_BUFF_OUT_OF_RANGE_SET_ARGS(*kernel, output->size());
  kernel->setArg(idx++, gws);
  kernel->setArg(idx++, *(input->opencl_buffer()));
  MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
             "buffer offset not aligned");
  kernel->setArg(idx++,
                 static_cast<uint32_t>(input->buffer_offset() /
                     GetEnumTypeSize(input->dtype())));
#ifdef TRANFORM_DATA_TYPE_V2
  kernel->setArg(idx++, static_cast<int32_t>(output->size()));
#endif
  kernel->setArg(idx++, *(output->opencl_buffer()));

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

#ifdef TRANFORM_DATA_TYPE_V2
#undef TRANFORM_DATA_TYPE_V2
#endif

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace
