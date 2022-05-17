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

#include "mace/ops/opencl/image/image_to_buffer.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus ImageToBuffer::Compute(OpContext *context,
                                  const Tensor *input,
                                  const OpenCLBufferType type,
                                  const int wino_blk_size,
                                  Tensor *output) {
  auto formatted_buffer_shape = FormatBufferShape(input->shape(), type);
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(formatted_buffer_shape,
                              type,
                              &image_shape,
                              wino_blk_size);
  MACE_RETURN_IF_ERROR(output->Resize(input->shape()));

  uint32_t gws[2] = {static_cast<uint32_t>(image_shape[0]),
                     static_cast<uint32_t>(image_shape[1])};
  std::string kernel_name;
  switch (type) {
    case CONV2D_FILTER:kernel_name = "filter_image_to_buffer";
      break;
    case IN_OUT_CHANNEL:kernel_name = "in_out_image_to_buffer";
      break;
    case ARGUMENT:kernel_name = "arg_image_to_buffer";
      break;
    case IN_OUT_HEIGHT:kernel_name = "in_out_height_image_to_buffer";
      break;
    case WINOGRAD_FILTER: {
      std::stringstream ss_tmp;
      gws[1] /= (wino_blk_size + 2) * (wino_blk_size + 2);
      ss_tmp << "winograd_filter_image_to_buffer_"
             << wino_blk_size << "x" << wino_blk_size;
      kernel_name = ss_tmp.str();
      break;
    }
    case WEIGHT_HEIGHT:kernel_name = "weight_height_image_to_buffer";
      break;
    case WEIGHT_WIDTH:kernel_name = "weight_width_image_to_buffer";
      break;
    case GEMM_IN_OUT:kernel_name = "gemm_in_out_image_to_buffer";
      break;
    case DW_CONV2D_FILTER:
    case IN_OUT_WIDTH:LOG(FATAL)
          << "IN_OUT_WIDTH only support buffer to image now";
      break;
  }

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    if (output->dtype() == input->dtype()) {
      auto data_dt = input->dtype();
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(data_dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(data_dt));
    } else {
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_to_image",
                                              obfuscated_kernel_name,
                                              built_options,
                                              &kernel_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_2D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(output->opencl_buffer()));
    if (type == CONV2D_FILTER) {
      const index_t
          inner_size = output->dim(1) * output->dim(2) * output->dim(3);
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(0)));
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(2)));
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(3)));
      kernel_.setArg(idx++, static_cast<uint32_t>(inner_size));
    } else if (type == ARGUMENT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(0)));
    } else if (type == WEIGHT_HEIGHT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(0)));
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(1)));
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(2)));
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(3)));
    } else {
      kernel_.setArg(idx++,
                     static_cast<uint32_t>(formatted_buffer_shape[1]));
      kernel_.setArg(idx++,
                     static_cast<uint32_t>(formatted_buffer_shape[2]));
      kernel_.setArg(idx++,
                     static_cast<uint32_t>(formatted_buffer_shape[3]));
    }
    kernel_.setArg(idx++, *(input->opencl_image()));
    input_shape_ = input->shape();
  }

  const uint32_t kwg_size =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  const std::vector<uint32_t> lws = {16, kwg_size / 16};

  cl::Event event;
  cl_int error;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange, cl::NDRange(gws[0], gws[1]),
        cl::NDRange(lws[0], lws[1]), nullptr, &event);
  } else {
    std::vector<uint32_t> roundup_gws(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
      roundup_gws[i] = RoundUp(gws[i], lws[i]);
    }

    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange, cl::NDRange(roundup_gws[0], roundup_gws[1]),
        cl::NDRange(lws[0], lws[1]), nullptr, &event);
  }
  MACE_CL_RET_STATUS(error);
  MACE_OUT_OF_RANGE_VALIDATION;
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

#ifdef MACE_ENABLE_CODL

MaceStatus PartImageToBuffer::Compute(OpContext *context,
                                      const Tensor *input,
                                      const OpenCLBufferType type,
                                      const int wino_blk_size,
                                      const OdimRanges &odim_ranges,
                                      Tensor *output) {
  MACE_CHECK(output->data_format() == DataFormat::NCHW);
  auto formatted_buffer_shape = FormatBufferShape(input->shape(), type);
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(formatted_buffer_shape,
                              type,
                              &image_shape,
                              wino_blk_size);

#if 0
  LOG(INFO) << "input_shape " << VectorToString<index_t>(input->shape())
            << ", formatted_input_shape " << VectorToString<index_t>(formatted_buffer_shape)
            << ", image_shape " << VectorToString<size_t>(image_shape);
#endif

  const int dim0 = ExtractDim0(odim_ranges);
  MACE_CHECK(odim_ranges[dim0].size() == 3);

  const uint32_t offset = odim_ranges[dim0][0] + odim_ranges[dim0][2];
  const uint32_t part_length = odim_ranges[dim0][1] - odim_ranges[dim0][0];

#if 0
  LOG(INFO) << "dim " << dim0
            << ", offset " << offset
            << ", part_length " << part_length;
#endif

  const index_t rank = input->dim_size();

  uint32_t gws[2];
  std::vector<index_t> part_output_shape;
  if (dim0 == N_NCHW) {
    if (rank == 2) {
      const index_t part_image_shape_1 = part_length * formatted_buffer_shape[1];
      gws[0] = static_cast<uint32_t>(image_shape[0]);
      gws[1] = static_cast<uint32_t>(part_image_shape_1);
      part_output_shape.push_back(part_length);
      part_output_shape.push_back(input->shape()[1]);
    } else if (rank == 4) {
      const index_t part_image_shape_1 = part_length;
      gws[0] = static_cast<uint32_t>(image_shape[0]);
      gws[1] = static_cast<uint32_t>(part_image_shape_1);

      part_output_shape.push_back(1);
      part_output_shape.push_back(part_length);
      part_output_shape.push_back(input->shape()[2]);
      part_output_shape.push_back(input->shape()[3]);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  } else if (dim0 == H_NCHW) {
    const index_t part_image_shape_1 = formatted_buffer_shape[0] * part_length;
    gws[0] = static_cast<uint32_t>(image_shape[0]);
    gws[1] = static_cast<uint32_t>(part_image_shape_1);

    part_output_shape.push_back(input->shape()[0]);
    part_output_shape.push_back(input->shape()[3]);
    part_output_shape.push_back(part_length);
    part_output_shape.push_back(input->shape()[2]);
  } else if (dim0 == C_NCHW) {
    MACE_CHECK(offset % 4 == 0, "channel offset must be aligned to 4");
    const index_t part_image_shape_0 =
        formatted_buffer_shape[2] * RoundUpDiv4(part_length);
    gws[0] = static_cast<uint32_t>(part_image_shape_0);
    gws[1] = static_cast<uint32_t>(image_shape[1]);

    part_output_shape.push_back(input->shape()[0]);
    part_output_shape.push_back(part_length);
    part_output_shape.push_back(input->shape()[1]);
    part_output_shape.push_back(input->shape()[2]);
  } else {
    LOG(ERROR) << "Unsupported dimension in partial transform";
    MACE_NOT_IMPLEMENTED;
  }

#if 0
  const std::vector<uint32_t> vgws = {gws[0], gws[1]};
  LOG(INFO) << "gws " << VectorToString<uint32_t>(vgws)
            << ", part_output_shape " << VectorToString<index_t>(part_output_shape);
#endif

  if (output->shape().empty()) {
    MACE_RETURN_IF_ERROR(output->Resize(part_output_shape));
  }

  std::string kernel_name;
  switch (type) {
    case IN_OUT_CHANNEL:
      if (dim0 == N_NHWC) {
        if (rank == 2) {
          kernel_name = "in_out_image_to_buffer_n_offset_nchw";
        } else if (rank == 4) {
          kernel_name = "in_out_image_to_buffer_h_offset";
        }
      } else if (dim0 == H_NCHW) {
        kernel_name = "in_out_image_to_buffer_h_offset_nchw";
      } else if (dim0 == C_NCHW) {
        kernel_name = "in_out_image_to_buffer_c_offset_nchw";
      } else {
        LOG(ERROR) << "Unsupported dimension in partial transform kernel";
        MACE_NOT_IMPLEMENTED;
      }
      break;
    default:
      LOG(FATAL) << "Not supported opencl buffer type";
      break;
  }

#if 0
  LOG(INFO) << "kernel_name " << kernel_name;
#endif

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    if (output->dtype() == input->dtype()) {
      auto data_dt = input->dtype();
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(data_dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(data_dt));
    } else {
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_to_image",
                                              obfuscated_kernel_name,
                                              built_options,
                                              &kernel_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_2D_GWS_ARGS(kernel_, gws);
    
    index_t inner_size = 1;
    if (rank == 4) {
      inner_size = part_output_shape[2] * part_output_shape[3];
    }
    
    kernel_.setArg(idx++, *(output->opencl_buffer()));
    kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[1]));
    kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[2]));
    kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[3]));
    kernel_.setArg(idx++, static_cast<uint32_t>(offset));
    kernel_.setArg(idx++, static_cast<uint32_t>(part_length));
    kernel_.setArg(idx++, static_cast<uint32_t>(inner_size));
    kernel_.setArg(idx++, *(input->opencl_image()));

    input_shape_ = input->shape();
  }

  const uint32_t kwg_size =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  const std::vector<uint32_t> lws = {16, kwg_size / 16};

  cl::Event event;
  cl_int error;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange, cl::NDRange(gws[0], gws[1]),
        cl::NDRange(lws[0], lws[1]), nullptr, &event);
  } else {
    std::vector<uint32_t> roundup_gws(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
      roundup_gws[i] = RoundUp(gws[i], lws[i]);
    }

    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange, cl::NDRange(roundup_gws[0], roundup_gws[1]),
        cl::NDRange(lws[0], lws[1]), nullptr, &event);
  }
  MACE_CL_RET_STATUS(error);
  MACE_OUT_OF_RANGE_VALIDATION;

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

#endif  // MACE_ENABLE_CODL

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
