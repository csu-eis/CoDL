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

#include "mace/ops/opencl/image/buffer_to_image.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus BufferToImage::Compute(
    OpContext *context,
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
  MACE_RETURN_IF_ERROR(output->ResizeImage(input->shape(), image_shape));

  uint32_t gws[2] = {static_cast<uint32_t>(image_shape[0]),
                     static_cast<uint32_t>(image_shape[1])};
  std::string kernel_name;
  switch (type) {
    case CONV2D_FILTER:kernel_name = "filter_buffer_to_image";
      break;
    case DW_CONV2D_FILTER:kernel_name = "dw_filter_buffer_to_image";
      break;
    case IN_OUT_CHANNEL:kernel_name = "in_out_buffer_to_image";
      break;
    case ARGUMENT:kernel_name = "arg_buffer_to_image";
      break;
    case IN_OUT_HEIGHT:kernel_name = "in_out_height_buffer_to_image";
      break;
    case IN_OUT_WIDTH:kernel_name = "in_out_width_buffer_to_image";
      break;
    case WEIGHT_HEIGHT:kernel_name = "weight_height_buffer_to_image";
      break;
    case WEIGHT_WIDTH:kernel_name = "weight_width_buffer_to_image";
      break;
    case WINOGRAD_FILTER: {
      std::stringstream ss_tmp;
      gws[1] /= (wino_blk_size + 2) * (wino_blk_size + 2);
      ss_tmp << "winograd_filter_buffer_to_image_"
             << wino_blk_size << "x" << wino_blk_size;
      kernel_name = ss_tmp.str();
      break;
    }
    case GEMM_IN_OUT:kernel_name = "gemm_in_out_buffer_to_image";
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
    if (input->dtype() == output->dtype()) {
      auto input_dt = input->dtype();
      if (input_dt == DataType::DT_INT32) {
        VLOG(2) << "input_dt " << static_cast<int>(input_dt);
        return MaceStatus::MACE_SUCCESS;
      }
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(input_dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(input_dt));
    } else {
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    }

    MACE_RETURN_IF_ERROR(runtime->BuildKernel(
        "buffer_to_image", obfuscated_kernel_name, built_options, &kernel_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_2D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
               "buffer offset not aligned");
    kernel_.setArg(idx++,
                   static_cast<uint32_t>(input->buffer_offset() /
                       GetEnumTypeSize(input->dtype())));
    if (type == CONV2D_FILTER) {
      if (input->dim_size() == 4) {
        const index_t
            inner_size = input->dim(1) * input->dim(2) * input->dim(3);
        kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(0)));
        kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(2)));
        kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(3)));
        kernel_.setArg(idx++, static_cast<uint32_t>(inner_size));
      } else if (input->dim_size() == 2) {
        const index_t inner_size = input->dim(1);
        kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(0)));
        kernel_.setArg(idx++, static_cast<uint32_t>(1));
        kernel_.setArg(idx++, static_cast<uint32_t>(1));
        kernel_.setArg(idx++, static_cast<uint32_t>(inner_size));
      }
    } else if (type == DW_CONV2D_FILTER || type == WEIGHT_HEIGHT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(0)));
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(1)));
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(2)));
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(3)));
    } else if (type == ARGUMENT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(0)));
    } else {
      kernel_.setArg(idx++,
                     static_cast<uint32_t>(formatted_buffer_shape[1]));
      kernel_.setArg(idx++,
                     static_cast<uint32_t>(formatted_buffer_shape[2]));
      kernel_.setArg(idx++,
                     static_cast<uint32_t>(formatted_buffer_shape[3]));
    }
    kernel_.setArg(idx++, *(output->opencl_image()));
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

MaceStatus PartBufferToImage::Compute(
    OpContext *context,
    const Tensor *input,
    const OpenCLBufferType type,
    const int wino_blk_size,
    const OdimRanges &odim_ranges,
    Tensor *output) {
  MACE_CHECK(input->data_format() == DataFormat::NCHW);

  auto formatted_buffer_shape = FormatBufferShape(output->shape(), type);
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(formatted_buffer_shape,
                              type,
                              &image_shape,
                              wino_blk_size);

  const int dim0 = ExtractDim0(odim_ranges);
  const uint32_t offset = odim_ranges[dim0][0];
  const uint32_t part_length = odim_ranges[dim0][1] - odim_ranges[dim0][0];
  MACE_CHECK(part_length > 0, "part length should be more than 0");

  const index_t rank = input->dim_size();

  uint32_t gws[2];
  if (dim0 == N_NHWC) {
    index_t part_image_shape_1 = 0;
    if (rank == 2) {
      part_image_shape_1 = part_length * formatted_buffer_shape[1];
    } else if (rank == 4) {
      part_image_shape_1 = part_length;
    }
    gws[0] = static_cast<uint32_t>(image_shape[0]);
    gws[1] = static_cast<uint32_t>(part_image_shape_1);
  } else if (dim0 == H_NHWC) {
    const index_t part_image_shape_1 = formatted_buffer_shape[0] * part_length;
    gws[0] = static_cast<uint32_t>(image_shape[0]);
    gws[1] = static_cast<uint32_t>(part_image_shape_1);
  } else if (dim0 == C_NHWC) {
    MACE_CHECK(offset % 4 == 0, "channel offset must be aligned to 4");
    const index_t part_image_shape_0 =
        formatted_buffer_shape[2] * RoundUpDiv4(part_length);
    gws[0] = static_cast<uint32_t>(part_image_shape_0);
    gws[1] = static_cast<uint32_t>(image_shape[1]);
  } else {
    LOG(ERROR) << "Unsupported dimension in partial transform";
    MACE_NOT_IMPLEMENTED;
  }

  std::string kernel_name;
  switch (type) {
    case IN_OUT_CHANNEL:
      if (dim0 == N_NHWC) {
        if (rank == 2) {
          kernel_name = "in_out_buffer_to_image_n_offset_nchw";
        } else if (rank == 4) {
          kernel_name = "in_out_buffer_to_image_h_offset";
        }
      } else if (dim0 == H_NHWC) {
        kernel_name = "in_out_buffer_to_image_h_offset_nchw";
      } else if (dim0 == C_NHWC) {
        kernel_name = "in_out_buffer_to_image_c_offset_nchw";
      }
      break;
    default:
      LOG(FATAL) << "Not supported opencl buffer type";
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
    if (input->dtype() == output->dtype()) {
      auto input_dt = input->dtype();
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(input_dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(input_dt));
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
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
               "buffer offset not aligned");
    
    //const index_t inner_size = input->dim(2) * input->dim(3);
    index_t inner_size = 1;
    if (rank == 4) {
      inner_size = formatted_buffer_shape[2] * formatted_buffer_shape[3];
    }
    //const index_t input_offset = input->buffer_offset() /
    //                                GetEnumTypeSize(input->dtype());
    
    kernel_.setArg(idx++, static_cast<uint32_t>(
        input->buffer_offset() / GetEnumTypeSize(input->dtype())));
    kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[1]));
    kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[2]));
    kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[3]));
    kernel_.setArg(idx++, static_cast<uint32_t>(offset));
    kernel_.setArg(idx++, static_cast<uint32_t>(part_length));
    kernel_.setArg(idx++, static_cast<uint32_t>(inner_size));
    kernel_.setArg(idx++, *(output->opencl_image()));
    
    input_shape_ = input->shape();
  }

  cl::Event event;
  cl_int error;
  const uint32_t kwg_size =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  const std::vector<uint32_t> lws = {16, kwg_size / 16};

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
