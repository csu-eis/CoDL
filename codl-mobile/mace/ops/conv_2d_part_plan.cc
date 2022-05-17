
#include <stdio.h>
#include "mace/ops/conv_2d_part_plan.h"
#include "mace/utils/logging.h"

namespace mace {
namespace ops {

bool ConvPool2dPartPlanUtils::CheckInputHeight(
    const index_t height,
    const index_t filter_height,
    const int stride) {
  bool ret = true;
  
  ret = ret && (height >= filter_height);
  ret = ret && ((height - filter_height) % stride == 0);
  //MACE_UNUSED(stride);
  
  if (!ret) {
    LOG(WARNING) << "CheckInputHeight: Invalid height " << height
                 << " (filter height " << filter_height
                 << " stride " << stride << ")";
  }
  
  return ret;
}

bool ConvPool2dPartPlanUtils::IsWinograd(
    const std::vector<index_t> &filter_shape,
    const std::vector<int> &strides,
    const std::vector<int> &dilations) {
  bool is_winograd_flag = filter_shape[H_OIHW] == 3 &&
                          filter_shape[W_OIHW] == 3 &&
                          strides[H_HW] == 1 &&
                          strides[W_HW] == 1 &&
                          dilations[H_HW] == 1 &&
                          dilations[W_HW] == 1 &&
                          filter_shape[I_OIHW] >= 8 &&
                          filter_shape[O_OIHW] >= 8;
  
  return is_winograd_flag;
}

index_t ConvPool2dPartPlanUtils::GetWinogradOutputTileSize(
    const std::vector<index_t> &input_shape) {
  index_t out_tile_size;
  CalcWinogradOutTileSize(input_shape[H_NHWC],
                          input_shape[W_NHWC],
                          &out_tile_size);
  return out_tile_size;
}

bool ConvPool2dPartPlanUtils::IsAlignToWinogradTileSize(
    const std::vector<index_t> &input_shape,
    const index_t length) {
  return (length % GetWinogradOutputTileSize(input_shape) == 0);
}

index_t ConvPool2dPartPlanUtils::AlignToWinogradTileSize(
    const std::vector<index_t> &input_shape,
    const index_t length) {
  const index_t out_tile_size = GetWinogradOutputTileSize(input_shape);
  if (length % out_tile_size == 0) {
    return length;
  } else {
    return length - (length % out_tile_size);
  }
}

void ConvPool2dPartPlanUtils::CalcConv2dInputShape(
    const std::vector<index_t> output_shape,
    const std::vector<index_t> filter_shape,
    const std::vector<int> strides,
    const std::vector<int> paddings,
    const Padding padding_type,
    std::vector<index_t> &input_shape) {
  index_t in_height = 0;
  index_t in_width = 0;
  switch (padding_type) {
    case VALID:
      in_height = (output_shape[H_NHWC] - 1) * strides[H_HW]
                      + filter_shape[H_OIHW];
      in_width = (output_shape[W_NHWC] - 1) * strides[W_HW]
                      + filter_shape[W_OIHW];
      break;
    case SAME:
      if (!paddings.empty()) {
        in_height = (output_shape[H_NHWC] - 1) * strides[H_HW]
                        + filter_shape[H_OIHW];
        in_width = (output_shape[W_NHWC] - 1) * strides[W_HW]
                        + filter_shape[W_OIHW];
      } else {
        //in_height = (output_shape[H_NHWC] - 1) * strides[H_HW] + 1;
        //in_width = (output_shape[W_NHWC] - 1) * strides[W_HW] + 1;
        // NOTE(fucheng): We use same rule to calculate input shape.
        in_height = (output_shape[H_NHWC] - 1) * strides[H_HW]
                        + filter_shape[H_OIHW];
        in_width = (output_shape[W_NHWC] - 1) * strides[W_HW]
                        + filter_shape[W_OIHW];
      }
      break;
    case FULL:
      MACE_CHECK(false, "Unsupported padding type FULL");
      break;
    default:
      MACE_CHECK(false, "Unsupported padding type");
  }

  if (output_shape[H_NHWC] == 0) {
    in_height = 0;
  }

  if (output_shape[W_NHWC] == 0) {
    in_width = 0;
  }

  input_shape = {output_shape[N_NHWC], in_height,
                 in_width, filter_shape[I_OIHW]};

#ifdef CODL_ENABLE_DEBUG_INFO
  LOG(INFO) << "input_shape: " << VectorToString<index_t>(input_shape);
#endif
}

void ConvPool2dPartPlanUtils::CalcConv2dOutputShape(
    const std::vector<index_t> input_shape,
    const std::vector<index_t> filter_shape,
    const std::vector<int> strides,
    const std::vector<int> paddings,
    const Padding padding_type,
    std::vector<index_t> &output_shape) {
  index_t out_height = 0;
  index_t out_width = 0;
  switch (padding_type) {
    case VALID:
      out_height = (input_shape[H_NHWC] - filter_shape[H_OIHW])
                      / strides[H_HW] + 1;
      out_width = (input_shape[W_NHWC] - filter_shape[W_OIHW])
                      / strides[W_HW] + 1;
      break;
    case SAME:
      // Input shape has beed padded.
      if (!paddings.empty()) {
        out_height = (input_shape[H_NHWC] - filter_shape[H_OIHW])
                      / strides[H_HW] + 1;
        out_width = (input_shape[W_NHWC] - filter_shape[W_OIHW])
                      / strides[W_HW] + 1;

        //out_height = (input_shape[1] + paddings[0] - filter_shape[2])
        //                 / strides[0] + 1;
        //out_width = (input_shape[2] + paddings[1] - filter_shape[3])
        //                 / strides[1] + 1;
      } else {
        out_height = (input_shape[H_NHWC] - 1) / strides[H_HW] + 1;
        out_width = (input_shape[W_NHWC] - 1) / strides[W_HW] + 1;
      }
      break;
    case FULL:
      MACE_CHECK(false, "Unsupported padding type FULL");
      break;
    default:
      MACE_CHECK(false, "Unsupported padding type");
  }
  
  output_shape = {input_shape[N_NHWC], out_height,
                  out_width, filter_shape[O_OIHW]};

#ifdef CODL_ENABLE_DEBUG_INFO
  LOG(INFO) << "output_shape: " << VectorToString<index_t>(output_shape);
#endif
}

void ConvPool2dPartPlanUtils::CalcConv2dBiasShape(
    const std::vector<index_t> &output_shape,
    std::vector<index_t> &bias_shape) {
  bias_shape = {output_shape[C_NHWC]};
}

void ConvPool2dPartPlanUtils::CalcPaddings(
    const std::vector<index_t> input_shape,
    const std::vector<index_t> filter_shape,
    const std::vector<int> strides,
    const std::vector<index_t> output_shape,
    std::vector<int> &paddings) {
  const int padding_h = (output_shape[H_NHWC] - 1) * strides[H_HW]
                          + filter_shape[H_OIHW] - input_shape[H_NHWC];
  const int padding_w = (output_shape[W_NHWC]- 1) * strides[W_HW]
                          + filter_shape[W_OIHW] - input_shape[W_NHWC];
  paddings = {padding_h, padding_w};
}

//#define CODL_WINOGRAD_ALIGNMENT_ENABLED

template<>
PartitionResult ConvPool2dPartPlan::BuildRange<CONV2D_DIM_INPUT_HEIGHT>(
    const std::vector<index_t> &input_shape,
    const std::vector<index_t> &filter_shape,
    const std::vector<int> &strides,
    const std::vector<int> &dilations,
    const std::vector<index_t> &output_shape) {
  MACE_UNUSED(dilations);
  // CPU input range.
  index_t input_h_start_cpu = PartPlanUtils::MultipleAndRoundUp(
      input_shape[H_NHWC], ratio_, strides[H_HW]);
  if (input_h_start_cpu > (input_shape[H_NHWC] - 1)) {
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(WARNING) << "input_h_start_cpu " << input_h_start_cpu
                 << " > (input_shape_height - 1) " << (input_shape[H_NHWC] - 1);
#endif
    input_h_start_cpu = input_shape[H_NHWC];
  }
  const index_t input_h_end_cpu = input_shape[H_NHWC] - 1;
  const index_t input_height_cpu = input_h_end_cpu - input_h_start_cpu + 1;
  
  if (IsCpuGpu()) {
    if(!ConvPool2dPartPlanUtils::CheckInputHeight(input_height_cpu,
                                                  filter_shape[H_OIHW],
                                                  strides[H_HW])) {
      LOG(WARNING) << "Invalid CPU input range:"
                   << " h_start " << input_h_start_cpu
                   << " h_end " << input_h_end_cpu
                   << " height " << input_height_cpu;
      return PartitionResult::PARTITION_FAILED;
    }
  }
  
  cpu_input_range_.push_back(input_h_start_cpu);
  cpu_input_range_.push_back(input_h_end_cpu);
  cpu_input_range_.push_back(input_height_cpu);
  
  // GPU input range.
  const index_t input_h_start_gpu = 0;
  const index_t input_h_end_gpu   = (input_h_start_cpu > 0) ?
      (input_h_start_cpu - strides[H_HW] + filter_shape[H_OIHW] - 1) : -1;
  const index_t input_height_gpu  = input_h_end_gpu - input_h_start_gpu + 1;
  
  if (IsCpuGpu()) {
    if(!ConvPool2dPartPlanUtils::CheckInputHeight(input_height_gpu,
                                                  filter_shape[H_OIHW],
                                                  strides[H_HW])) {
      LOG(WARNING) << "Invalid GPU input range:"
                   << " h_start " << input_h_start_gpu
                   << ", h_end " << input_h_end_gpu
                   << ", height " << input_height_gpu;
      return PartitionResult::PARTITION_FAILED;
    }
  }
  
  gpu_input_range_.push_back(input_h_start_gpu);
  gpu_input_range_.push_back(input_h_end_gpu);
  gpu_input_range_.push_back(input_height_gpu);

  // GPU output range.
  const index_t output_h_start_gpu = 0;
  const index_t output_h_end_gpu   = (input_h_start_cpu - strides[H_HW]) / strides[H_HW];
  const index_t output_height_gpu  = output_h_end_gpu - output_h_start_gpu + 1;
  
  gpu_output_range_.push_back(output_h_start_gpu);
  gpu_output_range_.push_back(output_h_end_gpu);
  gpu_output_range_.push_back(output_height_gpu);
  
  // CPU output range.
  const index_t output_h_start_cpu = input_h_start_cpu / strides[H_HW];
  const index_t output_h_end_cpu   = output_shape[H_NHWC] - 1;
  const index_t output_height_cpu  = output_h_end_cpu - output_h_start_cpu + 1;
  
  cpu_output_range_.push_back(output_h_start_cpu);
  cpu_output_range_.push_back(output_h_end_cpu);
  cpu_output_range_.push_back(output_height_cpu);

#ifdef CODL_WINOGRAD_ALIGNMENT_ENABLED
  if (output_height_cpu > 0 &&
      ConvPool2dPartPlanUtils::IsWinograd(filter_shape, strides, dilations)) {
    std::vector<index_t> cpu_input_part_shape = input_shape;
    cpu_input_part_shape[H_NHWC] = input_height_cpu;
    if (!ConvPool2dPartPlanUtils::IsAlignToWinogradTileSize(
        cpu_input_part_shape, output_height_cpu)) {
      LOG(WARNING)
          << "Warning: CPU output height " << output_height_cpu
          << " is not aligned to winograd tile size "
          << ConvPool2dPartPlanUtils::GetWinogradOutputTileSize(cpu_input_part_shape);
    }
  }
#endif

  /**
  LOG(INFO) << "Final ratio: CPU "
            << output_height_cpu * 1.0f / output_shape[1]
            << " (" << output_height_cpu << "/" << output_shape[1]
            << ") GPU " << output_height_gpu * 1.0f / output_shape[1]
            << " (" << output_height_gpu << "/" << output_shape[1] << ")";
  */

  return PartitionResult::PARTITION_SUCCESS;
}

template<>
PartitionResult ConvPool2dPartPlan::BuildRange<CONV2D_DIM_OUTPUT_HEIGHT>(
    const std::vector<index_t> &input_shape,
    const std::vector<index_t> &filter_shape,
    const std::vector<int> &strides,
    const std::vector<int> &dilations,
    const std::vector<index_t> &output_shape) {
  MACE_UNUSED(dilations);
  const index_t output_h_start_gpu = 0;
  index_t output_h_end_gpu;
  index_t output_height_cpu;
  if (ratio_ != kRatioCpuOnly) {
    output_height_cpu =
        output_shape[H_NHWC] - PartPlanUtils::MultipleAndRoundUp(
            output_shape[H_NHWC], ratio_, 1);
    if (output_height_cpu == 0) {
      VLOG(2) << "Output CPU height is 0, we set ratio to 1";
      ratio_ = kRatioGpuOnly;
    }
    output_h_end_gpu = output_shape[H_NHWC] - 1 - output_height_cpu;

#ifdef CODL_WINOGRAD_ALIGNMENT_ENABLED
    if (output_height_cpu > 0 &&
        ConvPool2dPartPlanUtils::IsWinograd(filter_shape, strides, dilations)) {
      output_height_cpu = ConvPool2dPartPlanUtils::AlignToWinogradTileSize(
          input_shape, output_height_cpu);
      if (output_height_cpu == 0 && ratio_ != kRatioGpuOnly) {
        ratio_ = kRatioGpuOnly;
        VLOG(2) << "Output height (CPU) is 0";
        return PartitionResult::PARTITION_REDO;
      }
      output_h_end_gpu = output_shape[H_NHWC] - 1 - output_height_cpu;
    }
#endif
  } else {
    output_h_end_gpu = -1;
    output_height_cpu = output_shape[H_NHWC] - 1 - output_h_end_gpu;
  }
  
  index_t output_height_gpu = output_h_end_gpu - output_h_start_gpu + 1;
  if (ratio_mode_ == RATIO_MODE_MIN_GPU) {
    output_height_gpu = 1;
    output_h_end_gpu = output_h_start_gpu;
  }

  MACE_CHECK(output_height_gpu <= output_shape[H_NHWC],
      output_height_gpu, " > ", output_shape[H_NHWC]);

  gpu_output_range_.push_back(output_h_start_gpu);
  gpu_output_range_.push_back(output_h_end_gpu);
  gpu_output_range_.push_back(output_height_gpu);

  const index_t output_h_end_cpu = output_shape[H_NHWC] - 1;
  const index_t output_h_start_cpu = output_h_end_cpu - output_height_cpu + 1;

  cpu_output_range_.push_back(output_h_start_cpu);
  cpu_output_range_.push_back(output_h_end_cpu);
  cpu_output_range_.push_back(output_height_cpu);

  const index_t input_h_start_gpu = 0;
  const index_t input_h_end_gpu =
      output_h_end_gpu >= 0 ?
      output_h_end_gpu * strides[H_HW] + filter_shape[H_OIHW] - 1 :
      -1;
  const index_t input_height_gpu = input_h_end_gpu - input_h_start_gpu + 1;

  gpu_input_range_.push_back(input_h_start_gpu);
  gpu_input_range_.push_back(input_h_end_gpu);
  gpu_input_range_.push_back(input_height_gpu);

  const index_t input_h_start_cpu = output_h_start_cpu * strides[H_HW];
  const index_t input_h_end_cpu = input_shape[H_NHWC] - 1;
  index_t input_height_cpu = input_h_end_cpu - input_h_start_cpu + 1;
  if (output_height_cpu == 0) {
    input_height_cpu = 0;
  }

  cpu_input_range_.push_back(input_h_start_cpu);
  cpu_input_range_.push_back(input_h_end_cpu);
  cpu_input_range_.push_back(input_height_cpu);

#ifdef CODL_WINOGRAD_ALIGNMENT_ENABLED
  if (output_height_cpu > 0 &&
      ConvPool2dPartPlanUtils::IsWinograd(filter_shape, strides, dilations)) {
    std::vector<index_t> cpu_input_part_shape = input_shape;
    cpu_input_part_shape[H_NHWC] = input_height_cpu;
    if (!ConvPool2dPartPlanUtils::IsAlignToWinogradTileSize(
        cpu_input_part_shape, output_height_cpu)) {
      LOG(WARNING)
          << "Warning: CPU output height " << output_height_cpu
          << " is not aligned to winograd tile size "
          << ConvPool2dPartPlanUtils::GetWinogradOutputTileSize(cpu_input_part_shape);
    }
  }
#endif

  return PartitionResult::PARTITION_SUCCESS;
}

template<>
PartitionResult ConvPool2dPartPlan::BuildRange<CONV2D_DIM_OUTPUT_CHANNEL>(
    const std::vector<index_t> &input_shape,
    const std::vector<index_t> &filter_shape,
    const std::vector<int> &strides,
    const std::vector<int> &dilations,
    const std::vector<index_t> &output_shape) {
  MACE_UNUSED(strides);
  MACE_UNUSED(dilations);
  MACE_UNUSED(output_shape);

  const index_t in_ch = input_shape[C_NHWC];
  gpu_input_range_.push_back(0);
  gpu_input_range_.push_back(in_ch - 1);
  gpu_input_range_.push_back(in_ch);

  cpu_input_range_ = gpu_input_range_;
  
  const index_t out_ch = filter_shape[O_OIHW];
  index_t out_ch_gpu = out_ch * ratio_;
  out_ch_gpu = PartPlanUtils::RoundUp(out_ch_gpu, 4);
  if (out_ch_gpu > out_ch) {
    if (ratio_ < kRatioGpuOnly) {
      VLOG(2) << "Calculated output channels > output channels"
              << " (" << out_ch_gpu << " > " << out_ch << ")";
      ratio_ = kRatioGpuOnly;
      return PartitionResult::PARTITION_REDO;
    } else {
      // Set output channel on gpu as default.
      out_ch_gpu = out_ch;
    }
  }
  index_t out_ch_cpu = out_ch - out_ch_gpu;
  if (out_ch_cpu == 0) {
    VLOG(2) << "CPU output channel is 0, we set ratio as GPU only";
    ratio_ = kRatioGpuOnly;
  }
  
  gpu_output_range_.push_back(0);
  gpu_output_range_.push_back(out_ch_gpu - 1);
  gpu_output_range_.push_back(out_ch_gpu);
  
  cpu_output_range_.push_back(out_ch_gpu);
  cpu_output_range_.push_back(out_ch - 1);
  cpu_output_range_.push_back(out_ch_cpu);
  
  return PartitionResult::PARTITION_SUCCESS;
}

void ConvPool2dPartPlan::BuildShape(const std::vector<index_t> &input_shape,
                                    const std::vector<index_t> &filter_shape,
                                    const std::vector<index_t> &output_shape) {
  if (dim_ == DIM_INPUT_HEIGHT) {
    // Build partial shape for input/output.
    gpu_input_part_shape_.push_back(input_shape[N_NHWC]);    // N
    gpu_input_part_shape_.push_back(gpu_input_range_[2]);    // H
    gpu_input_part_shape_.push_back(input_shape[W_NHWC]);    // W
    gpu_input_part_shape_.push_back(input_shape[C_NHWC]);    // C
    
    gpu_output_part_shape_.push_back(output_shape[N_NHWC]);  // N
    gpu_output_part_shape_.push_back(gpu_output_range_[2]);  // H
    gpu_output_part_shape_.push_back(output_shape[W_NHWC]);  // W
    gpu_output_part_shape_.push_back(output_shape[C_NHWC]);  // C
    
    // Cases of various data format of CPU input/output.
    if (cpu_in_out_data_format_ == DataFormat::NCHW) {
      cpu_input_part_shape_.push_back(input_shape[N_NHWC]);    // N
      cpu_input_part_shape_.push_back(input_shape[C_NHWC]);    // C
      cpu_input_part_shape_.push_back(cpu_input_range_[2]);    // H
      cpu_input_part_shape_.push_back(input_shape[W_NHWC]);    // W
      
      cpu_output_part_shape_.push_back(output_shape[N_NHWC]);  // N
      cpu_output_part_shape_.push_back(output_shape[C_NHWC]);  // C
      cpu_output_part_shape_.push_back(cpu_output_range_[2]);  // H
      cpu_output_part_shape_.push_back(output_shape[W_NHWC]);  // W
    } else if (cpu_in_out_data_format_ == DataFormat::NHWC) {
      cpu_input_part_shape_.push_back(input_shape[N_NHWC]);    // N
      cpu_input_part_shape_.push_back(cpu_input_range_[2]);    // H
      cpu_input_part_shape_.push_back(input_shape[W_NHWC]);    // W
      cpu_input_part_shape_.push_back(input_shape[C_NHWC]);    // C
      
      cpu_output_part_shape_.push_back(output_shape[N_NHWC]);  // N
      cpu_output_part_shape_.push_back(cpu_output_range_[2]);  // H
      cpu_output_part_shape_.push_back(output_shape[W_NHWC]);  // W
      cpu_output_part_shape_.push_back(output_shape[C_NHWC]);  // C
    } else {
      LOG(ERROR) << "Not supported data format while building shape";
      MACE_NOT_IMPLEMENTED;
    }

    // Copy full shape for filter.
    gpu_filter_part_shape_ = filter_shape;

    if (cpu_in_out_data_format_ == DataFormat::NCHW) {
      cpu_filter_part_shape_ = filter_shape;
    } else if (cpu_in_out_data_format_ == DataFormat::NHWC) {
      if (cpu_dtype_ == DataType::DT_FLOAT) {
        cpu_filter_part_shape_ = filter_shape;
      } else if (cpu_dtype_ == DataType::DT_UINT8) {
        cpu_filter_part_shape_.push_back(filter_shape[0]);  // O
        cpu_filter_part_shape_.push_back(filter_shape[2]);  // H
        cpu_filter_part_shape_.push_back(filter_shape[3]);  // W
        cpu_filter_part_shape_.push_back(filter_shape[1]);  // I
      } else {
        LOG(ERROR) << "Not supported data type while building shape";
        MACE_NOT_IMPLEMENTED;
      }
    } else {
      LOG(ERROR) << "Not supported data format while building shape";
      MACE_NOT_IMPLEMENTED;
    }
  } else if (dim_ == DIM_OUTPUT_CHANNEL) {
    // Build partial shape for filter/output.
    if (cpu_in_out_data_format_ == DataFormat::NCHW) {
      cpu_filter_part_shape_.push_back(cpu_output_range_[2]);  // O
      cpu_filter_part_shape_.push_back(filter_shape[I_OIHW]);  // I
      cpu_filter_part_shape_.push_back(filter_shape[H_OIHW]);  // H
      cpu_filter_part_shape_.push_back(filter_shape[W_OIHW]);  // W
    } else if (cpu_in_out_data_format_ == DataFormat::NHWC) {
      if (cpu_dtype_ == DataType::DT_FLOAT) {
        cpu_filter_part_shape_.push_back(cpu_output_range_[2]);  // O
        cpu_filter_part_shape_.push_back(filter_shape[I_OIHW]);  // I
        cpu_filter_part_shape_.push_back(filter_shape[H_OIHW]);  // H
        cpu_filter_part_shape_.push_back(filter_shape[W_OIHW]);  // W
      } else if (cpu_dtype_ == DataType::DT_UINT8) {
        cpu_filter_part_shape_.push_back(cpu_output_range_[2]);  // O
        cpu_filter_part_shape_.push_back(filter_shape[H_OIHW]);  // H
        cpu_filter_part_shape_.push_back(filter_shape[W_OIHW]);  // W
        cpu_filter_part_shape_.push_back(filter_shape[I_OIHW]);  // I
      } else {
        LOG(ERROR) << "Not supported data type while building shape";
        MACE_NOT_IMPLEMENTED;
      }
    }

    gpu_filter_part_shape_.push_back(gpu_output_range_[2]);
    gpu_filter_part_shape_.push_back(filter_shape[I_OIHW]);
    gpu_filter_part_shape_.push_back(filter_shape[H_OIHW]);
    gpu_filter_part_shape_.push_back(filter_shape[W_OIHW]);

    gpu_output_part_shape_.push_back(output_shape[N_NHWC]);
    gpu_output_part_shape_.push_back(output_shape[H_NHWC]);
    gpu_output_part_shape_.push_back(output_shape[W_NHWC]);
    gpu_output_part_shape_.push_back(gpu_output_range_[2]);

    gpu_input_part_shape_ = input_shape;

    // Cases of various data format of CPU input/output.
    if (cpu_in_out_data_format_ == DataFormat::NCHW) {
      cpu_output_part_shape_.push_back(output_shape[N_NHWC]);
      cpu_output_part_shape_.push_back(cpu_output_range_[2]);
      cpu_output_part_shape_.push_back(output_shape[H_NHWC]);
      cpu_output_part_shape_.push_back(output_shape[W_NHWC]);

      cpu_input_part_shape_.push_back(input_shape[N_NHWC]);
      cpu_input_part_shape_.push_back(input_shape[C_NHWC]);
      cpu_input_part_shape_.push_back(input_shape[H_NHWC]);
      cpu_input_part_shape_.push_back(input_shape[W_NHWC]);
    } else if (cpu_in_out_data_format_ == DataFormat::NHWC) {
      cpu_output_part_shape_.push_back(output_shape[N_NHWC]);
      cpu_output_part_shape_.push_back(output_shape[H_NHWC]);
      cpu_output_part_shape_.push_back(output_shape[W_NHWC]);
      cpu_output_part_shape_.push_back(cpu_output_range_[2]);

      cpu_input_part_shape_.push_back(input_shape[N_NHWC]);
      cpu_input_part_shape_.push_back(input_shape[H_NHWC]);
      cpu_input_part_shape_.push_back(input_shape[W_NHWC]);
      cpu_input_part_shape_.push_back(input_shape[C_NHWC]);
    } else {
      LOG(ERROR) << "Not supported data format while building shape";
      MACE_NOT_IMPLEMENTED;
    }

  } else {
    LOG(ERROR) << "Unsupported partition dimension";
    MACE_NOT_IMPLEMENTED;
  }
}

void ConvPool2dPartPlan::BuildOdimRanges() {
  input_odim_ranges_ = OdimRanges(4);
  filter_odim_ranges_ = OdimRanges(4);
  output_odim_ranges_ = OdimRanges(4);

  if (dim_ == DIM_INPUT_HEIGHT) {
    if (cpu_in_out_data_format_ == DataFormat::NCHW) {
      input_odim_ranges_[H_NCHW].push_back(0);
      input_odim_ranges_[H_NCHW].push_back(cpu_input_part_shape_[H_NCHW]);
      input_odim_ranges_[H_NCHW].push_back(cpu_input_range_[0]);
    } else if (cpu_in_out_data_format_ == DataFormat::NHWC) {
      input_odim_ranges_[H_NHWC].push_back(0);
      input_odim_ranges_[H_NHWC].push_back(cpu_input_part_shape_[H_NHWC]);
      input_odim_ranges_[H_NHWC].push_back(cpu_input_range_[0]);
    } else {
      LOG(ERROR) << "Not supported data format while building odim ranges";
      MACE_NOT_IMPLEMENTED;
    }

    output_odim_ranges_[H_NHWC].push_back(cpu_output_range_[0]);
    output_odim_ranges_[H_NHWC].push_back(cpu_output_range_[1] + 1);
    output_odim_ranges_[H_NHWC].push_back(0 - cpu_output_range_[0]);
    
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "Build odim ranges:"
              << " input " << VectorToString<index_t>(input_odim_ranges_[H_NHWC])
              << ", output " << VectorToString<index_t>(output_odim_ranges_[H_NHWC]);
#endif
  } else if (dim_ == DIM_OUTPUT_CHANNEL) {
    input_odim_ranges_[C_NCHW].push_back(0);
    input_odim_ranges_[C_NCHW].push_back(cpu_input_part_shape_[C_NCHW]);
    input_odim_ranges_[C_NCHW].push_back(0);

    filter_odim_ranges_[O_OIHW].push_back(0);
    filter_odim_ranges_[O_OIHW].push_back(cpu_filter_part_shape_[O_OIHW]);
    filter_odim_ranges_[O_OIHW].push_back(cpu_input_range_[0]);

    output_odim_ranges_[C_NHWC].push_back(cpu_output_range_[0]);
    output_odim_ranges_[C_NHWC].push_back(cpu_output_range_[1] + 1);
    output_odim_ranges_[C_NHWC].push_back(0 - cpu_output_range_[0]);

#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "Build odim ranges:"
              << " filter " << VectorToString<index_t>(filter_odim_ranges_[O_OIHW])
              << ", output " << VectorToString<index_t>(output_odim_ranges_[C_NHWC]);
#endif
  } else {
    LOG(ERROR) << "Unsupported partition dimension";
    MACE_NOT_IMPLEMENTED;
  }
}

PartitionResult ConvPool2dPartPlan::Make(
    const std::vector<index_t> input_shape,
    const std::vector<index_t> filter_shape,
    const std::vector<int> strides,
    const std::vector<int> dilations,
    const Padding padding_type,
    const std::vector<int> paddings) {
  MACE_CHECK(padding_type == Padding::VALID || padding_type == Padding::SAME);
  MACE_CHECK(dilations.size() == 2);
  MACE_CHECK(dilations[H_HW] == 1 && dilations[W_HW] == 1);

  filter_shape_ = filter_shape;
  strides_ = strides;
  padding_type_ = padding_type;

  std::vector<index_t> padded_input_shape = input_shape;
  if (!paddings.empty()) {
    MACE_CHECK(paddings.size() == 2);
    paddings_ = paddings;
    padded_input_shape[H_NHWC] += paddings_[H_HW];
    padded_input_shape[W_NHWC] += paddings_[W_HW];
  }

  std::vector<index_t> output_shape;
  ConvPool2dPartPlanUtils::CalcConv2dOutputShape(padded_input_shape,
                                                 filter_shape,
                                                 strides,
                                                 paddings,
                                                 padding_type,
                                                 output_shape);

  if (paddings.empty() && padding_type == Padding::SAME &&
      filter_shape[H_OIHW] > 1 && filter_shape[W_OIHW] > 1) {
    ConvPool2dPartPlanUtils::CalcPaddings(
        input_shape, filter_shape, strides, output_shape, paddings_);
    padded_input_shape[H_NHWC] += paddings_[H_HW];
    padded_input_shape[W_NHWC] += paddings_[W_HW];
  }

#if 0
  LOG(INFO) << "input_shape "    << VectorToString<index_t>(input_shape)
            << ", filter_shape "  << VectorToString<index_t>(filter_shape)
            << ", strides_shape " << VectorToString<int>(strides)
            << ", output_shape "  << VectorToString<index_t>(output_shape)
            << ", paddings " << VectorToString<int>(paddings_)
            << ", padding_type " << padding_type
            << ", padded_input_shape " << VectorToString<index_t>(padded_input_shape);
#endif

  if (ratio_ == kRatioGpuMinimum) {
    ratio_mode_ = RATIO_MODE_MIN_GPU;
    ratio_ = 0.5f;  // To enable CPU+GPU co-execution.
  }

  if (ratio_ >= kRatioCpuOnly && ratio_ <= kRatioGpuOnly) {
    PartitionResult ret;
    switch (dim_) {
      case DIM_INPUT_HEIGHT:
        ret = BuildRange<CONV2D_DIM_OUTPUT_HEIGHT>(padded_input_shape,
                                                   filter_shape,
                                                   strides,
                                                   dilations,
                                                   output_shape);
        break;
      case DIM_OUTPUT_CHANNEL:
        ret = BuildRange<CONV2D_DIM_OUTPUT_CHANNEL>(padded_input_shape,
                                                    filter_shape,
                                                    strides,
                                                    dilations,
                                                    output_shape);
        break;
      default:
        LOG(ERROR) << "Unsupported partition dimension";
        return PartitionResult::PARTITION_FAILED;
    }
    if (ret != PartitionResult::PARTITION_SUCCESS) {
      return ret;
    }
  } else if (ratio_ == kRatioCpuGpuFull) {
    // CPU input range.
    const index_t input_h_start = 0;
    const index_t input_h_end = padded_input_shape[H_NHWC] - 1;
    const index_t input_height = input_h_end - input_h_start + 1;

    cpu_input_range_.push_back(input_h_start);
    cpu_input_range_.push_back(input_h_end);
    cpu_input_range_.push_back(input_height);
    // GPU input range.
    gpu_input_range_.push_back(input_h_start);
    gpu_input_range_.push_back(input_h_end);
    gpu_input_range_.push_back(input_height);
    // CPU output range.
    const index_t output_h_start = 0;
    const index_t output_h_end = output_shape[H_NHWC] - 1;
    const index_t output_height = output_h_end - output_h_start + 1;

    cpu_output_range_.push_back(output_h_start);
    cpu_output_range_.push_back(output_h_end);
    cpu_output_range_.push_back(output_height);
    // GPU output range.
    gpu_output_range_.push_back(output_h_start);
    gpu_output_range_.push_back(output_h_end);
    gpu_output_range_.push_back(output_height);
  } else {
    LOG(ERROR) << "Unsupported partition ratio " << ratio_;
    return PartitionResult::PARTITION_FAILED;
  }
  
  // Build shape.
  BuildShape(padded_input_shape, filter_shape, output_shape);

  // Build output dimension ranges.
  BuildOdimRanges();

  is_ready_ = true;
  
  return PartitionResult::PARTITION_SUCCESS;
}

void ConvPool2dPartPlan::UpdateInputPartShape() {
  if (dim_ == DIM_INPUT_HEIGHT) {
    ConvPool2dPartPlanUtils::CalcConv2dInputShape(gpu_output_part_shape_,
                                                  filter_shape_,
                                                  strides_,
                                                  paddings_,
                                                  padding_type_,
                                                  gpu_input_part_shape_);

    std::vector<index_t> tmp_out_shape;
    std::vector<index_t> tmp_in_shape;
    if (cpu_in_out_data_format_ == DataFormat::NCHW) {
      tmp_out_shape.push_back(cpu_output_part_shape_[0]);
      tmp_out_shape.push_back(cpu_output_part_shape_[2]);
      tmp_out_shape.push_back(cpu_output_part_shape_[3]);
      tmp_out_shape.push_back(cpu_output_part_shape_[1]);
    } else if (cpu_in_out_data_format_ == DataFormat::NHWC) {
      tmp_out_shape = cpu_output_part_shape_;
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    
    ConvPool2dPartPlanUtils::CalcConv2dInputShape(tmp_out_shape,
                                                  filter_shape_,
                                                  strides_,
                                                  paddings_,
                                                  padding_type_,
                                                  tmp_in_shape);

    if (cpu_in_out_data_format_ == DataFormat::NCHW) {
      cpu_input_part_shape_ = std::vector<index_t>(4);
      cpu_input_part_shape_[0] = tmp_in_shape[0];
      cpu_input_part_shape_[1] = tmp_in_shape[3];
      cpu_input_part_shape_[2] = tmp_in_shape[1];
      cpu_input_part_shape_[3] = tmp_in_shape[2];
    } else if (cpu_in_out_data_format_ == DataFormat::NHWC) {
      cpu_input_part_shape_ = tmp_in_shape;
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  } else {
    return;
  }
  
  BuildOdimRanges();
}

void ConvPool2dPartPlan::Show() const {
  if (!is_ready_) {
    return;
  }

  const size_t buf_size = 128;
  char buffer[buf_size];

  VLOG(1) << "===== Part Plan =====";
  VLOG(1) << "Type: Conv2D/Pooling";
  VLOG(1) << "Dim: " << static_cast<int>(dim_);
  VLOG(1) << "Ratio: " << ratio_;
  VLOG(1) << "Filter shape: " << VectorToString<index_t>(filter_shape_);
  VLOG(1) << "Strides: " << VectorToString<int>(strides_);

  snprintf(buffer, buf_size, "GPU input shape range: [%ld,%ld,%ld]",
      gpu_input_range_[0], gpu_input_range_[1], gpu_input_range_[2]);
  VLOG(1) << buffer;
  snprintf(buffer, buf_size, "GPU output shape range: [%ld,%ld,%ld]",
      gpu_output_range_[0], gpu_output_range_[1], gpu_output_range_[2]);
  VLOG(1) << buffer;
  snprintf(buffer, buf_size, "GPU input part shape: [%ld,%ld,%ld,%ld]",
      gpu_input_part_shape_.at(0), gpu_input_part_shape_.at(1),
      gpu_input_part_shape_.at(2), gpu_input_part_shape_.at(3));
  VLOG(1) << buffer;
  snprintf(buffer, buf_size, "GPU filter part shape: [%ld,%ld,%ld,%ld]",
      gpu_filter_part_shape_.at(0), gpu_filter_part_shape_.at(1),
      gpu_filter_part_shape_.at(2), gpu_filter_part_shape_.at(3));
  VLOG(1) << buffer;
  snprintf(buffer, buf_size, "GPU output part shape: [%ld,%ld,%ld,%ld]",
      gpu_output_part_shape_.at(0), gpu_output_part_shape_.at(1),
      gpu_output_part_shape_.at(2), gpu_output_part_shape_.at(3));
  VLOG(1) << buffer;
  snprintf(buffer, buf_size, "CPU input shape range: [%ld,%ld,%ld]",
      cpu_input_range_[0], cpu_input_range_[1], cpu_input_range_[2]);
  VLOG(1) << buffer;
  snprintf(buffer, buf_size, "CPU output shape range: [%ld,%ld,%ld]",
      cpu_output_range_[0], cpu_output_range_[1], cpu_output_range_[2]);
  VLOG(1) << buffer;
  snprintf(buffer, buf_size, "CPU input part shape: [%ld,%ld,%ld,%ld]",
      cpu_input_part_shape_.at(0), cpu_input_part_shape_.at(1),
      cpu_input_part_shape_.at(2), cpu_input_part_shape_.at(3));
  VLOG(1) << buffer;
  snprintf(buffer, buf_size, "CPU filter part shape: [%ld,%ld,%ld,%ld]",
      cpu_filter_part_shape_.at(0), cpu_filter_part_shape_.at(1),
      cpu_filter_part_shape_.at(2), cpu_filter_part_shape_.at(3));
  VLOG(1) << buffer;
  snprintf(buffer, buf_size, "CPU output part shape: [%ld,%ld,%ld,%ld]",
      cpu_output_part_shape_.at(0), cpu_output_part_shape_.at(1),
      cpu_output_part_shape_.at(2), cpu_output_part_shape_.at(3));
  VLOG(1) << buffer;
}

}  // namespace ops
}  // namespace mace
