
#include "mace/core/tensor.h"
#include "mace/ops/deconv_2d_part_plan.h"

namespace mace {
namespace ops {

void Deconv2dPartPlanUtils::CalcOutputShape(
    const Tensor *output_shape_tensor,
    const std::vector<index_t> &input_shape,
    const std::vector<index_t> &filter_shape,
    const std::vector<int> &strides,
    const Padding padding_type,
    const std::vector<int> &paddings,
    const FrameworkType framework_type,
    std::vector<index_t> &out_shape) {
  if (output_shape_tensor) {
    Tensor::MappingGuard out_shape_guard(output_shape_tensor);
    MACE_CHECK(output_shape_tensor->size() == 4, "output shape should be 4-dims");
    out_shape =
        std::vector<index_t>(output_shape_tensor->data<int32_t>(),
                             output_shape_tensor->data<int32_t>() + 4);
  } else {
    index_t output_height = 0, output_width = 0;
    if (padding_type == Padding::SAME) {
      output_height = input_shape[H_NHWC] * strides[H_HW];
      output_width = input_shape[W_NHWC] * strides[W_HW];
    } else if (padding_type == Padding::VALID) {
      output_height = input_shape[H_NHWC] * strides[H_HW] + filter_shape[H_OIHW] - 1;
      output_width = input_shape[W_NHWC] * strides[W_HW] + filter_shape[W_OIHW] - 1;
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    out_shape = {input_shape[N_NHWC],
                 output_height,
                 output_width,
                 filter_shape[O_OIHW]};
  }

  std::vector<int> in_pad_size;
  std::vector<int> out_pad_size;
  std::vector<index_t> padded_out_shape;
  CalDeconvOutputShapeAndPadSize(input_shape,
                                 filter_shape,
                                 strides,
                                 padding_type,
                                 paddings,
                                 {0, 0},
                                 1,
                                 &out_shape,
                                 &in_pad_size,
                                 &out_pad_size,
                                 &padded_out_shape,
                                 framework_type,
                                 DataFormat::NHWC);
}

void Deconv2dPartPlanUtils::CalcBiasShape(
    const std::vector<index_t> &out_shape,
    std::vector<index_t> &bias_shape) {
  bias_shape = {out_shape[C_NHWC]};
}

PartitionResult Deconv2dPartPlan::BuildRange(
    const std::vector<index_t> &input_shape,
    const std::vector<index_t> &filter_shape) {
  const index_t in_ch = input_shape[C_NHWC];
  gpu_input_range_.push_back(0);
  gpu_input_range_.push_back(in_ch - 1);
  gpu_input_range_.push_back(in_ch);

  cpu_input_range_ = gpu_input_range_;
  
  const index_t out_ch = filter_shape[O_OIHW];
  index_t out_ch_gpu = out_ch * ratio_;
  out_ch_gpu = PartPlanUtils::RoundUp(out_ch_gpu, 4);
  if (out_ch_gpu > out_ch) {
    if (ratio_ < 1.0f) {
      LOG(WARNING) << "Calculated output channels > output channels"
                   << " (" << out_ch_gpu << " > " << out_ch << ")";
      ratio_ = 1.0f;
      return PartitionResult::PARTITION_REDO;
    } else {
      out_ch_gpu = out_ch;
    }
  }
  index_t out_ch_cpu = out_ch - out_ch_gpu;
  
  gpu_output_range_.push_back(0);
  gpu_output_range_.push_back(out_ch_gpu - 1);
  gpu_output_range_.push_back(out_ch_gpu);
  
  cpu_output_range_.push_back(out_ch_gpu);
  cpu_output_range_.push_back(out_ch - 1);
  cpu_output_range_.push_back(out_ch_cpu);
  
  return PartitionResult::PARTITION_SUCCESS;
}

void Deconv2dPartPlan::BuildShape(
    const std::vector<index_t> &input_shape,
    const std::vector<index_t> &filter_shape,
    const std::vector<index_t> &output_shape) {
  MACE_CHECK(input_shape.size() == 4);
  MACE_CHECK(filter_shape.size() == 4);
  MACE_CHECK(output_shape.size() == 4);
  MACE_CHECK(cpu_output_range_.size() == 3);
  MACE_CHECK(gpu_output_range_.size() == 3);
  if (dim_ == DIM_OUTPUT_CHANNEL) {
    // Build partial shape for filter/output.
    cpu_filter_part_shape_.push_back(cpu_output_range_[2]);
    cpu_filter_part_shape_.push_back(filter_shape[I_OIHW]);
    cpu_filter_part_shape_.push_back(filter_shape[H_OIHW]);
    cpu_filter_part_shape_.push_back(filter_shape[W_OIHW]);
    
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

void Deconv2dPartPlan::BuildOdimRanges() {
  input_odim_ranges_ = OdimRanges(4);
  filter_odim_ranges_ = OdimRanges(4);
  output_odim_ranges_ = OdimRanges(4);

  if (dim_ == DIM_OUTPUT_CHANNEL) {
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

PartitionResult Deconv2dPartPlan::Make(
    const Tensor *output_shape_tensor,
    const std::vector<index_t> input_shape,
    const std::vector<index_t> filter_shape,
    const std::vector<int> strides,
    const Padding padding_type,
    const std::vector<int> paddings,
    const FrameworkType framework_type) {
  filter_shape_ = filter_shape;
  strides_ = strides;

  if (ratio_ == kRatioGpuMinimum) {
    ratio_mode_ = RATIO_MODE_MIN_GPU;
    ratio_ = 0.5f;  // To enable CPU+GPU co-execution.
  }

  std::vector<index_t> output_shape(4, 0);
  Deconv2dPartPlanUtils::CalcOutputShape(output_shape_tensor,
                                         input_shape,
                                         filter_shape,
                                         strides,
                                         padding_type,
                                         paddings,
                                         framework_type,
                                         output_shape);

  if (ratio_ >= kRatioCpuOnly && ratio_ <= kRatioGpuOnly) {
    PartitionResult ret;
    switch (dim_) {
      case DIM_OUTPUT_CHANNEL:
        //LOG(INFO) << "Build range";
        ret = BuildRange(input_shape, filter_shape);
        break;
      default:
        LOG(ERROR) << "Unsupported partition dimension";
        return PartitionResult::PARTITION_FAILED;
    }
    if (ret != PartitionResult::PARTITION_SUCCESS) {
      return ret;
    }
  } else if (ratio_ == kRatioCpuGpuFull) {
    LOG(ERROR) << "Unsupported partition ratio " << ratio_;
    return PartitionResult::PARTITION_FAILED;
  } else {
    LOG(ERROR) << "Unsupported partition ratio " << ratio_;
    return PartitionResult::PARTITION_FAILED;
  }

  //LOG(INFO) << "Build shape";
  BuildShape(input_shape, filter_shape, output_shape);

  //LOG(INFO) << "Build odim range";
  BuildOdimRanges();

  is_ready_ = true;
  
  return PartitionResult::PARTITION_SUCCESS;
}

void Deconv2dPartPlan::Show() const {
  if (!is_ready_) {
    return;
  }

  const size_t buf_size = 128;
  char buffer[buf_size];

  LOG(INFO) << "===== Part Plan =====";
  LOG(INFO) << "Type: Deconv2D";
  LOG(INFO) << "Dim: " << static_cast<int>(dim_);
  LOG(INFO) << "Ratio: " << ratio_;
  LOG(INFO) << "Filter shape: " << VectorToString<index_t>(filter_shape_);
  LOG(INFO) << "Strides: " << VectorToString<int>(strides_);

  snprintf(buffer, buf_size, "GPU input shape range: [%ld,%ld,%ld]",
      gpu_input_range_[0], gpu_input_range_[1], gpu_input_range_[2]);
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "GPU output shape range: [%ld,%ld,%ld]",
      gpu_output_range_[0], gpu_output_range_[1], gpu_output_range_[2]);
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "GPU input part shape: [%ld,%ld,%ld,%ld]",
      gpu_input_part_shape_.at(0), gpu_input_part_shape_.at(1),
      gpu_input_part_shape_.at(2), gpu_input_part_shape_.at(3));
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "GPU filter part shape: [%ld,%ld,%ld,%ld]",
      gpu_filter_part_shape_.at(0), gpu_filter_part_shape_.at(1),
      gpu_filter_part_shape_.at(2), gpu_filter_part_shape_.at(3));
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "GPU output part shape: [%ld,%ld,%ld,%ld]",
      gpu_output_part_shape_.at(0), gpu_output_part_shape_.at(1),
      gpu_output_part_shape_.at(2), gpu_output_part_shape_.at(3));
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "CPU input shape range: [%ld,%ld,%ld]",
      cpu_input_range_[0], cpu_input_range_[1], cpu_input_range_[2]);
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "CPU output shape range: [%ld,%ld,%ld]",
      cpu_output_range_[0], cpu_output_range_[1], cpu_output_range_[2]);
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "CPU input part shape: [%ld,%ld,%ld,%ld]",
      cpu_input_part_shape_.at(0), cpu_input_part_shape_.at(1),
      cpu_input_part_shape_.at(2), cpu_input_part_shape_.at(3));
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "CPU filter part shape: [%ld,%ld,%ld,%ld]",
      cpu_filter_part_shape_.at(0), cpu_filter_part_shape_.at(1),
      cpu_filter_part_shape_.at(2), cpu_filter_part_shape_.at(3));
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "CPU output part shape: [%ld,%ld,%ld,%ld]",
      cpu_output_part_shape_.at(0), cpu_output_part_shape_.at(1),
      cpu_output_part_shape_.at(2), cpu_output_part_shape_.at(3));
  LOG(INFO) << buffer;
}

}  // namespace ops
}  // namespace mace
