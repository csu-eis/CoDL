
#include "mace/core/tensor.h"
#include "mace/ops/fully_connected_part_plan.h"

namespace mace {
namespace ops {

PartitionResult FullyConnectedPartPlan::BuildRange(
    const std::vector<index_t> &input_shape,
    const std::vector<index_t> &weight_shape) {
  const index_t in_ch = input_shape[C_NHWC];
  gpu_input_range_.push_back(0);
  gpu_input_range_.push_back(in_ch - 1);
  gpu_input_range_.push_back(in_ch);

  cpu_input_range_ = gpu_input_range_;
  
  const index_t out_ch = weight_shape[O_OIHW];
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

void FullyConnectedPartPlan::BuildShape(
    const std::vector<index_t> &input_shape,
    const std::vector<index_t> &weight_shape,
    const std::vector<index_t> &output_shape) {
  MACE_CHECK(input_shape.size() == 4);
  MACE_CHECK(weight_shape.size() == 4);
  MACE_CHECK(output_shape.size() == 4);
  MACE_CHECK(cpu_output_range_.size() == 3);
  MACE_CHECK(gpu_output_range_.size() == 3);
  if (dim_ == DIM_OUTPUT_CHANNEL) {
    // Build partial shape for weight/output.
    cpu_weight_part_shape_.push_back(cpu_output_range_[2]);
    cpu_weight_part_shape_.push_back(weight_shape[I_OIHW]);
    cpu_weight_part_shape_.push_back(weight_shape[H_OIHW]);
    cpu_weight_part_shape_.push_back(weight_shape[W_OIHW]);
    
    gpu_weight_part_shape_.push_back(gpu_output_range_[2]);
    gpu_weight_part_shape_.push_back(weight_shape[I_OIHW]);
    gpu_weight_part_shape_.push_back(weight_shape[H_OIHW]);
    gpu_weight_part_shape_.push_back(weight_shape[W_OIHW]);

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

void FullyConnectedPartPlan::BuildOdimRanges() {
  input_odim_ranges_ = OdimRanges(4);
  weight_odim_ranges_ = OdimRanges(4);
  output_odim_ranges_ = OdimRanges(4);

  if (dim_ == DIM_OUTPUT_CHANNEL) {
    input_odim_ranges_[C_NCHW].push_back(0);
    input_odim_ranges_[C_NCHW].push_back(cpu_input_part_shape_[C_NCHW]);
    input_odim_ranges_[C_NCHW].push_back(0);

    weight_odim_ranges_[O_OIHW].push_back(0);
    weight_odim_ranges_[O_OIHW].push_back(cpu_weight_part_shape_[O_OIHW]);
    weight_odim_ranges_[O_OIHW].push_back(cpu_input_range_[0]);

    output_odim_ranges_[C_NHWC].push_back(cpu_output_range_[0]);
    output_odim_ranges_[C_NHWC].push_back(cpu_output_range_[1] + 1);
    output_odim_ranges_[C_NHWC].push_back(0 - cpu_output_range_[0]);

#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "Build odim ranges:"
              << " weight " << VectorToString<index_t>(weight_odim_ranges_[O_OIHW])
              << ", output " << VectorToString<index_t>(output_odim_ranges_[C_NHWC]);
#endif
  } else {
    LOG(ERROR) << "Unsupported partition dimension";
    MACE_NOT_IMPLEMENTED;
  }
}

PartitionResult FullyConnectedPartPlan::Make(
    const std::vector<index_t> input_shape,
    const std::vector<index_t> weight_shape) {
  weight_shape_ = weight_shape;

  if (ratio_ == kRatioGpuMinimum) {
    ratio_mode_ = RATIO_MODE_MIN_GPU;
    ratio_ = 0.5f;  // To enable CPU+GPU co-execution.
  }

  if (dim_ == DIM_INPUT_HEIGHT) {
    dim_ = DIM_OUTPUT_CHANNEL;
    ratio_ = 1.0f;
  }

  if (ratio_ >= kRatioCpuOnly && ratio_ <= kRatioGpuOnly) {
    PartitionResult ret;
    switch (dim_) {
      case DIM_OUTPUT_CHANNEL:
        //LOG(INFO) << "Build range";
        ret = BuildRange(input_shape, weight_shape);
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
  std::vector<index_t> output_shape;
  FullyConnectedPartPlanUtils::CalcOutputShape(input_shape, weight_shape, output_shape);
  BuildShape(input_shape, weight_shape, output_shape);

  //LOG(INFO) << "Build odim range";
  BuildOdimRanges();

  is_ready_ = true;
  
  return PartitionResult::PARTITION_SUCCESS;
}

void FullyConnectedPartPlan::Show() const {
  if (!is_ready_) {
    return;
  }

  const size_t buf_size = 128;
  char buffer[buf_size];

  VLOG(1) << "===== Part Plan =====";
  VLOG(1) << "Type: FullyConnected";
  VLOG(1) << "Dim: " << static_cast<int>(dim_);
  VLOG(1) << "Ratio: " << ratio_;
  VLOG(1) << "Weight shape: " << VectorToString<index_t>(weight_shape_);

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
  snprintf(buffer, buf_size, "GPU weight part shape: [%ld,%ld,%ld,%ld]",
      gpu_weight_part_shape_.at(0), gpu_weight_part_shape_.at(1),
      gpu_weight_part_shape_.at(2), gpu_weight_part_shape_.at(3));
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
  snprintf(buffer, buf_size, "CPU weight part shape: [%ld,%ld,%ld,%ld]",
      cpu_weight_part_shape_.at(0), cpu_weight_part_shape_.at(1),
      cpu_weight_part_shape_.at(2), cpu_weight_part_shape_.at(3));
  VLOG(1) << buffer;
  snprintf(buffer, buf_size, "CPU output part shape: [%ld,%ld,%ld,%ld]",
      cpu_output_part_shape_.at(0), cpu_output_part_shape_.at(1),
      cpu_output_part_shape_.at(2), cpu_output_part_shape_.at(3));
  VLOG(1) << buffer;
}

}  // namespace ops
}  // namespace mace
