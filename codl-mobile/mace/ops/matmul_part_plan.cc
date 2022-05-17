
#include "mace/core/tensor.h"
#include "mace/ops/matmul_part_plan.h"

namespace mace {
namespace ops {

void MatMulPartPlanUtils::CalcOutputShape(
      const std::vector<index_t> &input_shape,
      const std::vector<index_t> &rhs_shape,
      const bool transpose_a,
      const bool transpose_b,
      std::vector<index_t> &out_shape) {
  const index_t rank_a = input_shape.size();
  const index_t rank_b = rhs_shape.size();
#if 0
  LOG(INFO) << "Rank: a " << rank_a << ", b " << rank_b;
#endif

  const index_t height
      = transpose_a ? input_shape[rank_a - 1] : input_shape[rank_a - 2];
  const index_t width
      = transpose_b ? rhs_shape[rank_b - 2] : rhs_shape[rank_b - 1];
#if 0
  LOG(INFO) << "Output: h " << height << ", w " << width;
#endif

  MACE_CHECK(rank_a == rank_b);
  const index_t rank = rank_a;
  if (rank == 2) {
    out_shape = {height, width};
  } else if (rank == 4) {
    const index_t batch = std::accumulate(input_shape.begin(),
                                          input_shape.end() - 2,
                                          1,
                                          std::multiplies<index_t>());
    out_shape = {1, batch, height, width};
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

PartitionResult MatMulPartPlan::BuildRange(
    const std::vector<index_t> &input_shape,
    const std::vector<index_t> &rhs_shape) {
  MACE_UNUSED(rhs_shape);

  const index_t rank = lhs_shape_.size();
  if (rank == 2) {
    const index_t M = input_shape[0];
    index_t m_gpu = PartPlanUtils::RoundUp(M * ratio_, 4);
    if (m_gpu >= M) {
      m_gpu = M;
      ratio_ = kRatioGpuOnly;
    }
    
    gpu_input_range_.push_back(0);
    gpu_input_range_.push_back(m_gpu - 1);
    gpu_input_range_.push_back(m_gpu);

    cpu_input_range_.push_back(m_gpu);
    cpu_input_range_.push_back(M - 1);
    cpu_input_range_.push_back(M - m_gpu);

    gpu_output_range_ = gpu_input_range_;
    cpu_output_range_ = cpu_input_range_;
  } else if (rank == 4) {
    const index_t batch = std::accumulate(input_shape.begin(),
                                          input_shape.end() - 2,
                                          1,
                                          std::multiplies<index_t>());
    const index_t batch_gpu = batch * ratio_;
    if (batch_gpu == 0) {
      ratio_ = kRatioCpuOnly;
    }

    gpu_input_range_.push_back(0);
    gpu_input_range_.push_back(batch_gpu - 1);
    gpu_input_range_.push_back(batch_gpu);

    cpu_input_range_.push_back(batch_gpu);
    cpu_input_range_.push_back(batch - 1);
    cpu_input_range_.push_back(batch - batch_gpu);

    gpu_output_range_ = gpu_input_range_;
    cpu_output_range_ = cpu_input_range_;
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  return PartitionResult::PARTITION_SUCCESS;
}

void MatMulPartPlan::BuildShape(const std::vector<index_t> &input_shape,
                                const std::vector<index_t> &rhs_shape,
                                const std::vector<index_t> &output_shape) {
  const index_t rank = lhs_shape_.size();
  if (rank == 2) {
    gpu_input_part_shape_ = input_shape;
    gpu_rhs_part_shape_ = rhs_shape;
    gpu_output_part_shape_ = output_shape;

    const int kMIdx = 0;

    gpu_input_part_shape_[kMIdx] = gpu_input_range_[2];
    gpu_output_part_shape_[kMIdx] = gpu_output_range_[2];

    cpu_input_part_shape_ = input_shape;
    cpu_rhs_part_shape_ = rhs_shape;
    cpu_output_part_shape_ = output_shape;

    cpu_input_part_shape_[kMIdx] = cpu_input_range_[2];
    cpu_output_part_shape_[kMIdx] = cpu_output_range_[2];
  } else if (rank == 4) {
    MACE_CHECK(input_shape[0] == 1);

    const int kBatchIdx = 1;
    
    gpu_input_part_shape_ = input_shape;
    gpu_rhs_part_shape_ = rhs_shape;
    gpu_output_part_shape_ = output_shape;

    gpu_input_part_shape_[kBatchIdx] = gpu_input_range_[2];
    gpu_rhs_part_shape_[kBatchIdx] = gpu_input_range_[2];
    gpu_output_part_shape_[kBatchIdx] = gpu_output_range_[2];

    cpu_input_part_shape_ = input_shape;
    cpu_rhs_part_shape_ = rhs_shape;
    cpu_output_part_shape_ = output_shape;

    cpu_input_part_shape_[kBatchIdx] = cpu_input_range_[2];
    cpu_rhs_part_shape_[kBatchIdx] = cpu_input_range_[2];
    cpu_output_part_shape_[kBatchIdx] = cpu_output_range_[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

void MatMulPartPlan::BuildOdimRanges() {
  const index_t rank = lhs_shape_.size();
#if 0
  LOG(INFO) << "BuildOdimRanges:"
            << " rank " << rank
            << ", rhs_shape " << VectorToString<index_t>(rhs_shape_);
#endif
  if (rank == 2) {
    input_odim_ranges_ = OdimRanges(rank);
    rhs_odim_ranges_ = OdimRanges(rank);
    output_odim_ranges_ = OdimRanges(rank);

    const int kMIdx = 0;

    input_odim_ranges_[kMIdx].push_back(0);
    input_odim_ranges_[kMIdx].push_back(cpu_input_part_shape_[kMIdx]);
    input_odim_ranges_[kMIdx].push_back(cpu_input_range_[0]);

    rhs_odim_ranges_[kMIdx].push_back(0);
    rhs_odim_ranges_[kMIdx].push_back(cpu_rhs_part_shape_[kMIdx]);
    rhs_odim_ranges_[kMIdx].push_back(0);

    output_odim_ranges_[kMIdx].push_back(cpu_output_range_[0]);
    output_odim_ranges_[kMIdx].push_back(cpu_output_range_[1] + 1);
    output_odim_ranges_[kMIdx].push_back(0 - cpu_output_range_[0]);
  } else if (rank == 4) {
    input_odim_ranges_ = OdimRanges(rank);
    rhs_odim_ranges_ = OdimRanges(rank);
    output_odim_ranges_ = OdimRanges(rank);

    // NOTE(fucheng): We use 0 to present batch idx for MatMul.
    const int kBatchIdx = 0;

    input_odim_ranges_[kBatchIdx].push_back(0);
    input_odim_ranges_[kBatchIdx].push_back(cpu_input_part_shape_[1]);
    input_odim_ranges_[kBatchIdx].push_back(cpu_input_range_[0]);

    rhs_odim_ranges_[kBatchIdx] = input_odim_ranges_[kBatchIdx];

    output_odim_ranges_[kBatchIdx].push_back(cpu_output_range_[0]);
    output_odim_ranges_[kBatchIdx].push_back(cpu_output_range_[1] + 1);
    output_odim_ranges_[kBatchIdx].push_back(0 - cpu_output_range_[0]);
  }
}

PartitionResult MatMulPartPlan::Make(
    const std::vector<index_t> input_shape,
    const std::vector<index_t> rhs_shape,
    const bool transpose_a,
    const bool transpose_b) {
  MACE_CHECK(input_shape.size() == rhs_shape.size(),
             ", only support same rank of lhs and rhs");
  const index_t rank = input_shape.size();
  if (rank == 2) {
    MACE_CHECK(dim_ == DIM_OUTPUT_CHANNEL,
               ", dimension at rank 2 must be output channel");
  } else if (rank == 4) {
    MACE_CHECK(dim_ == DIM_INPUT_HEIGHT,
               ", dimension at rank 4 must be input height");
  }

  lhs_shape_ = input_shape;
  rhs_shape_ = rhs_shape;

  std::vector<index_t> output_shape;
  MatMulPartPlanUtils::CalcOutputShape(
      input_shape, rhs_shape, transpose_a, transpose_b, output_shape);

  BuildRange(input_shape, rhs_shape);

  BuildShape(input_shape, rhs_shape, output_shape);

  BuildOdimRanges();

  is_ready_ = true;

  return PartitionResult::PARTITION_SUCCESS;
}

void MatMulPartPlan::Show() const {
  if (!is_ready_) {
    return;
  }

  const size_t buf_size = 128;
  char buffer[buf_size];

  LOG(INFO) << "===== Part Plan =====";
  LOG(INFO) << "Type: MatMul";
  LOG(INFO) << "Dim: " << static_cast<int>(dim_);
  LOG(INFO) << "Ratio: " << ratio_;
  LOG(INFO) << "LHS shape: " << VectorToString<index_t>(lhs_shape_);
  LOG(INFO) << "RHS shape: " << VectorToString<index_t>(rhs_shape_);

  const size_t rank = lhs_shape_.size();

  snprintf(buffer, buf_size, "GPU LHS shape range: [%ld,%ld,%ld]",
      gpu_input_range_[0], gpu_input_range_[1], gpu_input_range_[2]);
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "GPU output shape range: [%ld,%ld,%ld]",
      gpu_output_range_[0], gpu_output_range_[1], gpu_output_range_[2]);
  LOG(INFO) << buffer;
  
  if (rank == 2) {
    snprintf(buffer, buf_size, "GPU LHS part shape: [%ld,%ld]",
        gpu_input_part_shape_[0], gpu_input_part_shape_[1]);
    LOG(INFO) << buffer;
    snprintf(buffer, buf_size, "GPU RHS part shape: [%ld,%ld]",
        gpu_rhs_part_shape_[0], gpu_rhs_part_shape_[1]);
    LOG(INFO) << buffer;
    snprintf(buffer, buf_size, "GPU output part shape: [%ld,%ld]",
        gpu_output_part_shape_[0], gpu_output_part_shape_[1]);
    LOG(INFO) << buffer;
  } else if (rank == 4) {
    snprintf(buffer, buf_size, "GPU LHS part shape: [%ld,%ld,%ld,%ld]",
        gpu_input_part_shape_[0], gpu_input_part_shape_[1],
        gpu_input_part_shape_[2], gpu_input_part_shape_[3]);
    LOG(INFO) << buffer;
    snprintf(buffer, buf_size, "GPU RHS part shape: [%ld,%ld,%ld,%ld]",
        gpu_rhs_part_shape_[0], gpu_rhs_part_shape_[1],
        gpu_rhs_part_shape_[2], gpu_rhs_part_shape_[3]);
    LOG(INFO) << buffer;
    snprintf(buffer, buf_size, "GPU output part shape: [%ld,%ld,%ld,%ld]",
        gpu_output_part_shape_[0], gpu_output_part_shape_[1],
        gpu_output_part_shape_[2], gpu_output_part_shape_[3]);
    LOG(INFO) << buffer;
  }

  snprintf(buffer, buf_size, "CPU LHS shape range: [%ld,%ld,%ld]",
      cpu_input_range_[0], cpu_input_range_[1], cpu_input_range_[2]);
  LOG(INFO) << buffer;
  snprintf(buffer, buf_size, "CPU output shape range: [%ld,%ld,%ld]",
      cpu_output_range_[0], cpu_output_range_[1], cpu_output_range_[2]);
  LOG(INFO) << buffer;

  if (rank == 2) {
    snprintf(buffer, buf_size, "CPU LHS part shape: [%ld,%ld]",
        cpu_input_part_shape_[0], cpu_input_part_shape_[1]);
    LOG(INFO) << buffer;
    snprintf(buffer, buf_size, "CPU RHS part shape: [%ld,%ld]",
        cpu_rhs_part_shape_[0], cpu_rhs_part_shape_[1]);
    LOG(INFO) << buffer;
    snprintf(buffer, buf_size, "CPU output part shape: [%ld,%ld]",
        cpu_output_part_shape_[0], cpu_output_part_shape_[1]);
    LOG(INFO) << buffer;
  } else if (rank == 4) {
    snprintf(buffer, buf_size, "CPU LHS part shape: [%ld,%ld,%ld,%ld]",
        cpu_input_part_shape_[0], cpu_input_part_shape_[1],
        cpu_input_part_shape_[2], cpu_input_part_shape_[3]);
    LOG(INFO) << buffer;
    snprintf(buffer, buf_size, "CPU RHS part shape: [%ld,%ld,%ld,%ld]",
        cpu_rhs_part_shape_[0], cpu_rhs_part_shape_[1],
        cpu_rhs_part_shape_[2], cpu_rhs_part_shape_[3]);
    LOG(INFO) << buffer;
    snprintf(buffer, buf_size, "CPU output part shape: [%ld,%ld,%ld,%ld]",
        cpu_output_part_shape_[0], cpu_output_part_shape_[1],
        cpu_output_part_shape_[2], cpu_output_part_shape_[3]);
    LOG(INFO) << buffer;
  }
}

}  // namespace ops
}  // namespace mace
