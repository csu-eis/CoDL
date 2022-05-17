
#ifndef MACE_OPS_MATMUL_PART_PLAN_H_
#define MACE_OPS_MATMUL_PART_PLAN_H_

#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/op_part_plan.h"

namespace mace {
namespace ops {

class MatMulPartPlanUtils {
 public:
  static void CalcOutputShape(
      const std::vector<index_t> &input_shape,
      const std::vector<index_t> &rhs_shape,
      const bool transpose_a,
      const bool transpose_b,
      std::vector<index_t> &out_shape);
};

class MatMulPartPlan : public OpPartPlan {
 public:
  explicit MatMulPartPlan(const PartitionDim dim,
                            const float ratio,
                            const DataFormat cpu_data_format)
      : OpPartPlan(dim, ratio, cpu_data_format) {}

  void Show() const override;

  PartitionResult Make(const std::vector<index_t> input_shape,
                       const std::vector<index_t> rhs_shape,
                       const bool transpose_a,
                       const bool transpose_b);

  std::vector<index_t> cpu_rhs_part_shape() const {
    return cpu_rhs_part_shape_;
  }

  std::vector<index_t> gpu_rhs_part_shape() const {
    return gpu_rhs_part_shape_;
  }

  const OdimRanges *rhs_odim_ranges() const {
    return &rhs_odim_ranges_;
  }

 private:
  PartitionResult BuildRange(const std::vector<index_t> &input_shape,
                             const std::vector<index_t> &rhs_shape);
  
  void BuildShape(const std::vector<index_t> &input_shape,
                  const std::vector<index_t> &rhs_shape,
                  const std::vector<index_t> &output_shape);

  void BuildOdimRanges() override;

  std::vector<index_t> lhs_shape_;
  std::vector<index_t> rhs_shape_;
  std::vector<index_t> cpu_rhs_part_shape_;
  std::vector<index_t> gpu_rhs_part_shape_;
  OdimRanges rhs_odim_ranges_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_MATMUL_PART_PLAN_H_
