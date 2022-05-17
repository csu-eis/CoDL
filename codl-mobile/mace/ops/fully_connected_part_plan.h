
#ifndef MACE_OPS_FULLY_CONNECTED_PART_PLAN_H_
#define MACE_OPS_FULLY_CONNECTED_PART_PLAN_H_

#include "mace/ops/op_part_plan.h"

namespace mace {
namespace ops {

class FullyConnectedPartPlanUtils {
 public:
  static void CalcOutputShape(const std::vector<index_t> &input_shape,
                              const std::vector<index_t> &weight_shape,
                              std::vector<index_t> &output_shape) {
    MACE_CHECK(input_shape.size() == 4);
    MACE_CHECK(weight_shape.size() == 4);
    output_shape = {input_shape[0], 1, 1, weight_shape[0]};
  }

  static void CalcBiasShape(const std::vector<index_t> &output_shape,
                            std::vector<index_t> &bias_shape) {
    bias_shape = {output_shape[C_NHWC]};
  }
};

class FullyConnectedPartPlan : public OpPartPlan {
 public:
  explicit FullyConnectedPartPlan(const PartitionDim dim,
                                  const float ratio,
                                  const DataFormat cpu_data_format)
      : OpPartPlan(dim, ratio, cpu_data_format) {}

  void Show() const override;

  PartitionResult Make(const std::vector<index_t> input_shape,
                       const std::vector<index_t> weight_shape);

  std::vector<index_t> cpu_weight_part_shape() const {
    return cpu_weight_part_shape_;
  }

  std::vector<index_t> gpu_weight_part_shape() const {
    return gpu_weight_part_shape_;
  }

 private:
  PartitionResult BuildRange(const std::vector<index_t> &input_shape,
                             const std::vector<index_t> &weight_shape);
  
  void BuildShape(const std::vector<index_t> &input_shape,
                  const std::vector<index_t> &weight_shape,
                  const std::vector<index_t> &output_shape);

  void BuildOdimRanges() override;

  std::vector<index_t> weight_shape_;
  std::vector<index_t> cpu_weight_part_shape_;
  std::vector<index_t> gpu_weight_part_shape_;
  OdimRanges weight_odim_ranges_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_FULLY_CONNECTED_PART_PLAN_H_
