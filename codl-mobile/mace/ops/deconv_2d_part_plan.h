
#ifndef MACE_OPS_DECONV_2D_PART_PLAN_H_
#define MACE_OPS_DECONV_2D_PART_PLAN_H_

#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/op_part_plan.h"

namespace mace {
namespace ops {

class Deconv2dPartPlanUtils {
 public:
  static void CalcOutputShape(
      const Tensor *output_shape_tensor,
      const std::vector<index_t> &input_shape,
      const std::vector<index_t> &filter_shape,
      const std::vector<int> &strides,
      const Padding padding_type,
      const std::vector<int> &paddings,
      const FrameworkType framework_type,
      std::vector<index_t> &out_shape);

  static void CalcBiasShape(
      const std::vector<index_t> &out_shape,
      std::vector<index_t> &bias_shape);
};

class Deconv2dPartPlan : public OpPartPlan {
 public:
  explicit Deconv2dPartPlan(const PartitionDim dim,
                            const float ratio,
                            const DataFormat cpu_data_format)
      : OpPartPlan(dim, ratio, cpu_data_format) {}

  void Show() const override;

  PartitionResult Make(const Tensor *output_shape_tensor,
                       const std::vector<index_t> input_shape,
                       const std::vector<index_t> filter_shape,
                       const std::vector<int> strides,
                       const Padding padding_type,
                       const std::vector<int> paddings,
                       const FrameworkType framework_type);

  std::vector<index_t> cpu_filter_part_shape() const {
    return cpu_filter_part_shape_;
  }

  std::vector<index_t> gpu_filter_part_shape() const {
    return gpu_filter_part_shape_;
  }

 private:
  PartitionResult BuildRange(const std::vector<index_t> &input_shape,
                             const std::vector<index_t> &filter_shape);
  
  void BuildShape(const std::vector<index_t> &input_shape,
                  const std::vector<index_t> &filter_shape,
                  const std::vector<index_t> &output_shape);

  void BuildOdimRanges() override;

  std::vector<index_t> filter_shape_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<index_t> cpu_filter_part_shape_;
  std::vector<index_t> gpu_filter_part_shape_;
  OdimRanges filter_odim_ranges_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DECONV_2D_PART_PLAN_H_
