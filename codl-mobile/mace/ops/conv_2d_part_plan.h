
#ifndef MACE_OPS_CONV_2D_PART_PLAN_H_
#define MACE_OPS_CONV_2D_PART_PLAN_H_

#include <stdbool.h>
#include <stdint.h>

#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/op_part_plan.h"

namespace mace {
namespace ops {

enum Conv2dDimension {
  CONV2D_DIM_INPUT_HEIGHT   = 0,
  CONV2D_DIM_INPUT_WIDTH    = 1,
  CONV2D_DIM_OUTPUT_HEIGHT  = 2,
  CONV2D_DIM_OUTPUT_WIDTH   = 3,
  CONV2D_DIM_INPUT_CHANNEL  = 4,
  CONV2D_DIM_OUTPUT_CHANNEL = 5
};

class ConvPool2dPartPlanUtils {
 public:
  static bool CheckInputHeight(const index_t height,
                               const index_t filter_height,
                               const int stride);

  inline static bool IsWinograd(const std::vector<index_t> &filter_shape,
                                const std::vector<int> &strides,
                                const std::vector<int> &dilations);

  inline static index_t GetWinogradOutputTileSize(
      const std::vector<index_t> &input_shape);

  inline static bool IsAlignToWinogradTileSize(
      const std::vector<index_t> &input_shape,
      const index_t length);

  inline static index_t AlignToWinogradTileSize(
      const std::vector<index_t> &input_shape,
      const index_t length);

  static void CalcConv2dInputShape(
      const std::vector<index_t> output_shape,
      const std::vector<index_t> filter_shape,
      const std::vector<int> strides,
      const std::vector<int> paddings,
      const Padding padding_type,
      std::vector<index_t> &input_shape);

  static void CalcConv2dOutputShape(
      const std::vector<index_t> input_shape,
      const std::vector<index_t> filter_shape,
      const std::vector<int> strides,
      const std::vector<int> paddings,
      const Padding padding_type,
      std::vector<index_t> &output_shape);

  static void CalcConv2dBiasShape(
      const std::vector<index_t> &output_shape,
      std::vector<index_t> &bias_shape);

  static void CalcPaddings(
      const std::vector<index_t> input_shape,
      const std::vector<index_t> filter_shape,
      const std::vector<int> strides,
      const std::vector<index_t> output_shape,
      std::vector<int> &paddings);
};

class ConvPool2dPartPlan : public OpPartPlan {
 public:
  explicit ConvPool2dPartPlan(const PartitionDim dim,
                              const float ratio,
                              const DataFormat cpu_data_format = DataFormat::NCHW,
                              const DataType cpu_dtype = DataType::DT_FLOAT)
      : OpPartPlan(dim, ratio, cpu_data_format, cpu_dtype) {}

  void Show() const override;
  
  PartitionResult Make(const std::vector<index_t> input_shape,
                       const std::vector<index_t> filter_shape,
                       const std::vector<int> strides,
                       const std::vector<int> dilations,
                       const Padding padding_type,
                       const std::vector<int> paddings);

  std::vector<int> paddings() const {
    return paddings_;
  }

  std::vector<index_t> cpu_filter_part_shape() const {
    return cpu_filter_part_shape_;
  }
  
  std::vector<index_t> gpu_filter_part_shape() const {
    return gpu_filter_part_shape_;
  }

  void UpdateInputPartShape() override;
  
 private:
  template<Conv2dDimension D>
  PartitionResult BuildRange(const std::vector<index_t> &input_shape,
                             const std::vector<index_t> &filter_shape,
                             const std::vector<int>     &strides,
                             const std::vector<int>     &dilations,
                             const std::vector<index_t> &output_shape);
  
  void BuildShape(const std::vector<index_t> &input_shape,
                  const std::vector<index_t> &filter_shape,
                  const std::vector<index_t> &output_shape);
                  
  void BuildOdimRanges() override;

  std::vector<index_t> filter_shape_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  Padding padding_type_;
  std::vector<index_t> cpu_filter_part_shape_;
  std::vector<index_t> gpu_filter_part_shape_;
  OdimRanges filter_odim_ranges_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CONV_2D_PART_PLAN_H_
