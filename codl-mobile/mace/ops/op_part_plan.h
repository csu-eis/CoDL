
#ifndef MACE_OPS_OP_PART_PLAN_H_
#define MACE_OPS_OP_PART_PLAN_H_

#include <vector>

#include "mace/core/types.h"
#include "mace/ops/common/odim_ranges.h"

namespace mace {
namespace ops {

constexpr float kRatioCpuOnly = 0;
constexpr float kRatioGpuOnly = 1;
constexpr float kRatioCpuGpuFull = 2;

enum PartitionResult {
  PARTITION_SUCCESS = 0,
  PARTITION_REDO = 1,
  PARTITION_FAILED = 2
};

enum RatioMode {
  RATIO_MODE_NONE = 0,
  RATIO_MODE_MIN_GPU = 1,
};

class PartPlanUtils {
 public:
  static index_t RoundUpDiv(const index_t v, const index_t f);

  static index_t RoundUp(const index_t v, const index_t f);

  static index_t MultipleAndAlign(const index_t len,
                                  const float ratio,
                                  const int align);

  static index_t MultipleAndRoundUp(const double len,
                                    const double ratio,
                                    const index_t base);
};

class OpPartPlan {
 public:
  explicit OpPartPlan(const PartitionDim dim,
                      const float ratio,
                      const DataFormat cpu_data_format = DataFormat::NCHW,
                      const DataType cpu_dtype = DataType::DT_FLOAT)
      : is_ready_(false),
        dim_(dim),
        ratio_(ratio),
        ratio_mode_(RATIO_MODE_NONE),
        cpu_in_out_data_format_(cpu_data_format),
        cpu_dtype_(cpu_dtype) {}
  
  virtual ~OpPartPlan() = default;

  virtual void Show() const = 0;

  inline float ratio() const {
    return ratio_;
  }
  
  inline PartitionDim dimension() const {
    return dim_;
  }

  inline index_t cpu_input_height_start() const {
    MACE_CHECK(dim_ == DIM_INPUT_HEIGHT);
    return cpu_input_range_[0];
  }
  
  inline index_t cpu_output_height_start() const {
    MACE_CHECK(dim_ == DIM_INPUT_HEIGHT);
    return cpu_output_range_[0];
  }

  inline std::vector<index_t> cpu_input_part_shape() const {
    return cpu_input_part_shape_;
  }

  inline void set_cpu_output_part_shape(const std::vector<index_t> &shape) {
    cpu_output_part_shape_ = shape;
  }
  
  inline std::vector<index_t> cpu_output_part_shape() const {
    return cpu_output_part_shape_;
  }
  
  inline std::vector<index_t> gpu_input_part_shape() const {
    return gpu_input_part_shape_;
  }

  inline void set_gpu_output_part_shape(const std::vector<index_t> &shape) {
    gpu_output_part_shape_ = shape;
  }

  inline std::vector<index_t> gpu_output_part_shape() const {
    return gpu_output_part_shape_;
  }

  inline const OdimRanges *input_odim_ranges() const {
    return &input_odim_ranges_;
  }
  
  inline const OdimRanges *output_odim_ranges() const {
    return &output_odim_ranges_;
  }

  inline bool IsGpuOnly() const {
    return (ratio_ == 1.0f);
  }
  
  inline bool IsCpuOnly() const {
    return (ratio_ == 0.0f);
  }
  
  inline bool IsCpuGpu() const {
    return ((ratio_ > 0.0f && ratio_ < 1.0f) || (ratio_ == 2.0f));
  }

  inline bool CheckPlanChanged(const PartitionDim dim,
                               const float ratio) const {
    return (dim_ != dim || ratio_ != ratio);
  }

  inline bool CheckIsReady() const {
    return is_ready_;
  }

  virtual void UpdateInputPartShape() {}

 protected:
  virtual void BuildOdimRanges() = 0;

 protected:
  bool is_ready_;
  PartitionDim dim_;
  float ratio_;
  RatioMode ratio_mode_;
  DataFormat cpu_in_out_data_format_;
  DataType cpu_dtype_;

  std::vector<index_t> cpu_input_range_;
  std::vector<index_t> gpu_input_range_;
  std::vector<index_t> cpu_output_range_;
  std::vector<index_t> gpu_output_range_;
  std::vector<index_t> cpu_input_part_shape_;
  std::vector<index_t> cpu_output_part_shape_;
  std::vector<index_t> gpu_input_part_shape_;
  std::vector<index_t> gpu_output_part_shape_;
  OdimRanges input_odim_ranges_;
  OdimRanges output_odim_ranges_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OP_PART_PLAN_H_
