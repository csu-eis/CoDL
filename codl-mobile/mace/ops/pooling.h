
#ifndef MACE_OPS_POOLING_H_
#define MACE_OPS_POOLING_H_

#include "mace/core/operator.h"
#include "mace/ops/conv_pool_2d_base.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/common/pooling_type.h"

namespace mace {
namespace ops {

template<DeviceType D, class T>
class PoolingDelegator;

template<>
class PoolingDelegator<DeviceType::CPU, float> {
 public:
  explicit PoolingDelegator(const PoolingType pooling_type)
      : pooling_type_(pooling_type) {}

  MaceStatus Compute(const OpContext *context,
                     const float *input,
                     const index_t *in_shape,
                     const index_t *out_shape,
                     const int *filter_hw,
                     const int *stride_hw,
                     const int *dilation_hw,
                     const int *pad_hw,
                     float *output);
 private:
  void MaxPooling(const OpContext *context,
                  const float *input,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *dilation_hw,
                  const int *pad_hw,
                  float *output);

  void AvgPooling(const OpContext *context,
                  const float *input,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *dilation_hw,
                  const int *pad_hw,
                  float *output);

 private:
  PoolingType pooling_type_;
};

template<>
class PoolingDelegator<DeviceType::CPU, uint8_t> {
 public:
  explicit PoolingDelegator(const PoolingType pooling_type)
      : pooling_type_(pooling_type) {}

  MaceStatus Compute(const OpContext *context,
                     const uint8_t *input,
                     const index_t *in_shape,
                     const index_t *out_shape,
                     const int *filter_hw,
                     const int *stride_hw,
                     const int *pad_hw,
                     uint8_t *output);
 private:
  void MaxPooling(const OpContext *context,
                  const uint8_t *input,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *pad_hw,
                  uint8_t *output);

  void AvgPooling(const OpContext *context,
                  const uint8_t *input,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *pad_hw,
                  uint8_t *output);

 private:
  PoolingType pooling_type_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_POOLING_H_
