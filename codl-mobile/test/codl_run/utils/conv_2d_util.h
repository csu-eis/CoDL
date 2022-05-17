
#ifndef MACE_TEST_CODL_RUN_UTILS_CONV_2D_UTIL_H_
#define MACE_TEST_CODL_RUN_UTILS_CONV_2D_UTIL_H_

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/conv_2d.h"
#include "mace/ops/arm/fp32/conv_2d_1x1.h"
#include "mace/ops/arm/fp32/conv_2d_3x3.h"
#include "mace/ops/arm/fp32/conv_2d_3x3_winograd.h"
#include "mace/ops/arm/fp32/conv_2d_5x5.h"
#include "mace/ops/arm/fp32/conv_2d_7x7.h"
#include "mace/ops/arm/fp32/conv_2d_9x9.h"
#include "mace/ops/arm/fp32/conv_2d_1xn.h"
#include "mace/ops/arm/fp32/conv_general.h"
#include "mace/ops/arm/fp32/bias_add.h"
#include "mace/ops/arm/fp32/activation.h"
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer/conv_2d.h"
#include "mace/ops/opencl/image/conv_2d.h"
#endif

namespace mace {

#ifdef MACE_ENABLE_NEON
ops::arm::fp32::Conv2dBase *CreateNEONConv2dDelegator(
    const Tensor *input,
    const Tensor *filter,
    const std::vector<int> &strides,
    const std::vector<int> &paddings,
    const Padding padding_type,
    const std::vector<int> &dilations);
#endif  // MACE_ENABLE_NEON

class Conv2dKernel {
 public:
  Conv2dKernel() {};
  virtual ~Conv2dKernel() = default;

  virtual MaceStatus Compute(const OpContext *context,
                             const Tensor *input,
                             const Tensor *filter,
                             const Tensor *bias,
                             Tensor *output) = 0;
};

#ifdef MACE_ENABLE_NEON
class Conv2dCpuFloatKernel : public Conv2dKernel {
 public:
  Conv2dCpuFloatKernel(const Tensor *input,
                       const Tensor *filter,
                       const std::vector<int> &strides,
                       const std::vector<int> &paddings,
                       const Padding padding_type,
                       const std::vector<int> &dilations,
                       ops::ActivationType activation_type,
                       const float limit,
                       const float leakyrelu_coefficient)
      : conv2d_delegator_(CreateNEONConv2dDelegator(input,
                                                    filter,
                                                    strides,
                                                    paddings,
                                                    padding_type,
                                                    dilations)),
        activation_delegator_(activation_type, limit, leakyrelu_coefficient) {}

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input,
                     const Tensor *filter,
                     const Tensor *bias,
                     Tensor *output) override;

 private:
  std::shared_ptr<ops::arm::fp32::Conv2dBase> conv2d_delegator_;
  ops::arm::fp32::BiasAdd bias_add_delegator_;
  ops::arm::fp32::Activation activation_delegator_;
};
#else  // MACE_ENABLE_NEON
class Conv2dCpuFloatKernel : public Conv2dKernel {
 public:
  Conv2dCpuFloatKernel(const std::vector<int> &strides,
                       const std::vector<int> &dilations,
                       const std::vector<int> &paddings,
                       const Padding padding_type)
      : ref_conv2d_delegator_(new ref::Conv2d<float>(strides,
                                                     dilations,
                                                     paddings,
                                                     padding_type)) {}

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input,
                     const Tensor *filter,
                     const Tensor *bias,
                     Tensor *output) override;

 private:
  std::shared_ptr<ref::Conv2d<float>> ref_conv2d_delegator_;
};
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_QUANTIZE

class Conv2dCpuUint8Kernel : public Conv2dKernel {
 public:
  Conv2dCpuUint8Kernel(const std::vector<int> &strides,
                       const std::vector<int> &paddings)
      : strides_(strides),
        paddings_(paddings) {}

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input,
                     const Tensor *filter,
                     const Tensor *bias,
                     Tensor *output) override;

 private:
  template<typename T>
  void Im2col(
      const OpContext *context,
      const T *in_data, const std::vector<index_t> &in_shape,
      const index_t filter_h, const index_t filter_w, const index_t stride_h,
      const index_t stride_w, const T zero_point, const int pad_height,
      const int pad_width, const std::vector<index_t> &out_shape,
      const index_t depth, T *im2col_data);

 private:
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int32_t> bias_;
};

#endif  // MACE_ENABLE_QUANTIZE

class Conv2dCpuTask {
 public:
  Conv2dCpuTask(std::shared_ptr<Conv2dCpuFloatKernel> kernel,
                Tensor *input,
                Tensor *filter,
                Tensor *bias,
                Tensor *output)
    : conv2d_kernel_(kernel),
      input_(input),
      filter_(filter),
      bias_(bias),
      output_(output) {}

  void set_in_transform_future(StatsFuture *future);

  void set_out_transform_user_event(cl::UserEvent *event);

  MaceStatus Run(OpContext *cpu_context,
                 OpContext *gpu_context);

 private:
  std::shared_ptr<Conv2dCpuFloatKernel> conv2d_kernel_;
  Tensor *input_;
  Tensor *filter_;
  Tensor *bias_;
  Tensor *output_;
  StatsFuture in_transform_future_;
  cl::UserEvent *out_transform_user_event_;
};

#ifdef MACE_ENABLE_OPENCL
std::unique_ptr<ops::OpenCLConv2dKernel>
  CreateOpenCLConv2dKernel(const MemoryType mtype);
#endif  // MACE_ENABLE_OPENCL

}  // namespace mace

#endif  // MACE_TEST_CODL_RUN_UTILS_CONV_2D_UTIL_H_
