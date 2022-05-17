
#ifndef MACE_TEST_CODL_RUN_UTILS_DECONV_2D_UTIL_H_
#define MACE_TEST_CODL_RUN_UTILS_DECONV_2D_UTIL_H_

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/deconv_2d.h"
#include "mace/ops/arm/fp32/bias_add.h"
#include "mace/ops/arm/fp32/activation.h"
#endif

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer/deconv_2d.h"
#include "mace/ops/opencl/image/deconv_2d.h"
#endif

namespace mace {

ops::arm::fp32::Deconv2dBase *CreateNEONDeconv2dDelegator(
    const Tensor *filter,
    const std::vector<int> &strides,
    const std::vector<int> &paddings,
    const Padding padding_type,
    const FrameworkType model_type);

template<DeviceType D, class T>
class Deconv2dKernel;

#ifdef MACE_ENABLE_NEON
template<>
class Deconv2dKernel<DeviceType::CPU, float> {
public:
  Deconv2dKernel(const Tensor *filter,
                 const std::vector<int> &strides,
                 const std::vector<int> &paddings,
                 const Padding padding_type,
                 const FrameworkType model_type,
                 ops::ActivationType activation_type,
                 const float limit,
                 const float leakyrelu_coefficient)
      : activation_delegator_(activation_type, limit, leakyrelu_coefficient) {
    deconv2d_delegator_.reset(CreateNEONDeconv2dDelegator(filter,
                                                          strides,
                                                          paddings,
                                                          padding_type,
                                                          model_type));
  }

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input,
                     const Tensor *filter,
                     const Tensor *bias,
                     Tensor *output);

private:
  std::unique_ptr<ops::arm::fp32::Deconv2dBase> deconv2d_delegator_;
  ops::arm::fp32::BiasAdd bias_add_delegator_;
  ops::arm::fp32::Activation activation_delegator_;
};

typedef Deconv2dKernel<DeviceType::CPU, float> Deconv2dCpuFloatKernel;

#endif

#ifdef MACE_ENABLE_OPENCL
std::unique_ptr<ops::OpenCLDeconv2dKernel>
    CreateOpenCLDeconv2dKernel(const MemoryType mtype);
#endif

}  // namespace mace

#endif  // MACE_TEST_CODL_RUN_UTILS_DECONV_2D_UTIL_H_
