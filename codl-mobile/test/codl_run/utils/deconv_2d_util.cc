
#include "mace/ops/arm/fp32/deconv_2d_2x2.h"
#include "mace/ops/arm/fp32/deconv_2d_3x3.h"
#include "mace/ops/arm/fp32/deconv_2d_4x4.h"
#include "mace/ops/arm/fp32/deconv_2d_general.h"

#include "test/codl_run/utils/deconv_2d_util.h"

namespace mace {

ops::arm::fp32::Deconv2dBase *CreateNEONDeconv2dDelegator(
    const Tensor *filter,
    const std::vector<int> &strides,
    const std::vector<int> &paddings,
    const Padding padding_type,
    const FrameworkType model_type) {
  const index_t kernel_h = filter->dim(2);
  const index_t kernel_w = filter->dim(3);

  bool use_neon_2x2_s1 = kernel_h == kernel_w && kernel_h == 2 &&
      strides[0] == strides[1] && strides[0] == 1;
  bool use_neon_2x2_s2 = kernel_h == kernel_w && kernel_h == 2 &&
      strides[0] == strides[1] && strides[0] == 2;

  bool use_neon_3x3_s1 = kernel_h == kernel_w && kernel_h == 3 &&
      strides[0] == strides[1] && strides[0] == 1;
  bool use_neon_3x3_s2 = kernel_h == kernel_w && kernel_h == 3 &&
      strides[0] == strides[1] && strides[0] == 2;

  bool use_neon_4x4_s1 = kernel_h == kernel_w && kernel_h == 4 &&
      strides[0] == strides[1] && strides[0] == 1;
  bool use_neon_4x4_s2 = kernel_h == kernel_w && kernel_h == 4 &&
      strides[0] == strides[1] && strides[0] == 2;

  const std::vector<int> output_paddings = {0, 0};

  if (use_neon_2x2_s1) {
    return new ops::arm::fp32::Deconv2dK2x2S1(
        paddings, output_paddings, padding_type, model_type);
  } else if (use_neon_2x2_s2) {
    return new ops::arm::fp32::Deconv2dK2x2S2(
        paddings, output_paddings, padding_type, model_type);
  } else if (use_neon_3x3_s1) {
    return new ops::arm::fp32::Deconv2dK3x3S1(
        paddings, output_paddings, padding_type, model_type);
  } else if (use_neon_3x3_s2) {
    return new ops::arm::fp32::Deconv2dK3x3S2(
        paddings, output_paddings, padding_type, model_type);
  } else if (use_neon_4x4_s1) {
    return new ops::arm::fp32::Deconv2dK4x4S1(
        paddings, output_paddings, padding_type, model_type);
  } else if (use_neon_4x4_s2) {
    return new ops::arm::fp32::Deconv2dK4x4S2(
        paddings, output_paddings, padding_type, model_type);
  } else {
    return new ops::arm::fp32::Deconv2dGeneral(strides,
                                               std::vector<int>{1, 1},
                                               paddings,
                                               output_paddings,
                                               padding_type,
                                               model_type);
  }
}

MaceStatus Deconv2dCpuFloatKernel::Compute(
    const OpContext *context,
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    Tensor *output) {
  //LOG(INFO) << "Run deconv2d delegator";
  deconv2d_delegator_->Compute(context,
                               input,
                               filter,
                               nullptr,
                               output);
  //LOG(INFO) << "Run bias add delegator";
  bias_add_delegator_.Compute(context, output, bias, output);
  //LOG(INFO) << "Run activation delegator";
  activation_delegator_.Compute(context, output, output);

  return MaceStatus::MACE_SUCCESS;
}

#ifdef MACE_ENABLE_OPENCL
std::unique_ptr<ops::OpenCLDeconv2dKernel>
    CreateOpenCLDeconv2dKernel(const MemoryType mtype) {
  switch (mtype) {
    case MemoryType::GPU_IMAGE:
      return std::unique_ptr<ops::OpenCLDeconv2dKernel>(
          new ops::opencl::image::Deconv2dKernel());
    case MemoryType::GPU_BUFFER:
      return std::unique_ptr<ops::OpenCLDeconv2dKernel>(
          new ops::opencl::buffer::Deconv2dKernel());
    default:
      LOG(ERROR) << "Not support memory type";
      return nullptr;
  }
}
#endif

}  // namespace mace
