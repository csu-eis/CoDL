
#include "test/codl_run/utils/pooling_util.h"

namespace mace {

MaceStatus PoolingCpuFloatKernel::Compute(
    const OpContext *context,
    const Tensor *input,
    Tensor *output) {
  int pad_hw[2] = {0, 0};
  return delegator_.Compute(context,
                            input->data<float>(),
                            input->shape().data(),
                            output->shape().data(),
                            kernels_.data(),
                            strides_.data(),
                            dilations_.data(),
                            pad_hw,
                            output->mutable_data<float>());
}

#ifdef MACE_ENABLE_QUANTIZE
MaceStatus PoolingCpuUint8Kernel::Compute(
    const OpContext *context,
    const Tensor *input,
    Tensor *output) {
  int pad_hw[2] = {0, 0};
  return delegator_.Compute(context,
                            input->data<uint8_t>(),
                            input->shape().data(),
                            output->shape().data(),
                            kernels_.data(),
                            strides_.data(),
                            pad_hw,
                            output->mutable_data<uint8_t>());
}
#endif

#ifdef MACE_ENABLE_OPENCL
std::unique_ptr<ops::OpenCLPoolingKernel>
    CreateOpenCLPoolingKernel(const MemoryType mtype) {
  switch (mtype) {
    case MemoryType::GPU_IMAGE:
      return std::unique_ptr<ops::OpenCLPoolingKernel>(
          new ops::opencl::image::PoolingKernel());
    case MemoryType::GPU_BUFFER:
      return std::unique_ptr<ops::OpenCLPoolingKernel>(
          new ops::opencl::buffer::PoolingKernel());
    default:
      LOG(ERROR) << "Not support memory type";
      return nullptr;
  }
}
#endif

}  // namespace mace
