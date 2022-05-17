
#include "test/codl_run/utils/fully_connected_util.h"

namespace mace {

#ifdef MACE_ENABLE_NEON
MaceStatus FullyConnectedCpuFloatKernel::Compute(
    const OpContext *context,
    const Tensor *input,
    const Tensor *weight,
    const Tensor *bias,
    Tensor *output) {
  MACE_CHECK(
      input->dim(1) == weight->dim(1) && input->dim(2) == weight->dim(2) &&
          input->dim(3) == weight->dim(3),
      "The shape of Input: ", MakeString(input->shape()),
      "The shape of Weight: ", MakeString(weight->shape()),
      " don't match.");
#if 0
  if (bias) {
    MACE_CHECK(weight->dim(0) == bias->dim(0),
               "The shape of Weight: ", MakeString(weight->shape()),
               " and shape of Bias: ", bias->dim(0),
               " don't match.");
  }
  std::vector<index_t> output_shape = {input->dim(0), weight->dim(0), 1, 1};
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));
#endif
  const index_t batch = output->dim(0);
  const index_t input_size = weight->dim(1) * weight->dim(2) * weight->dim(3);
  const index_t output_size = weight->dim(0);

  gemv_.Compute(context,
                weight,
                input,
                bias,
                batch,
                output_size,
                input_size,
                false,
                true,
                output);

  activation_delegator_.Compute(context, output, output);

  return MaceStatus::MACE_SUCCESS;
}

#ifdef MACE_ENABLE_QUANTIZE

MaceStatus FullyConnectedCpuUint8Kernel::Compute(
    const OpContext *context,
    const Tensor *input,
    const Tensor *weight,
    const Tensor *bias,
    Tensor *output) {

  MACE_CHECK(
      input->dim(1) == weight->dim(1) && input->dim(2) == weight->dim(2) &&
          input->dim(3) == weight->dim(3),
      "The shape of Input: ", MakeString(input->shape()),
      "The shape of Weight: ", MakeString(weight->shape()),
      " don't match.");
  if (bias) {
    MACE_CHECK(weight->dim(0) == bias->dim(0),
               "The shape of Weight: ", MakeString(weight->shape()),
               " and shape of Bias: ", bias->dim(0),
               " don't match.");
  }
  auto gemm_context = context->device()->cpu_runtime()->GetGemmlowpContext();
  MACE_CHECK_NOTNULL(gemm_context);

  std::vector<index_t> output_shape = {input->dim(0), 1, 1, weight->dim(0)};
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));
  const int batch = static_cast<int>(output->dim(0));
  const int input_size =
      static_cast<int>(weight->dim(1) * weight->dim(2) * weight->dim(3));
  const int output_size = static_cast<int>(weight->dim(0));
  gemv_.Compute(context,
                weight,
                input,
                bias,
                batch,
                output_size,
                input_size,
                false,
                true,
                output);
  return MaceStatus::MACE_SUCCESS;
}

#endif  // MACE_ENABLE_QUANTIZE
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_OPENCL
std::unique_ptr<ops::OpenCLFullyConnectedKernel>
    CreateOpenCLFullyConnectedKernel(const MemoryType mtype) {
  switch (mtype) {
    case MemoryType::GPU_IMAGE:
      return std::unique_ptr<ops::OpenCLFullyConnectedKernel>(
          new ops::opencl::image::FullyConnectedKernel());
    case MemoryType::GPU_BUFFER:
      return std::unique_ptr<ops::OpenCLFullyConnectedKernel>(
          new ops::opencl::buffer::FullyConnectedKernel());
    default:
      LOG(ERROR) << "Not support memory type";
      return nullptr;
  }
}
#endif

}  // namespace mace
