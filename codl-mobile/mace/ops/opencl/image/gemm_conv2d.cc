
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/op_context.h"
#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/opencl/helper.h"
#include "mace/utils/memory.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

extern MaceStatus GemmConv2dK1x1S1(OpContext *context,
                                   cl::Kernel *kernel,
                                   const Tensor *input,
                                   const Tensor *filter,
                                   const Tensor *bias,
                                   const ActivationType activation,
                                   const float relux_max_limit,
                                   const float leakyrelu_coefficient,
                                   std::vector<index_t> *prev_input_shape,
                                   Tensor *output,
                                   uint32_t *kwg_size) {
  // TODO(fucheng): Support bias add and activation in matmul.
  MACE_UNUSED(bias);
  MACE_UNUSED(activation);
  MACE_UNUSED(relux_max_limit);
  MACE_UNUSED(leakyrelu_coefficient);

  //const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  //const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  MACE_CHECK(input->dim(3) == filter->dim(1));
  
  const index_t height_blocks = RoundUpDiv4(height);
  const index_t width_blocks = RoundUpDiv4(width);
  const uint32_t gws[2] = {
      static_cast<uint32_t>(width_blocks),
      static_cast<uint32_t>(height_blocks),
  };

  OpenCLRuntime *runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("matmul");
    built_options.emplace("-Dmatmul=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("matmul", kernel_name,
                                              built_options, kernel));

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }
  
  MACE_OUT_OF_RANGE_INIT(*kernel);
  if (!IsVecEqual(*prev_input_shape, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(*kernel);
    MACE_SET_2D_GWS_ARGS(*kernel, gws);
    kernel->setArg(idx++, *(filter->opencl_image()));
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(output->opencl_image()));
    kernel->setArg(idx++, static_cast<int>(height));
    kernel->setArg(idx++, static_cast<int>(width));
    kernel->setArg(idx++, static_cast<int>(input_channels));
    kernel->setArg(idx++, static_cast<int>(height_blocks));
    kernel->setArg(idx++, static_cast<int>(RoundUpDiv4(input_channels)));

    *prev_input_shape = input->shape();
  }

  const std::vector<uint32_t> lws = {*kwg_size / 64, 64, 0};
  std::string tuning_key = Concat("matmul_opencl_kernel",
                                  output->dim(1), output->dim(2));
  StatsFuture future;
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, *kernel, tuning_key,
                                           gws, lws, &future));

  MACE_OUT_OF_RANGE_VALIDATION;

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
