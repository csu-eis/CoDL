
#include "mace/ops/opencl/image/gemm.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus GemmKernel::Compute(
    OpContext *context,
    const Tensor *lhs,
    const Tensor *rhs,
    const Tensor *bias,
    const ActivationType activation,
    const float relux_max_limit,
    const float leakyrelu_coefficient,
    Tensor *output) {
  MACE_CHECK(output->dim(1) == 1, "Output height must be 1");
  //const index_t batch = output->dim(0);
  const index_t m = output->dim(3);
  const index_t n = output->dim(2);
  //const index_t channels = output->dim(3);
  const index_t depth = rhs->dim(3);

  const index_t m_blocks = RoundUpDiv4(m);
  const index_t n_blocks = RoundUpDiv4(n);
  const uint32_t gws[2] = {
      static_cast<uint32_t>(n_blocks),
      static_cast<uint32_t>(m_blocks),
  };

  OpenCLRuntime *runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("matmul");
    built_options.emplace("-Dmatmul=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    switch (activation) {
      case NOOP:
        break;
      case RELU:
        built_options.emplace("-DUSE_RELU");
        break;
      case RELUX:
        built_options.emplace("-DUSE_RELUX");
        break;
      case TANH:
        built_options.emplace("-DUSE_TANH");
        break;
      case SIGMOID:
        built_options.emplace("-DUSE_SIGMOID");
        break;
      case LEAKYRELU:
        built_options.emplace("-DUSE_LEAKYRELU");
        break;
      default:
        LOG(FATAL) << "Unknown activation type: " << activation;
    }

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("matmul", kernel_name,
                                              built_options, &kernel_));
    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, rhs->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_2D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(lhs->opencl_image()));
    kernel_.setArg(idx++, *(rhs->opencl_image()));
    if (bias != nullptr) {
      kernel_.setArg(idx++, *(bias->opencl_buffer()));
    }
    kernel_.setArg(idx++, static_cast<int>(m));
    kernel_.setArg(idx++, static_cast<int>(n));
    kernel_.setArg(idx++, static_cast<int>(depth));
    kernel_.setArg(idx++, static_cast<int>(m_blocks));
    kernel_.setArg(idx++, static_cast<int>(RoundUpDiv4(depth)));
    kernel_.setArg(idx++, relux_max_limit);
    kernel_.setArg(idx++, leakyrelu_coefficient);
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = rhs->shape();
  }

  const std::vector<uint32_t> lws = {kwg_size_ / 64, 64, 0};
  std::string tuning_key = Concat("matmul_opencl_kernel",
                                  output->dim(1), output->dim(2));
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
