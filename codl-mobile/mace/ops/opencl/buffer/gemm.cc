
#include "mace/ops/opencl/buffer/gemm.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {
namespace gemm {

MaceStatus Gemm(OpContext *context,
                cl::Kernel *kernel,
                const Tensor *lhs,
                const Tensor *rhs,
                const Tensor *bias,
                const int m,
                const int n,
                const int k,
                const ActivationType activation,
                const float relux_max_limit,
                const float leakyrelu_coefficient,
                const bool input_changed,
                Tensor *output,
                StatsFuture *future) {
  const index_t m_blocks = RoundUpDiv4(m);
  const index_t n_blocks = RoundUpDiv4(n);
  const index_t k_blocks = RoundUpDiv4(k);
  const uint32_t gws[2] = {
      static_cast<uint32_t>(n_blocks),
      static_cast<uint32_t>(m_blocks),
  };

  //LOG(INFO) << "m " << m << ", n " << n << ", k " << k;
  //LOG(INFO) << "m_blocks " << m_blocks
  //          << ", n_blocks " << n_blocks
  //          << ", k_blocks " << k_blocks;

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
    
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("matmul_buffer", kernel_name,
                                              built_options, kernel));
  }

  MACE_OUT_OF_RANGE_INIT(*kernel);
  if (input_changed) {
    uint32_t idx = 0;
    MACE_BUFF_OUT_OF_RANGE_SET_ARGS(*kernel, output->size());
    MACE_SET_2D_GWS_ARGS(*kernel, gws);
    kernel->setArg(idx++, *(lhs->opencl_buffer()));
    kernel->setArg(idx++, *(rhs->opencl_buffer()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_buffer()));
    }
    kernel->setArg(idx++, static_cast<int>(m));
    kernel->setArg(idx++, static_cast<int>(n));
    kernel->setArg(idx++, static_cast<int>(k));
    kernel->setArg(idx++, static_cast<int>(m_blocks));
    kernel->setArg(idx++, static_cast<int>(k_blocks));
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, leakyrelu_coefficient);
    kernel->setArg(idx++, *(output->opencl_buffer()));
  }

  const std::vector<uint32_t> lws = {16, 4, 0};
  std::string tuning_key = Concat("matmul_opencl_kernel", m, n);
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, *kernel, tuning_key,
                                           gws, lws, future));
  MACE_OUT_OF_RANGE_VALIDATION;

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace gemm

/**
// For CPU buffer.
MaceStatus GemmKernel::Compute(
    OpContext *context,
    const Tensor *lhs,  // OIHW
    const Tensor *rhs,  // NCHW
    Tensor *output  // NCHW
) {
  const index_t m = lhs->dim(0);
  const index_t n = rhs->dim(2) * rhs->dim(3);
  const index_t k = lhs->dim(1);
  return ComputeInternal(context, lhs, rhs, m, n, k, output);
} */

// For GPU buffer.
MaceStatus GemmKernel::Compute(
    OpContext *context,
    const Tensor *lhs,  // OIHW
    const Tensor *rhs,  // NHWC
    const Tensor *bias,
    const ActivationType activation,
    const float relux_max_limit,
    const float leakyrelu_coefficient,
    Tensor *output  /* NHWC */) {
  const index_t m = lhs->dim(0);
  const index_t n = rhs->dim(1) * rhs->dim(2);
  const index_t k = lhs->dim(1);
  //return ComputeInternal(context, lhs, rhs, bias, m, n, k,
  //                       activation, relux_max_limit, leakyrelu_coefficient,
  //                       output);
  const bool input_changed = !IsVecEqual(input_shape_, rhs->shape());
  return gemm::Gemm(context, &kernel_, lhs, rhs, bias, m, n, k,
                    activation, relux_max_limit, leakyrelu_coefficient,
                    input_changed, output, context->future());
}

MaceStatus GemmKernel::ComputeInternal(
    OpContext *context,
    const Tensor *lhs,
    const Tensor *rhs,
    const Tensor *bias,
    const index_t m,
    const index_t n,
    const index_t k,
    const ActivationType activation,
    const float relux_max_limit,
    const float leakyrelu_coefficient,
    Tensor *output) {
  const index_t m_blocks = RoundUpDiv4(m);
  const index_t n_blocks = RoundUpDiv4(n);
  const index_t k_blocks = RoundUpDiv4(k);
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
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("matmul_gpu_buffer");
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
    
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("matmul_buffer", kernel_name,
                                              built_options, &kernel_));
    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, rhs->shape())) {
    uint32_t idx = 0;
    MACE_BUFF_OUT_OF_RANGE_SET_ARGS(kernel_, output->size());
    MACE_SET_2D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(lhs->opencl_buffer()));
    kernel_.setArg(idx++, *(rhs->opencl_buffer()));
    if (bias != nullptr) {
      kernel_.setArg(idx++, *(bias->opencl_buffer()));
    }
    kernel_.setArg(idx++, static_cast<int>(m));
    kernel_.setArg(idx++, static_cast<int>(n));
    kernel_.setArg(idx++, static_cast<int>(k));
    kernel_.setArg(idx++, static_cast<int>(m_blocks));
    kernel_.setArg(idx++, static_cast<int>(k_blocks));
    kernel_.setArg(idx++, relux_max_limit);
    kernel_.setArg(idx++, leakyrelu_coefficient);
    kernel_.setArg(idx++, *(output->opencl_buffer()));

    input_shape_ = rhs->shape();
  }

  const std::vector<uint32_t> lws = {kwg_size_ / 64, 64, 0};
  std::string tuning_key = Concat("matmul_opencl_kernel", m, n);
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace
