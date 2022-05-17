
#ifndef MACE_TEST_CODL_RUN_UTILS_FULLY_CONNECTED_UTIL_H_
#define MACE_TEST_CODL_RUN_UTILS_FULLY_CONNECTED_UTIL_H_

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/gemv.h"
#include "mace/ops/arm/fp32/activation.h"
#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/arm/q8/gemv.h"
#endif  // MACE_ENABLE_QUANTIZE
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer/fully_connected.h"
#include "mace/ops/opencl/image/fully_connected.h"
#endif

namespace mace {

class FullyConnectedKernel {
 public:
  FullyConnectedKernel() {}
  virtual ~FullyConnectedKernel() = default;

  virtual MaceStatus Compute(const OpContext *context,
                             const Tensor *input,
                             const Tensor *weight,
                             const Tensor *bias,
                             Tensor *output) = 0;
};

#ifdef MACE_ENABLE_NEON
class FullyConnectedCpuFloatKernel : public FullyConnectedKernel {
 public:
  FullyConnectedCpuFloatKernel(
      ops::ActivationType activation_type,
      const float limit,
      const float leakyrelu_coefficient)
          : activation_delegator_(activation_type,
                                  limit,
                                  leakyrelu_coefficient) {}

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input,
                     const Tensor *weight,
                     const Tensor *bias,
                     Tensor *output) override;

 private:
  ops::arm::fp32::Gemv gemv_;
  ops::arm::fp32::Activation activation_delegator_;
};

#ifdef MACE_ENABLE_QUANTIZE
class FullyConnectedCpuUint8Kernel : public FullyConnectedKernel {
 public:
  FullyConnectedCpuUint8Kernel() {}

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input,
                     const Tensor *weight,
                     const Tensor *bias,
                     Tensor *output) override;

 private:
  ops::arm::q8::Gemv<uint8_t> gemv_;
};
#endif  // MACE_ENABLE_QUANTIZE
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_OPENCL
std::unique_ptr<ops::OpenCLFullyConnectedKernel>
    CreateOpenCLFullyConnectedKernel(const MemoryType mtype);
#endif

}  // namespace mace

#endif  // MACE_TEST_CODL_RUN_UTILS_FULLY_CONNECTED_UTIL_H_
