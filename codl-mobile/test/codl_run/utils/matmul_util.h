
#ifndef MACE_TEST_CODL_RUN_UTILS_MATMUL_UTIL_H_
#define MACE_TEST_CODL_RUN_UTILS_MATMUL_UTIL_H_

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/q8/gemv.h"
#include "mace/ops/arm/fp32/gemm.h"
#include "mace/ops/arm/fp32/gemv.h"
#endif

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer/matmul.h"
#include "mace/ops/opencl/image/matmul.h"
#endif

namespace mace {

class MatMulUtils {
 public:
  static void Validate(const Tensor *A,
                       const Tensor *B,
                       const bool transpose_a,
                       const bool transpose_b);
};

class MatMulCpuKernelBase {
 public:
  MatMulCpuKernelBase() {}

  virtual ~MatMulCpuKernelBase() = default;

  virtual MaceStatus Compute(OpContext *context,
                             const Tensor *lhs,
                             const Tensor *rhs,
                             const Tensor *bias,
                             const bool transpose_a,
                             const bool transpose_b,
                             Tensor *C) = 0;
};

template<DeviceType D, class T>
class MatMulKernel;

#ifdef MACE_ENABLE_NEON

template<>
class MatMulKernel<DeviceType::CPU, float> : public MatMulCpuKernelBase {
 public:
  MatMulKernel() {}

  MaceStatus Compute(OpContext *context,
                     const Tensor *lhs,
                     const Tensor *rhs,
                     const Tensor *bias,
                     const bool transpose_a,
                     const bool transpose_b,
                     Tensor *C) override;

 private:
  ops::arm::fp32::Gemm gemm_;
  ops::arm::fp32::Gemv gemv_;
};

typedef MatMulKernel<DeviceType::CPU, float> MatMulCpuFloatKernel;

#ifdef MACE_ENABLE_QUANTIZE

template<>
class MatMulKernel<DeviceType::CPU, uint8_t> : public MatMulCpuKernelBase {
 public:
  MatMulKernel() {}

  MaceStatus Compute(OpContext *context,
                     const Tensor *lhs,
                     const Tensor *rhs,
                     const Tensor *bias,
                     const bool transpose_a,
                     const bool transpose_b,
                     Tensor *C) override;

 private:
  ops::arm::q8::Gemv<uint8_t> gemv_kernel_;
};

typedef MatMulKernel<DeviceType::CPU, uint8_t> MatMulCpuUint8Kernel;

#endif  // MACE_ENABLE_QUANTIZE
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_OPENCL
std::unique_ptr<ops::OpenCLMatMulKernel>
    CreateOpenCLMatMulKernel(const MemoryType mtype);
#endif

}  // namespace mace

#endif  // MACE_TEST_CODL_RUN_UTILS_MATMUL_UTIL_H_
