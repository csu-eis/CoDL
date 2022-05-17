
#ifndef MACE_TEST_CODL_RUN_UTILS_POOLING_UTIL_H_
#define MACE_TEST_CODL_RUN_UTILS_POOLING_UTIL_H_

#include "mace/ops/pooling.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer/pooling.h"
#include "mace/ops/opencl/image/pooling.h"
#endif

namespace mace {

class PoolingKernel {
 public:
  PoolingKernel() {}
  virtual ~PoolingKernel() = default;

  virtual MaceStatus Compute(const OpContext *context,
                             const Tensor *input,
                             Tensor *output) = 0;
};

class PoolingCpuFloatKernel : public PoolingKernel {
 public:
  PoolingCpuFloatKernel(const std::vector<int> kernels,
                        const std::vector<int> strides,
                        const std::vector<int> dilations,
                        const PoolingType pooling_type)
      : kernels_(kernels),
        strides_(strides),
        dilations_(dilations),
        delegator_(pooling_type) {}

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input,
                     Tensor *output) override;

 private:
  std::vector<int> kernels_;
  std::vector<int> strides_;
  std::vector<int> dilations_;
  ops::PoolingDelegator<DeviceType::CPU, float> delegator_;
};

#ifdef MACE_ENABLE_QUANTIZE
class PoolingCpuUint8Kernel : public PoolingKernel {
 public:
  PoolingCpuUint8Kernel(const std::vector<int> kernels,
                        const std::vector<int> strides,
                        const PoolingType pooling_type)
      : kernels_(kernels),
        strides_(strides),
        delegator_(pooling_type) {}

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input,
                     Tensor *output) override;

 private:
  std::vector<int> kernels_;
  std::vector<int> strides_;
  ops::PoolingDelegator<DeviceType::CPU, uint8_t> delegator_;
};
#endif

#ifdef MACE_ENABLE_OPENCL
std::unique_ptr<ops::OpenCLPoolingKernel>
    CreateOpenCLPoolingKernel(const MemoryType mtype);
#endif

}  // namespace mace

#endif  // MACE_TEST_CODL_RUN_UTILS_POOLING_UTIL_H_
