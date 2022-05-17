
#ifndef TEST_FUCHENG_GEMM_TEST_TASK_H_
#define TEST_FUCHENG_GEMM_TEST_TASK_H_

#include <memory>

#include "mace/core/op_context.h"
#include "mace/core/tensor_manage_util.h"
#include "mace/ops/arm/fp32/gemm.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/gemm.h"
#endif

#include "test/fucheng/device_util.h"
#include "test/fucheng/tensor_buffer_util.h"
#include "test/fucheng/test_task.h"

namespace mace {

class CodlGemmTestTask : public CodlTestTask {
public:
  CodlGemmTestTask() : rows_(0), depth_(0), cols_(0),
                       lhs_(nullptr), rhs_(nullptr), output_(nullptr),
                       output_buffer_(nullptr), context_(nullptr),
                       tensor_manage_util_(nullptr) {}

protected:
  int TensorRead(Tensor *tensor, const float *buf, const index_t size);

  int TensorWrite(const Tensor *tensor, float *buf, const index_t size);

  index_t rows_;
  index_t depth_;
  index_t cols_;
  Tensor *lhs_;
  Tensor *rhs_;
  Tensor *output_;
  float *output_buffer_;
  std::unique_ptr<OpContext> context_;
  std::unique_ptr<TensorManageUtil> tensor_manage_util_;
};

class CodlGemmCpuTestTask : public CodlGemmTestTask {
public:
  CodlGemmCpuTestTask() : gemm_(false) {}

  ~CodlGemmCpuTestTask() override {}

  int Prepare(TestParam *test_param) override;

  int Run(mace::DurationCollector<double> *dura_collector = nullptr) override;

  int Destroy() override;

private:
  ops::arm::fp32::Gemm gemm_;
};

#ifdef MACE_ENABLE_OPENCL
class CodlGemmGpuTestTask : public CodlGemmTestTask {
public:
  CodlGemmGpuTestTask() : memory_type_(MemoryType::GPU_BUFFER),
                          gemm_(nullptr) {}

  ~CodlGemmGpuTestTask() override {}

  int Prepare(TestParam *test_param) override;

  int Run(mace::DurationCollector<double> *dura_collector = nullptr) override;

  int Destroy() override;

private:
  ops::OpenCLGemmKernel *CreateOpenCLGemmKernel(MemoryType memory_type);

  int TensorTransformRead(const float *input,
                          const index_t size,
                          const OpenCLBufferType type,
                          Tensor *output);

  int TensorTransformWrite(const Tensor *input,
                           const index_t size,
                           const OpenCLBufferType type,
                           float *output);

  MemoryType memory_type_;
  std::shared_ptr<ops::OpenCLGemmKernel> gemm_;
};
#endif

} // namespace mace

#endif // TEST_FUCHENG_GEMM_TEST_TASK_H_
