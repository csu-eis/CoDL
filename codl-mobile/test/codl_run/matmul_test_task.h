
#ifndef TEST_CODL_RUN_MATMUL_TEST_TASK_H_
#define TEST_CODL_RUN_MATMUL_TEST_TASK_H_

#include "test/codl_run/core/test_task.h"
#include "test/codl_run/utils/matmul_util.h"

namespace mace {

class CodlMatMulCpuGpuTestTask : public CodlOpCpuGpuTestTask {
 public:
  CodlMatMulCpuGpuTestTask() {
    type_ = CodlTestTaskType::MATMUL_CPU_GPU_TEST_TASK;
    rhs_gpu_ = nullptr;
    rhs_cpu_ = nullptr;
  }

  ~CodlMatMulCpuGpuTestTask() override {}

  int Prepare(TestParam *test_param) override;

  int EnqueueInputDataTransformKernel(StatsFuture *future = nullptr) override;

  int EnqueueMapKernel(StatsFuture *future,
                       StatsFuture *map_in_future = nullptr) override;

  int EnqueueGpuComputeKerenl(StatsFuture *future = nullptr) override;

  int EnqueueUnmapKernel(cl::UserEvent **event,
                         StatsFuture *unmap_in_future = nullptr,
                         StatsFuture *unmap_out_future = nullptr) override;

  int EnqueueOutputDataTransformKernel(StatsFuture *future = nullptr) override;

  int RunCpuComputeKernel() override;

  int PostProcess() override;

 private:
  bool transpose_a_;
  bool transpose_b_;

  Tensor *rhs_;
  Tensor *rhs_gpu_;
  Tensor *rhs_cpu_;

  std::unique_ptr<MatMulCpuKernelBase> cpu_kernel_;
  std::unique_ptr<ops::OpenCLMatMulKernel> opencl_kernel_;
};

}  // namespace mace

#endif  // TEST_CODL_RUN_MATMUL_TEST_TASK_H_
