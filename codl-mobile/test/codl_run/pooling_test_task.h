
#ifndef TEST_CODL_RUN_POOLING_TEST_TASK_H_
#define TEST_CODL_RUN_POOLING_TEST_TASK_H_

#include "mace/ops/pooling.h"

#include "test/codl_run/core/test_task.h"
#include "test/codl_run/utils/pooling_util.h"

namespace mace {

class CodlPoolingCpuGpuTestTask : public CodlOpCpuGpuTestTask {
 public:
  CodlPoolingCpuGpuTestTask() {
    type_ = CodlTestTaskType::POOLING_CPU_GPU_TEST_TASK;
  }

  ~CodlPoolingCpuGpuTestTask() override {}

  inline std::vector<int> kernels() const {
    return kernels_;
  }

  inline std::vector<int> strides() const {
    return strides_;
  }

  inline std::vector<int> dilations() const {
    return dilations_;
  }

  inline Padding padding_type() const {
    return padding_type_;
  }

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

  int PostProcess() override {
    return 0;
  }

  void UpdatePartTensors() override;

 private:
  PoolingType pooling_type_;
  std::vector<int> kernels_;
  std::vector<int> strides_;
  Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  RoundType round_type_;

  std::unique_ptr<PoolingKernel> cpu_kernel_;
  std::unique_ptr<ops::OpenCLPoolingKernel> opencl_kernel_;
};

}  // namespace mace

#endif  // TEST_CODL_RUN_POOLING_TEST_TASK_H_
