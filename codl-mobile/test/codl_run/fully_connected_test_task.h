
#ifndef TEST_CODL_RUN_FULLY_CONNECTED_TEST_TASK_H_
#define TEST_CODL_RUN_FULLY_CONNECTED_TEST_TASK_H_

#include "mace/ops/fully_connected.h"

#include "test/codl_run/core/test_task.h"
#include "test/codl_run/utils/fully_connected_util.h"

namespace mace {

class CodlFullyConnectedCpuGpuTestTask : public CodlOpCpuGpuTestTask {
 public:
  CodlFullyConnectedCpuGpuTestTask() {
    type_ = CodlTestTaskType::FC_CPU_GPU_TEST_TASK;
    weight_gpu_ = nullptr;
    weight_cpu_ = nullptr;
  }

  ~CodlFullyConnectedCpuGpuTestTask() override {}

  inline Tensor *weight() {
    return weight_;
  }

  index_t cpu_weight_raw_size() const override {
    if (weight_cpu_ != nullptr) {
      return weight_cpu_->raw_size();
    }
    return 0;
  }

  index_t weight_raw_size() const override {
    return weight_->raw_size();
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

 private:
  ops::ActivationType activation_;
  float relux_max_limit_;
  float leakyrelu_coefficient_;

  Tensor *weight_;

  Tensor *weight_gpu_;

  Tensor *weight_cpu_;

  std::unique_ptr<FullyConnectedKernel> cpu_kernel_;
  std::unique_ptr<ops::OpenCLFullyConnectedKernel> opencl_kernel_;
};

}  // namespace mace

#endif  // TEST_CODL_RUN_FULLY_CONNECTED_TEST_TASK_H_
