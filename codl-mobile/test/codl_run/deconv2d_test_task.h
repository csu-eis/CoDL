
#ifndef TEST_CODL_RUN_DECONV2D_TEST_TASK_H_
#define TEST_CODL_RUN_DECONV2D_TEST_TASK_H_

#include "mace/ops/deconv_2d.h"

#include "test/codl_run/core/test_task.h"
#include "test/codl_run/utils/deconv_2d_util.h"

namespace mace {

class CodlDeconv2dCpuGpuTestTask : public CodlOpCpuGpuTestTask {
 public:
  CodlDeconv2dCpuGpuTestTask() {
    type_ = CodlTestTaskType::DECONV2D_CPU_GPU_TEST_TASK;
    filter_gpu_ = nullptr;
    filter_cpu_ = nullptr;
  }

  ~CodlDeconv2dCpuGpuTestTask() override {}

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
  std::vector<int> strides_;
  Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  ops::ActivationType activation_;
  float relux_max_limit_;
  float leakyrelu_coefficient_;

  Tensor *filter_;

  Tensor *filter_gpu_;

  Tensor *filter_cpu_;

  std::unique_ptr<Deconv2dCpuFloatKernel> cpu_kernel_;
  std::unique_ptr<ops::OpenCLDeconv2dKernel> opencl_kernel_;
};

}  // namespace mace

#endif  // TEST_CODL_RUN_DECONV2D_TEST_TASK_H_
