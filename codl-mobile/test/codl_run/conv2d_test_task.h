
#ifndef TEST_CODL_RUN_CONV2D_TEST_TASK_H_
#define TEST_CODL_RUN_CONV2D_TEST_TASK_H_

#include <vector>

#include "mace/ops/arm/fp32/gemm.h"
#include "mace/ops/conv_2d_part_plan.h"

#include "test/codl_run/utils/conv_2d_util.h"
#include "test/codl_run/core/test_task.h"

namespace mace {

class CodlConv2dCpuGpuTestTask : public CodlOpCpuGpuTestTask {
 public:
  CodlConv2dCpuGpuTestTask() {
    type_ = CodlTestTaskType::CONV2D_CPU_GPU_TEST_TASK;
    filter_gpu_ = nullptr;
    filter_cpu_ = nullptr;
  }

  ~CodlConv2dCpuGpuTestTask() override {}

  inline Tensor *filter() {
    return filter_;
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

  index_t cpu_weight_raw_size() const override {
    if (filter_cpu_ != nullptr) {
      return filter_cpu_->raw_size();
    }
    return 0;
  }

  index_t weight_raw_size() const override {
    return filter_->raw_size();
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

#if 0
 private:
  int RunCpu(mace::DurationCollector<double> *dura_collector = nullptr) override;
  
  int RunGpu(mace::DurationCollector<double> *dura_collector = nullptr) override;

  int RunCpuGpu(mace::DurationCollector<double> *dura_collector = nullptr) override;
#endif

 private:
  // Convolution parameters.
  std::vector<int> strides_;
  Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  ops::ActivationType activation_;
  float relux_max_limit_;
  float leakyrelu_coefficient_;
  int wino_block_size_;

  // Gpu tensors.
  Tensor *filter_;

  // Gpu partitial tensors.
  Tensor *filter_gpu_;

  // Cpu partitial tensors.
  Tensor *filter_cpu_;

  // Conv2d partition plan.
  //std::unique_ptr<ops::Conv2dPartPlan> part_plan_;

  // Cpu and gpu Compute kernel.
  std::unique_ptr<Conv2dKernel> conv2d_cpu_kernel_;
  std::unique_ptr<ops::OpenCLConv2dKernel> opencl_kernel_;
};

}  // namespace mace

#endif  // TEST_CODL_RUN_CONV2D_TEST_TASK_H_
