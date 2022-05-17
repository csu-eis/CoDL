
#ifndef TEST_CODL_RUN_CORE_TEST_TASK_H_
#define TEST_CODL_RUN_CORE_TEST_TASK_H_

#include "mace/core/tensor.h"
#include "mace/core/tensor_manage_util.h"
#include "mace/core/op_context.h"
#include "mace/utils/op_delay_tool.h"

#include "test/codl_run/core/test_param.h"
#include "test/codl_run/utils/device_util.h"
#include "test/codl_run/utils/tensor_buffer_util.h"

namespace mace {

enum CodlTestTaskType {
  NONE_CPU_GPU_TEST_TASK = 0,
  CONV2D_CPU_GPU_TEST_TASK = 1,
  POOLING_CPU_GPU_TEST_TASK = 2,
  FC_CPU_GPU_TEST_TASK = 3,
  DECONV2D_CPU_GPU_TEST_TASK = 4,
  MATMUL_CPU_GPU_TEST_TASK = 5,
};

enum CodlTestTaskDestroyType {
  DESTROY_TYPE_SOFT,
  DESTROY_TYPE_HARD,
};

std::string CodlTestTaskTypeToString(const CodlTestTaskType type);

class CodlTestTask {
 public:
  CodlTestTask() {}
  virtual ~CodlTestTask() noexcept {}
  virtual int Prepare(TestParam *test_param) = 0;
  virtual int Run(mace::DurationCollector<double> *dura_collector = nullptr) = 0;
  virtual int Destroy() = 0;
};

double FutureToMillis(const StatsFuture *future);

void SetDeviceContext(TestDeviceContext *ctx);

void ClearDeviceContext();

TestDeviceContext *GetDeviceContext();

void ShowText(const std::string &text);

class CodlOpCpuGpuTestTask : public CodlTestTask {
 public:
  CodlOpCpuGpuTestTask() : CodlTestTask() {
    type_ = CodlTestTaskType::NONE_CPU_GPU_TEST_TASK;
    num_gpu_enqueued_kernels_ = 0;
    is_debug_on_ = true;

    input_gpu_  = nullptr;
    output_gpu_ = nullptr;

    input_cpu_  = nullptr;
    bias_cpu_   = nullptr;
    output_cpu_ = nullptr;
  }

  inline CodlTestTaskType type() const {
    return type_;
  }

  inline OpContext *op_context() {
    return gpu_context_.get();
  }

  inline Tensor *input() {
    return input_;
  }

  inline ops::OpPartPlan *part_plan() {
    return part_plan_.get();
  }

  index_t cpu_input_raw_size() const {
    const std::vector<index_t> shape = part_plan_->cpu_input_part_shape();
    const index_t size = std::accumulate(shape.begin(),
                                         shape.end(),
                                         1,
                                         std::multiplies<index_t>());
    return size * 4;
  }

  virtual index_t cpu_weight_raw_size() const {
    return 0;
  };

  index_t cpu_output_raw_size() const {
    const std::vector<index_t> shape = part_plan_->cpu_output_part_shape();
    const index_t size = std::accumulate(shape.begin(),
                                         shape.end(),
                                         1,
                                         std::multiplies<index_t>());
    return size * 4;
  }

  index_t gpu_input_raw_size() const {
    const std::vector<index_t> shape = input_->shape();
    const index_t size = std::accumulate(shape.begin(),
                                         shape.end(),
                                         1,
                                         std::multiplies<index_t>());
    return size * 4;
  }

  index_t gpu_output_raw_size() const {
    const std::vector<index_t> shape = output_->shape();
    const index_t size = std::accumulate(shape.begin(),
                                         shape.end(),
                                         1,
                                         std::multiplies<index_t>());
    return size * 4;
  }

  virtual index_t weight_raw_size() const {
    return 0;
  }

  int Run(mace::DurationCollector<double> *dura_collector = nullptr) override;

  virtual int PostProcess() = 0;

  int Destroy(const CodlTestTaskDestroyType type);

  int Destroy() override {
    return Destroy(DESTROY_TYPE_HARD);
  }

  virtual int EnqueueInputDataTransformKernel(StatsFuture *future = nullptr) = 0;

  virtual int EnqueueMapKernel(StatsFuture *map_out_future,
                               StatsFuture *map_in_future = nullptr) = 0;

  virtual int EnqueueGpuComputeKerenl(StatsFuture *future = nullptr) = 0;

  virtual int EnqueueUnmapKernel(cl::UserEvent **event,
                                 StatsFuture *unmap_in_future = nullptr,
                                 StatsFuture *unmap_out_future = nullptr) = 0;

  virtual int EnqueueOutputDataTransformKernel(StatsFuture *future = nullptr) = 0;

  virtual int RunCpuComputeKernel() = 0;

  virtual void UpdatePartTensors() {}

 protected:
  int RunCpu(mace::DurationCollector<double> *dura_collector = nullptr);
  
  int RunGpu(mace::DurationCollector<double> *dura_collector = nullptr);

  int RunCpuGpu(mace::DurationCollector<double> *dura_collector = nullptr);

 protected:
  CodlTestTaskType type_;
  // Number of gpu enqueued kernels.
  size_t num_gpu_enqueued_kernels_;
  // Debug option.
  bool is_debug_on_;

  // Runtime context.
  std::unique_ptr<TestDeviceContext> dev_context_;
  std::unique_ptr<OpContext> cpu_context_;
  std::unique_ptr<OpContext> gpu_context_;

  // Tensor manage utils.
  std::unique_ptr<TensorManageUtil> tensor_manage_util_;

  // Op parameters.
  bool do_data_transform_;
  bool do_compute_;
  ComputeUnitHint compute_unit_hint_;

  // Gpu tensors.
  Tensor *input_;
  Tensor *bias_;
  Tensor *output_;

  // Gpu partitial tensors.
  Tensor *input_gpu_;
  Tensor *output_gpu_;

  // Cpu partitial tensors.
  Tensor *input_cpu_;
  Tensor *output_cpu_;
  Tensor *bias_cpu_;

  // Op partition plan.
  std::unique_ptr<ops::OpPartPlan> part_plan_;
};

}  // namespace mace

#endif  // TEST_CODL_RUN_CORE_TEST_TASK_H_
