
#include "mace/ops/fully_connected.h"

namespace mace {
namespace ops {

#if 0
static void ShowInfo(const std::string info) {
  LOG(INFO) << info;
}
#endif

#if 0
static void ShowLatency(const std::string name, const double latency) {
  LOG(INFO) << name << ": " << latency << " ms";
}
#endif

MaceStatus FullyConnectedGpuFloatOp::RunCpuGpuImage(
    OpContext *context) {
  if (context->part_run_config() == nullptr) {
    return RunGpu(context);
  }

  // Prepare tensors.
  //ShowInfo("Prepare tensors");
  const Tensor *input = this->Input(INPUT);
  const Tensor *weight = this->Input(WEIGHT);
  const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
  Tensor *output = this->Output(OUTPUT);

  // Make sure output tensor keeps the shape.
  kernel_->ResizeOutputTensor(input, weight, output);

#if 0
  return RunGpu(context);
#endif

  // Prepare partition parameters.
  //ShowInfo("Prepare partition plan");
  PartitionDim partition_dim = context->part_run_config()->dim_type();
  float partition_ratio = context->part_run_config()->ratio();
  if (partition_ratio == kRatioFromConfigFile ||
      partition_ratio == kRatioFromPredictor) {
    partition_dim = static_cast<PartitionDim>(context->partition_configer()->dim());
    partition_ratio = context->partition_configer()->ratio();
  }
  
#if 0
  // Make partition plan.
  //ShowInfo("Make partition plan");
  if (part_plan_) {
    MACE_CHECK(context->op_runtime_mode() == OpRuntimeMode::RUNNING);
    if (part_plan_->CheckPlanChanged(partition_dim, partition_ratio)) {
      const DataFormat df = DataFormat::NCHW;
      part_plan_.reset(new FullyConnectedPartPlan(partition_dim, partition_ratio, df));
      PartitionResult partition_result;
      do {
        partition_result = part_plan_->Make(input->shape(), weight->shape());
      } while (partition_result == PartitionResult::PARTITION_REDO);
      part_plan_->Show();
    }
  } else {
    const DataFormat df = DataFormat::NCHW;
    part_plan_.reset(new FullyConnectedPartPlan(partition_dim, partition_ratio, df));
    PartitionResult partition_result;
    do {
      partition_result = part_plan_->Make(input->shape(), weight->shape());
    } while (partition_result == PartitionResult::PARTITION_REDO);
    part_plan_->Show();
  }
#else
  MakePartitionPlan(context, input, weight, partition_dim, partition_ratio);
#endif

  // CPU+GPU co-execution.
  //ShowInfo("CPU+GPU co-execution");
  if (part_plan_->CheckIsReady()) {
    if (!part_plan_->IsGpuOnly()) {
      // Prepare GPU tensors.
      //ShowInfo("Prepare GPU tensors");
      TensorManageUtil *tensor_manage_util = context->workspace()->tensor_manage_util();

      Tensor *gpu_input = nullptr, *gpu_weight = nullptr, *gpu_output = nullptr;
      if (!part_plan_->IsCpuOnly()) {
        gpu_input = tensor_manage_util->CreateOrReshapePartTensor(
            input, partition_dim,
            0, part_plan_->gpu_input_part_shape()[C_NHWC],
            /* reshape */ true, &temp_tensor_indices_[GPU_INPUT]);
        gpu_weight = tensor_manage_util->CreateOrReshapePartTensor(
            weight, partition_dim,
            0, part_plan_->gpu_weight_part_shape()[O_OIHW],
            /* reshape */ true, &temp_tensor_indices_[GPU_WEIGHT]);
        gpu_output = tensor_manage_util->CreateOrReshapePartTensor(
            output, partition_dim,
            0, part_plan_->gpu_output_part_shape()[C_NHWC],
            /* reshape */ false, &temp_tensor_indices_[GPU_OUTPUT]);
      }

      // Prepare CPU tensors.
      //ShowInfo("Prepare CPU tensors");
      const DataFormat in_out_data_format = DataFormat::NCHW;
      const DeviceType in_out_device_type = DeviceType::GPU;
      const DeviceType weights_device_type = DeviceType::CPU;
      bool in_out_map_buffer_hint = false;
      bool weights_map_gpu_buffer_hint = false;
      const AllocatorMapType in_map_type = AllocatorMapType::AMT_READ_ONLY;
      const AllocatorMapType out_map_type = AllocatorMapType::AMT_WRITE_ONLY;
      Tensor *cpu_input = nullptr, *cpu_output = nullptr;
      Tensor *cpu_weight1 = nullptr, *cpu_weight2 = nullptr;
      Tensor *cpu_bias1 = nullptr, *cpu_bias2 = nullptr;
      cpu_input = tensor_manage_util->CreateOrReshapeTensor(
          part_plan_->cpu_input_part_shape(),
          DT_FLOAT, false, in_out_data_format,
          in_out_device_type, std::string("cpu_input"),
          in_out_map_buffer_hint, in_map_type,
          CpuBufferIdx::BUF_IDX_IN, AllocateType::ALLOC_TYPE_REUSE,
          &temp_tensor_indices_[CPU_INPUT]);
      cpu_weight1 = tensor_manage_util->CreateOrReshapeTensor(
          weight->shape(),
          DT_FLOAT, weight->is_weight(), DataFormat::OIHW,
          weights_device_type, std::string("cpu_weight1"),
          weights_map_gpu_buffer_hint, in_map_type,
          CpuBufferIdx::BUF_IDX_WEIGHTS, AllocateType::ALLOC_TYPE_GROW,
          &temp_tensor_indices_[CPU_WEIGHT1]);
      if (bias != nullptr) {
        cpu_bias1 = tensor_manage_util->CreateOrReshapeTensor(
            bias->shape(),
            DT_FLOAT, bias->is_weight(), DataFormat::NONE,
            weights_device_type, std::string("cpu_bias1"),
            weights_map_gpu_buffer_hint, in_map_type,
            CpuBufferIdx::BUF_IDX_WEIGHTS, AllocateType::ALLOC_TYPE_GROW,
            &temp_tensor_indices_[CPU_BIAS1]);
      }
      cpu_output = tensor_manage_util->CreateOrReshapeTensor(
          part_plan_->cpu_output_part_shape(),
          DT_FLOAT, false, in_out_data_format,
          in_out_device_type, std::string("cpu_output"),
          in_out_map_buffer_hint, out_map_type,
          CpuBufferIdx::BUF_IDX_OUT, AllocateType::ALLOC_TYPE_REUSE,
          &temp_tensor_indices_[CPU_OUTPUT]);

      // Transform weight and bias tensor if it is the first run.
      //ShowInfo("Transform weight and bias tensor");
      if (is_first_time_run_) {
        TransformWeightGpuToCpu(context, weight, cpu_weight1);
        if (bias != nullptr && cpu_bias1 != nullptr) {
          TransformBiasGpuToCpu(context, bias, cpu_bias1);
        }
        is_first_time_run_ = false;
      }

      cpu_weight2 = tensor_manage_util->CreateOrReshapeCpuConv2dWeightsTensor(
          cpu_weight1, partition_dim,
          part_plan_->cpu_weight_part_shape()[O_OIHW],
          &temp_tensor_indices_[CPU_WEIGHT2]);
      if (cpu_bias1 != nullptr) {
        cpu_bias2 = tensor_manage_util->CreateOrReshapeCpuConv2dWeightsTensor(
            cpu_bias1, partition_dim,
            part_plan_->cpu_weight_part_shape()[O_OIHW],
            &temp_tensor_indices_[CPU_BIAS2]);
      }

      // Enqueue input transforming kernel.
      //ShowInfo("Enqueue transforming input kernel");
      const OdimRanges *input_odim_ranges = part_plan_->input_odim_ranges();
      OpenCLPartBufferTransformer input_transformer(MemoryType::GPU_IMAGE,
                                                    MemoryType::GPU_BUFFER);
      MACE_RETURN_IF_ERROR(input_transformer.Transform(
          context, input, OpenCLBufferType::IN_OUT_CHANNEL,
          MemoryType::GPU_BUFFER, 0, *input_odim_ranges, cpu_input));

      // Enqueue input/output mapping kernels.
      //ShowInfo("Enqueue mapping kernel");
      cpu_input->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                           BlockFlag::BF_FALSE);
      StatsFuture map_out_future;
      cpu_output->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                            BlockFlag::BF_FALSE,
                            context->future());
      map_out_future.wait_fn = context->future()->wait_fn;

      // Enqueue GPU computation kernel.
      if (!part_plan_->IsCpuOnly()) {
        //ShowInfo("Enqueue GPU compute kernel");
        MACE_RETURN_IF_ERROR(RunGpu(context,
                                    gpu_input,
                                    gpu_weight,
                                    bias,
                                    gpu_output));
      }

      auto opencl_runtime = context->device()->gpu_runtime()->opencl_runtime();
      OpenCLEventManager *event_manager = opencl_runtime->event_manager();

      // Enqueue input/output unmapping kernels.
      //ShowInfo("Enqueue unmapping kernel");
      cl::UserEvent *transform_out_user_event =
          event_manager->CreateSingleUserEvent(
              opencl_runtime->context(),
              EventActionType::WAIT,
              EventOpType::TRANSFORM_OUTPUT);
      cpu_input->UnmapBuffer();
      event_manager->InsertNullEvent(EventActionType::WAIT);
      cpu_output->UnmapBuffer();

      // Enqueue output transforming kernel.
      //ShowInfo("Enqueue transfomring output kernel");
      const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
      OpenCLPartBufferTransformer output_transformer(MemoryType::GPU_BUFFER,
                                                     MemoryType::GPU_IMAGE);
      MACE_RETURN_IF_ERROR(output_transformer.Transform(
          context, cpu_output, OpenCLBufferType::IN_OUT_CHANNEL,
          MemoryType::GPU_IMAGE, 0, *output_odim_ranges, output));

      // Synchronize for output mapping.
      //ShowInfo("Sync for mapping");
      map_out_future.wait_fn(nullptr);

      // Run CPU computation kernel.
      //ShowInfo("Run CPU compute kernel");
      MACE_RETURN_IF_ERROR(RunCpu(context,
                                  cpu_input,
                                  cpu_weight2,
                                  cpu_bias2,
                                  cpu_output));

      // Synchronize for input unmapping.
      //ShowInfo("Sync for unmapping");
      event_manager->SetUserEventComplete(transform_out_user_event);

      return MaceStatus::MACE_SUCCESS;
    }
  }

  return RunGpu(context);
}

}  // namespace ops
}  // namespace mace
