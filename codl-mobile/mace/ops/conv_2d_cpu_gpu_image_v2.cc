
#include <limits>
#include "mace/ops/arm/fp32/gemm.h"
#include "mace/ops/conv_2d.h"
#include "mace/utils/io_util.h"

namespace mace {
namespace ops {

#ifndef CODL_ENABLE_MACE_CONV2D_GPU

#ifdef MACE_ENABLE_OPENCL
#ifdef MACE_ENABLE_CODL

#if 0
static void ShowLatency(const std::string name, const double latency) {
  LOG(INFO) << name << ": " << latency << " ms";
}
#endif

MaceStatus Conv2dGpuFloatOp::RunCpuGpuImage(OpContext *context) {
  if (context->part_run_config() == nullptr) {
    return RunGpu(context);
  }

  // Prepare tensors.
  const Tensor *input  = this->Input(INPUT);
  const Tensor *filter = this->Input(FILTER);
  const Tensor *bias   = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
  Tensor *output       = this->Output(OUTPUT);

  // Prepare partition plan.
  PartitionDim partition_dim = context->part_run_config()->dim_type();
  float partition_ratio = context->part_run_config()->ratio();
  if (partition_ratio == kRatioFromConfigFile ||
      partition_ratio == kRatioFromPredictor) {
    partition_dim = static_cast<PartitionDim>(context->partition_configer()->dim());
    partition_ratio = context->partition_configer()->ratio();
  }

  // Make partition plan.
  if (part_plan_) {
    MACE_CHECK(context->op_runtime_mode() == OpRuntimeMode::RUNNING);
    if (part_plan_->CheckPlanChanged(partition_dim, partition_ratio)) {
      const DataFormat df = DataFormat::NCHW;
      part_plan_.reset(new ConvPool2dPartPlan(partition_dim, partition_ratio, df));
      PartitionResult partition_result;
      do {
        partition_result = part_plan_->Make(input->shape(),
                                            filter->shape(),
                                            strides_,
                                            dilations_,
                                            padding_type_,
                                            paddings_);
      } while (partition_result == PartitionResult::PARTITION_REDO);
      part_plan_->Show();
    }
  } else {
    const DataFormat df = DataFormat::NCHW;
    part_plan_.reset(new ConvPool2dPartPlan(partition_dim, partition_ratio, df));
    PartitionResult partition_result;
    do {
      partition_result = part_plan_->Make(input->shape(),
                                          filter->shape(),
                                          strides_,
                                          dilations_,
                                          padding_type_,
                                          paddings_);
    } while (partition_result == PartitionResult::PARTITION_REDO);
    part_plan_->Show();
  }

  // CPU+GPU co-execution.
  int64_t t0;
  if (part_plan_->CheckIsReady()) {
    if (!part_plan_->IsGpuOnly()) {
      // Make sure output tensor keeps the shape.
      kernel_->ResizeOutputTensor(input, filter, strides_.data(),
                                  padding_type_, paddings_,
                                  dilations_.data(), output);

      // Pad input tensor.
      const Tensor *padded_input = RunPadInput(context);

      // Prepare GPU tensors.
      TensorManageUtil *tensor_manage_util = context->workspace()->tensor_manage_util();

      Tensor *gpu_input = nullptr;
      Tensor *gpu_filter = nullptr;
      Tensor *gpu_output = nullptr;
      if (!part_plan_->IsCpuOnly()) {
        index_t in_out_dim_idx = partition_dim == DIM_OUTPUT_CHANNEL ? C_NHWC : H_NHWC;
        gpu_input = tensor_manage_util->CreateOrReshapePartTensor(
            padded_input, partition_dim,
            0, part_plan_->gpu_input_part_shape()[in_out_dim_idx],
            /* reshape */ true, &temp_tensor_indices_[GPU_INPUT]);
        gpu_filter = tensor_manage_util->CreateOrReshapePartTensor(
            filter, partition_dim,
            0, part_plan_->gpu_filter_part_shape()[O_OIHW],
            /* reshape */ true, &temp_tensor_indices_[GPU_FILTER]);
        gpu_output = tensor_manage_util->CreateOrReshapePartTensor(
            output, partition_dim,
            0, part_plan_->gpu_output_part_shape()[in_out_dim_idx],
            /* reshape */ false, &temp_tensor_indices_[GPU_OUTPUT]);
      }

      // Prepare CPU tensors.
      const DataFormat in_out_data_format = DataFormat::NCHW;
      const DeviceType in_out_device_type = DeviceType::GPU;
      const DeviceType weights_device_type = DeviceType::CPU;
      bool in_out_map_buffer_hint = false;
      bool weights_map_gpu_buffer_hint = false;
      const AllocatorMapType in_map_type = AllocatorMapType::AMT_READ_ONLY;
      const AllocatorMapType out_map_type = AllocatorMapType::AMT_WRITE_ONLY;
      Tensor *cpu_input = nullptr, *cpu_output = nullptr;
      Tensor *cpu_filter_v1 = nullptr, *cpu_filter_v2 = nullptr;
      Tensor *cpu_bias_v1 = nullptr, *cpu_bias_v2 = nullptr;
      cpu_input = tensor_manage_util->CreateOrReshapeTensor(
          part_plan_->cpu_input_part_shape(),
          DT_FLOAT, false, in_out_data_format,
          in_out_device_type, std::string("cpu_input"),
          in_out_map_buffer_hint, in_map_type,
          CpuBufferIdx::BUF_IDX_IN, AllocateType::ALLOC_TYPE_REUSE,
          &temp_tensor_indices_[CPU_INPUT]);
      cpu_filter_v1 = tensor_manage_util->CreateOrReshapeTensor(
          filter->shape(),
          DT_FLOAT, filter->is_weight(), DataFormat::OIHW,
          weights_device_type, std::string("cpu_filter"),
          weights_map_gpu_buffer_hint, in_map_type,
          CpuBufferIdx::BUF_IDX_WEIGHTS, AllocateType::ALLOC_TYPE_GROW,
          &temp_tensor_indices_[CPU_FILTER_V1]);
      if (bias != nullptr) {
        cpu_bias_v1 = tensor_manage_util->CreateOrReshapeTensor(
            bias->shape(),
            DT_FLOAT, bias->is_weight(), DataFormat::NONE,
            weights_device_type, std::string("cpu_bias"),
            weights_map_gpu_buffer_hint, in_map_type,
            CpuBufferIdx::BUF_IDX_WEIGHTS, AllocateType::ALLOC_TYPE_GROW,
            &temp_tensor_indices_[CPU_BIAS_V1]);
      }
      cpu_output = tensor_manage_util->CreateOrReshapeTensor(
          part_plan_->cpu_output_part_shape(),
          DT_FLOAT, false, in_out_data_format,
          in_out_device_type, std::string("cpu_output"),
          in_out_map_buffer_hint, out_map_type,
          CpuBufferIdx::BUF_IDX_OUT, AllocateType::ALLOC_TYPE_REUSE,
          &temp_tensor_indices_[CPU_OUTPUT]);

      // Transform filter and bias tensor if it is the first run.
      if (is_first_time_run_) {
#if 1
        TransformWeightGpuToCpu(context, filter, cpu_filter_v1);
        if (bias != nullptr && cpu_bias_v1 != nullptr) {
          TransformBiasGpuToCpu(context, bias, cpu_bias_v1);
        }
        is_first_time_run_ = false;
#endif
      }

      cpu_filter_v2 = tensor_manage_util->CreateOrReshapeCpuConv2dWeightsTensor(
          cpu_filter_v1, partition_dim,
          part_plan_->cpu_filter_part_shape()[O_OIHW],
          &temp_tensor_indices_[CPU_FILTER_V2]);
      if (cpu_bias_v1 != nullptr) {
        cpu_bias_v2 = tensor_manage_util->CreateOrReshapeCpuConv2dWeightsTensor(
            cpu_bias_v1, partition_dim,
            part_plan_->cpu_filter_part_shape()[O_OIHW],
            &temp_tensor_indices_[CPU_BIAS_V2]);
      }

#if 1
      // Enqueue input transforming kernel.
      const OdimRanges *input_odim_ranges = part_plan_->input_odim_ranges();
      OpenCLPartBufferTransformer input_transformer(MemoryType::GPU_IMAGE,
                                                    MemoryType::GPU_BUFFER);
      MACE_RETURN_IF_ERROR(input_transformer.Transform(
          context, padded_input, OpenCLBufferType::IN_OUT_CHANNEL,
          MemoryType::GPU_BUFFER, wino_block_size_, *input_odim_ranges, cpu_input));
#endif

      // Enqueue input/output mapping kernels.
      cpu_input->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                           BlockFlag::BF_FALSE);
      StatsFuture map_out_future;
      cpu_output->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                            BlockFlag::BF_FALSE,
                            context->future());
      map_out_future.wait_fn = context->future()->wait_fn;

#if 1
      // Enqueue GPU computation kernel.
      if (!part_plan_->IsCpuOnly()) {
        MACE_RETURN_IF_ERROR(RunGpu(context,
                                    gpu_input,
                                    gpu_filter,
                                    bias,
                                    gpu_output));
      }
#endif

      auto opencl_runtime = context->device()->gpu_runtime()->opencl_runtime();
      OpenCLEventManager *event_manager = opencl_runtime->event_manager();

      // Enqueue input/output unmapping kernels.
      cl::UserEvent *transform_out_user_event =
          event_manager->CreateSingleUserEvent(
              opencl_runtime->context(),
              EventActionType::WAIT,
              EventOpType::TRANSFORM_OUTPUT);
      cpu_input->UnmapBuffer();
      event_manager->InsertNullEvent(EventActionType::WAIT);
      cpu_output->UnmapBuffer();

#if 1
      // Enqueue output transforming kernel.
      const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
      OpenCLPartBufferTransformer output_transformer(MemoryType::GPU_BUFFER,
                                                     MemoryType::GPU_IMAGE);
      MACE_RETURN_IF_ERROR(output_transformer.Transform(
          context, cpu_output, OpenCLBufferType::IN_OUT_CHANNEL,
          MemoryType::GPU_IMAGE, wino_block_size_, *output_odim_ranges, output));
#endif

      // Synchronize for output mapping.
      t0 = NowMicros();
      map_out_future.wait_fn(nullptr);
      //ShowLatency("WaitMapOut", (NowMicros() - t0) / 1000.0);
      
      // Run CPU computation kernel.
      t0 = NowMicros();
      MACE_RETURN_IF_ERROR(RunCpu(context,
                                  cpu_input,
                                  cpu_filter_v2,
                                  cpu_bias_v2,
                                  cpu_output));
      //ShowLatency("RunCpu", (NowMicros() - t0) / 1000.0);

      // Synchronize for input unmapping.
      event_manager->SetUserEventComplete(transform_out_user_event);

      return MaceStatus::MACE_SUCCESS;
    }
  }
  
  return RunGpu(context);
}

MaceStatus Conv2dGpuFloatOp::RunCpuGpuImageV2(OpContext *context) {
  StatsFuture map_future;
  cl::UserEvent *unmap_event;
  OpenCLEventManager *event_manager
      = context->device()->gpu_runtime()->opencl_runtime()->event_manager();

  MakePartitionPlan(context);

  PrepareTemporaryTensors(context);

  EnqueueInputDataTransform(context);

  EnqueueMap(context, &map_future);

  EnqueueGpuCompute(context);

  EnqueueUnmap(context, &unmap_event);

  EnqueueOutputDataTransform(context);

  map_future.wait_fn(nullptr);
  
  RunCpuCompute(context);

  event_manager->SetUserEventComplete(unmap_event);

  return MaceStatus::MACE_SUCCESS;
}

#endif  // MACE_ENABLE_CODL
#endif  // MACE_ENABLE_OPENCL

#endif  // CODL_ENABLE_MACE_CONV2D_GPU

}  // namespace ops
}  // namespace mace
