
#include "mace/ops/common/transpose_util.h"
#include "mace/ops/fully_connected.h"

namespace mace {
namespace ops {

MaceStatus FullyConnectedGpuFloatOp::RunCpuGpuBuffer(
    OpContext *context) {
  if (context->part_run_config() == nullptr) {
    return RunGpu(context);
  }

  Tensor *input = const_cast<Tensor *>(this->Input(INPUT));
  Tensor *weight = const_cast<Tensor *>(this->Input(WEIGHT));
  Tensor *bias = this->InputSize() >= 3 ?
      const_cast<Tensor *>(this->Input(BIAS)) : nullptr;
  Tensor *output = this->Output(OUTPUT);
  
  kernel_->ResizeOutputTensor(input, weight, output);

  // Prepare partition parameters.
  const PartitionDim partition_dim = context->part_run_config()->dim_type();
  float partition_ratio = context->part_run_config()->ratio();
  if (partition_ratio == kRatioFromConfigFile ||
      partition_ratio == kRatioFromPredictor) {
    partition_ratio = context->partition_configer()->ratio();
  }

  MACE_CHECK(partition_dim == PartitionDim::DIM_OUTPUT_CHANNEL,
      "Only output channel-wise is supported when use gpu buffer");

  // Make partition plan.
#if 0
  DataFormat df = DataFormat::NCHW;
  if (part_plan_) {
    if (context->op_runtime_mode() == OpRuntimeMode::RUNNING) {
      if (part_plan_->CheckPlanChanged(partition_dim, partition_ratio)) {
        part_plan_.reset(new FullyConnectedPartPlan(partition_dim,
                                                    partition_ratio,
                                                    df));
        part_plan_->Make(input->shape(), weight->shape());
        part_plan_->Show();
      }
    }
  } else {
    if (partition_ratio == kRatioTuning) {
      partition_ratio = 0.0;
    }
    part_plan_.reset(new FullyConnectedPartPlan(partition_dim,
                                                partition_ratio,
                                                df));
    part_plan_->Make(input->shape(), weight->shape());
    part_plan_->Show();
  }
#else
  MakePartitionPlan(context, input, weight, partition_dim, partition_ratio);
#endif

  if (part_plan_->CheckIsReady()) {
    if (!part_plan_->IsGpuOnly()) {
      const bool do_map_unmap = true;
      const bool do_cpu_computation = true;
      const bool do_gpu_computation = true;

      auto opencl_runtime = context->device()->gpu_runtime()->opencl_runtime();
      OpenCLEventManager *event_manager = opencl_runtime->event_manager();
      
      // Create or reshape partial tensors.
      TensorManageUtil *tensor_manage_util
          = context->workspace()->tensor_manage_util();
      const index_t gpu_out_channels = part_plan_->gpu_weight_part_shape()[O_OIHW];
      const index_t cpu_out_channels = part_plan_->cpu_weight_part_shape()[O_OIHW];
      Tensor *gpu_weight = nullptr;
      Tensor *gpu_output = nullptr;
      if (!part_plan_->IsCpuOnly()) {
        gpu_weight = tensor_manage_util->CreateOrReshapePartTensor(
            weight, partition_dim, 0, gpu_out_channels,
            true, &temp_tensor_indices_[GPU_WEIGHT]);
        gpu_output = tensor_manage_util->CreateOrReshapePartTensor(
            output, partition_dim, 0, gpu_out_channels,
            true, &temp_tensor_indices_[GPU_OUTPUT]);
      }
      Tensor *cpu_weight = tensor_manage_util->CreateOrReshapePartTensor(
          weight, partition_dim, gpu_out_channels, cpu_out_channels,
          true, &temp_tensor_indices_[CPU_WEIGHT1]);
      Tensor *cpu_output = tensor_manage_util->CreateOrReshapePartTensor(
          output, partition_dim, gpu_out_channels, cpu_out_channels,
          true, &temp_tensor_indices_[CPU_OUTPUT]);
      //LOG(INFO) << "Create or reshape partial tensor success";

      // Enqueue map in/out tensor.
      StatsFuture map_out_future;
      if (do_map_unmap) {
        input->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                         BlockFlag::BF_FALSE,
                         nullptr);
        cpu_weight->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                              BlockFlag::BF_FALSE,
                              nullptr);
        if (bias != nullptr) {
          bias->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                          BlockFlag::BF_FALSE,
                          nullptr);
        }
        cpu_output->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                              BlockFlag::BF_FALSE,
                              &map_out_future);

        MACE_CHECK(input->IsBufferMapped());
        MACE_CHECK(cpu_weight->IsBufferMapped());
        if (bias != nullptr) {
          MACE_CHECK(bias->IsBufferMapped());
        }
        MACE_CHECK(cpu_output->IsBufferMapped());
        //LOG(INFO) << "Enqueue map buffer kernel success";
      }

      if (!part_plan_->IsCpuOnly()) {
        if (do_gpu_computation) {
          // Check dimension size.
          MACE_CHECK(input->dim_size() > 0, "name ", input->name());
          MACE_CHECK(gpu_weight->dim_size() > 0, "name ", gpu_weight->name());
          if (bias != nullptr) {
            MACE_CHECK(bias->dim_size() > 0, "name ", bias->name());
          }
          MACE_CHECK(gpu_output->dim_size() > 0, "name ", gpu_output->name());

          // Enqueue gpu computation kernel.
          RunGpu(context, input, gpu_weight, bias, gpu_output);
          //LOG(INFO) << "Enqueue gpu kernel success";
        }
      }

      // Enqueue unmap in/out tensor.
      cl::UserEvent *unmap_in_user_event = nullptr;
      if (do_map_unmap) {
        unmap_in_user_event = event_manager->CreateSingleUserEvent(
            opencl_runtime->context(),
            EventActionType::WAIT,
            EventOpType::TRANSFORM_OUTPUT);
        input->UnmapBuffer();
        event_manager->InsertNullEvent(EventActionType::WAIT);
        cpu_weight->UnmapBuffer();
        if (bias != nullptr) {
          bias->UnmapBuffer();
        }
        cpu_output->UnmapBuffer();
        //LOG(INFO) << "Enqueue unmap kernel success";
      }

      // Wait (and run) map event.
      map_out_future.wait_fn(nullptr);

      // CPU computation kernel.
      if (do_cpu_computation) {
        // Transpose.
        Tensor *tmp_input = const_cast<Tensor *>(input);
        TransposeUtil::TransposeTensorShape(tmp_input, DST_DIMS_NHWC_TO_NCHW);
        TransposeUtil::TransposeTensorShape(output, DST_DIMS_NHWC_TO_NCHW);
        
        // Check dimension size.
        MACE_CHECK(tmp_input->dim_size() > 0, "name ", input->name());
        MACE_CHECK(cpu_weight->dim_size() > 0, "name ", cpu_weight->name());
        if (bias != nullptr) {
          MACE_CHECK(bias->dim_size() > 0, "name ", bias->name());
        }
        MACE_CHECK(cpu_output->dim_size() > 0, "name ", cpu_output->name());

        RunCpu(context, tmp_input, cpu_weight, bias, cpu_output);
        
        // Tranpose back.
        TransposeUtil::TransposeTensorShape(tmp_input, DST_DIMS_NCHW_TO_NHWC);
        TransposeUtil::TransposeTensorShape(output, DST_DIMS_NCHW_TO_NHWC);
        //LOG(INFO) << "Run cpu kernel success";
      }

      // Set unmap waiting event to completion status.
      if (unmap_in_user_event != nullptr) {
        event_manager->SetUserEventComplete(unmap_in_user_event);
      }
      
      return MaceStatus::MACE_SUCCESS;
    } else {
      return RunGpu(context);
    }
  } else {
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
}

}  // namespace ops
}  // namespace mace
