
#include "mace/ops/common/transpose_util.h"
#include "mace/ops/deconv_2d.h"

namespace mace {
namespace ops {

MaceStatus Deconv2dGpuFloatOp::RunCpuGpuBuffer(OpContext *context) {
  if (context->part_run_config() == nullptr) {
    return RunGpu(context);
  }

  //ShowInfo("Prepare tensors");
  Tensor *input = const_cast<Tensor *>(this->Input(0));
  Tensor *filter = const_cast<Tensor *>(this->Input(1));
  Tensor *bias = nullptr;
  Tensor *output_shape_tensor = nullptr;
  if (model_type_ == TENSORFLOW) {
    output_shape_tensor =
        this->InputSize() >= 3 ? const_cast<Tensor *>(this->Input(2)) : nullptr;
    bias = this->InputSize() >= 4 ? const_cast<Tensor *>(this->Input(3)) : nullptr;
  } else {
    bias = this->InputSize() >= 3 ? const_cast<Tensor *>(this->Input(2)) : nullptr;
  }
  Tensor *output = this->Output(0);

  MACE_CHECK_NOTNULL(input);
  MACE_CHECK_NOTNULL(filter);
  MACE_CHECK_NOTNULL(output);

  std::vector<index_t> out_shape;
  if (output_shape_tensor) {
    Tensor::MappingGuard out_shape_guard(output_shape_tensor);
    MACE_CHECK(output_shape_tensor->size() == 4,
               "output shape should be 4-dims");
    out_shape =
        std::vector<index_t>(output_shape_tensor->data<int32_t>(),
                             output_shape_tensor->data<int32_t>() + 4);
    //LOG(INFO) << "out_shape " << VectorToString<index_t>(out_shape);
  }
  std::vector<int> in_paddings;
  std::vector<int> out_paddings;

  CalDeconvOutputShapeAndPadSize(input->shape(),
                                 filter->shape(),
                                 strides_,
                                 padding_type_,
                                 paddings_,
                                 output_paddings_,
                                 1,
                                 &out_shape,
                                 &in_paddings,
                                 &out_paddings,
                                 nullptr,
                                 model_type_,
                                 DataFormat::NHWC);
  
  kernel_->ResizeOutputTensor(out_shape, output);

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
  MakePartitionPlan(context,
                    output_shape_tensor,
                    input,
                    filter,
                    partition_dim,
                    partition_ratio);

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
      const index_t gpu_out_channels = part_plan_->gpu_filter_part_shape()[O_OIHW];
      const index_t cpu_out_channels = part_plan_->cpu_filter_part_shape()[O_OIHW];
      Tensor *gpu_filter = nullptr;
      Tensor *gpu_output = nullptr;
      if (!part_plan_->IsCpuOnly()) {
        gpu_filter = tensor_manage_util->CreateOrReshapePartTensor(
            filter, partition_dim, 0, gpu_out_channels,
            true, &temp_tensor_indices_[GPU_FILTER]);
        gpu_output = tensor_manage_util->CreateOrReshapePartTensor(
            output, partition_dim, 0, gpu_out_channels,
            true, &temp_tensor_indices_[GPU_OUTPUT]);
      }
      Tensor *cpu_filter = tensor_manage_util->CreateOrReshapePartTensor(
          filter, partition_dim, gpu_out_channels, cpu_out_channels,
          true, &temp_tensor_indices_[CPU_FILTER1]);
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
        cpu_filter->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
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
        MACE_CHECK(cpu_filter->IsBufferMapped());
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
          MACE_CHECK(gpu_filter->dim_size() > 0, "name ", gpu_filter->name());
          if (bias != nullptr) {
            MACE_CHECK(bias->dim_size() > 0, "name ", bias->name());
          }
          MACE_CHECK(gpu_output->dim_size() > 0, "name ", gpu_output->name());

          // Enqueue gpu computation kernel.
          RunGpu(context, input, gpu_filter, bias, gpu_output);
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
        cpu_filter->UnmapBuffer();
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
        MACE_CHECK(cpu_filter->dim_size() > 0, "name ", cpu_filter->name());
        if (bias != nullptr) {
          MACE_CHECK(bias->dim_size() > 0, "name ", bias->name());
        }
        MACE_CHECK(cpu_output->dim_size() > 0, "name ", cpu_output->name());

        RunCpu(context, tmp_input, cpu_filter, bias, nullptr, cpu_output);
        
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
