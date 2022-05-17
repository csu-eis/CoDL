
#include "test/codl_run/fully_connected_test_param.h"
#include "test/codl_run/fully_connected_test_task.h"

namespace mace {

int CodlFullyConnectedCpuGpuTestTask::Prepare(TestParam *test_param) {
  FullyConnectedTestParam *fc_test_param =
      reinterpret_cast<FullyConnectedTestParam *>(test_param);

  // Load common parameters.
  const int num_threads = fc_test_param->num_threads;
  const CPUAffinityPolicy policy
      = static_cast<CPUAffinityPolicy>(fc_test_param->cpu_affinity_policy);
  is_debug_on_ = fc_test_param->is_debug_on;
  do_data_transform_ = fc_test_param->do_data_transform;
  do_compute_ = fc_test_param->do_compute;
  const MemoryType gpu_memory_type = fc_test_param->gpu_memory_type;
  const PartitionDim part_dim = static_cast<PartitionDim>(fc_test_param->part_dim);
  const float part_ratio = fc_test_param->part_ratio;
  compute_unit_hint_     = fc_test_param->compute_unit_hint;
  const DataType cpu_dtype = fc_test_param->cpu_dtype;
  const DataType gpu_dtype = fc_test_param->gpu_dtype;

  activation_            = ops::ActivationType::NOOP;
  relux_max_limit_       = 0.0f;
  leakyrelu_coefficient_ = 0.0f;

  const std::vector<index_t> input_shape = fc_test_param->input_shape;
  const std::vector<index_t> weight_shape = fc_test_param->weight_shape;
  std::vector<index_t> output_shape;
  std::vector<index_t> bias_shape;
  ops::FullyConnectedPartPlanUtils::CalcOutputShape(input_shape,
                                                    weight_shape,
                                                    output_shape);
  ops::FullyConnectedPartPlanUtils::CalcBiasShape(output_shape, bias_shape);

  // Create partition plan.
  ShowText("Create partition plan");
  DataFormat cpu_data_format = DataFormat::NCHW;
  if (cpu_dtype == DataType::DT_UINT8) {
    cpu_data_format = DataFormat::NHWC;
  }

  ops::FullyConnectedPartPlan *fc_part_plan = new ops::FullyConnectedPartPlan(
      part_dim, part_ratio, cpu_data_format);
  ops::PartitionResult partition_result;
  do {
    partition_result = fc_part_plan->Make(input_shape, weight_shape);
  } while (partition_result == ops::PartitionResult::PARTITION_REDO);

  if (partition_result == ops::PartitionResult::PARTITION_FAILED) {
    LOG(ERROR) << "Make partition plan failed";
    return -1;
  } else {
    fc_part_plan->Show();
  }

  part_plan_.reset(fc_part_plan);

  // Initialize device context.
  ShowText("Initialize device context");
  TestDeviceContext *dev_context = GetDeviceContext();
  if (dev_context == nullptr) {
    SetDeviceContext(new TestDeviceContext(num_threads, policy));
    dev_context = GetDeviceContext();
  }
  if (!dev_context->is_initialized()) {
    dev_context->InitCpuDevice();
    dev_context->InitGpuDevice();
    dev_context->set_is_initialized(true);
  }
  // Initialize cpu and gpu context.
  gpu_context_.reset(new OpContext(GetWorkspace(), dev_context->GetGpuDevice()));
  cpu_context_.reset(new OpContext(GetWorkspace(), dev_context->GetCpuDevice()));
  cpu_context_->set_cpu_device(dev_context->GetCpuDevice());

  const bool use_opencl_image = (gpu_memory_type == MemoryType::GPU_IMAGE);
  const DataType in_out_dt = gpu_dtype;
  const DataType weights_dt = gpu_dtype;
  MACE_CHECK(gpu_dtype == DataType::DT_FLOAT || gpu_dtype == DataType::DT_HALF);

  if (!use_opencl_image) {
    // No data transforming for gpu buffer.
    do_data_transform_ = false;
  }

  ShowText("Create gpu tensor");
  if (use_opencl_image) {
    // Create GPU (image) full tensors.
    input_ = TensorUtils::CreateGPUImageTensor(
        dev_context, input_shape, in_out_dt, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("input_image"));
    weight_ = TensorUtils::CreateGPUImageTensor(
        dev_context, weight_shape, weights_dt, true,
        DataFormat::OIHW, OpenCLBufferType::WEIGHT_WIDTH,
        std::string("weight_image"));
    output_ = TensorUtils::CreateGPUImageTensor(
        dev_context, output_shape, in_out_dt, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("output_image"));
    bias_ = TensorUtils::CreateGPUImageTensor(
        dev_context, bias_shape, weights_dt, false,
        DataFormat::NONE, OpenCLBufferType::ARGUMENT,
        std::string("bias_image"));
  } else {
    // Create GPU (buffer) tensors.
    input_ = TensorUtils::CreateBufferTensor(
        dev_context, input_shape, in_out_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("input_buffer"), true);
    weight_ = TensorUtils::CreateBufferTensor(
        dev_context, weight_shape, weights_dt, true,
        DataFormat::OIHW, DeviceType::GPU,
        std::string("weight_buffer"), true);
    output_ = TensorUtils::CreateBufferTensor(
        dev_context, output_shape, in_out_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("output_buffer"), true);
    bias_ = TensorUtils::CreateBufferTensor(
        dev_context, bias_shape, weights_dt, false,
        DataFormat::NONE, DeviceType::GPU,
        std::string("bias_buffer"), true);
  }

  const std::vector<index_t>
      part_shape_input_gpu = part_plan_->gpu_input_part_shape();
  const std::vector<index_t>
      part_shape_weight_gpu = fc_part_plan->gpu_weight_part_shape();
  const std::vector<index_t>
      part_shape_output_gpu = part_plan_->gpu_output_part_shape();
  const std::vector<index_t>
      part_shape_input_cpu  = part_plan_->cpu_input_part_shape();
  const std::vector<index_t>
      part_shape_weight_cpu = fc_part_plan->cpu_weight_part_shape();
  const std::vector<index_t>
      part_shape_output_cpu = part_plan_->cpu_output_part_shape();

  tensor_manage_util_.reset(new TensorManageUtil(
      dev_context->GetCpuDevice()->allocator()));
  tensor_manage_util_->set_gpu_allocator(
      dev_context->GetGpuDevice()->allocator());
  //OpenCLRuntime *opencl_runtime =
  //    dev_context->GetGpuDevice()->gpu_runtime()->opencl_runtime();

  tensor_manage_util_->Manage(input_);
  tensor_manage_util_->Manage(weight_);
  tensor_manage_util_->Manage(output_);
  tensor_manage_util_->Manage(bias_);

  // Create cpu tensors.
  ShowText("Create cpu tensors");
  if (!part_plan_->IsGpuOnly()) {
    if (!part_plan_->IsCpuOnly()) {
      index_t in_out_dim_idx = H_NHWC;
      index_t part_length;
      if (part_dim == DIM_INPUT_HEIGHT) {
        in_out_dim_idx = H_NHWC;
      } else if (part_dim == DIM_OUTPUT_CHANNEL) {
        in_out_dim_idx = C_NHWC;
      } else {
        MACE_NOT_IMPLEMENTED;
      }
      part_length = part_shape_input_gpu[in_out_dim_idx];
      input_gpu_  = tensor_manage_util_->CreateOrReshapePartTensor(
                      input_, part_dim, 0, part_length, true);
      part_length = part_shape_weight_gpu[O_OIHW];
      weight_gpu_ = tensor_manage_util_->CreateOrReshapePartTensor(
                      weight_, part_dim, 0, part_length, true);
      part_length = part_shape_output_gpu[in_out_dim_idx];
      output_gpu_ = tensor_manage_util_->CreateOrReshapePartTensor(
                      output_, part_dim, 0, part_length, true);
    }

    const DeviceType in_out_device_type = DeviceType::GPU;
    const DeviceType weight_device_type = DeviceType::GPU;
    bool in_out_mapping_hint = false;
    const bool weights_mapping_hint = true;

    input_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_input_cpu, DT_FLOAT, false,
        DataFormat::NCHW, in_out_device_type,
        std::string("input_cpu"), in_out_mapping_hint);
    weight_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_weight_cpu, DT_FLOAT, weight_->is_weight(),
        weight_->data_format(), weight_device_type,
        std::string("weight_cpu"), weights_mapping_hint);
    output_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_output_cpu, DT_FLOAT, false,
        DataFormat::NCHW, in_out_device_type,
        std::string("output_cpu"), in_out_mapping_hint);
    bias_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        bias_->shape(), DT_FLOAT, false,
        bias_->data_format(), in_out_device_type,
        std::string("bias_cpu"), weights_mapping_hint);
  }

  // Create computing kernel.
  ShowText("Create cpu computing kernel");
  if (cpu_dtype == DataType::DT_FLOAT) {
    cpu_kernel_.reset(new FullyConnectedCpuFloatKernel(
        activation_, relux_max_limit_, leakyrelu_coefficient_));
  } else if (cpu_dtype == DataType::DT_UINT8) {
    cpu_kernel_.reset(new FullyConnectedCpuUint8Kernel());
    input_cpu_->SetScale(0.5);
    weight_cpu_->SetScale(0.5);
    output_cpu_->SetScale(0.5);
  } else {
    LOG(ERROR) << "Unsupported cpu data type " << static_cast<int>(cpu_dtype);
    MACE_NOT_IMPLEMENTED;
  }

  ShowText("Create gpu computing kernel");
  const MemoryType opencl_mtype = use_opencl_image ? MemoryType::GPU_IMAGE
                                                   : MemoryType::GPU_BUFFER;
  opencl_kernel_ = CreateOpenCLFullyConnectedKernel(opencl_mtype);

  // Warm up.
  //xxx

  return 1;
}

int CodlFullyConnectedCpuGpuTestTask::EnqueueInputDataTransformKernel(
    StatsFuture *future) {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_data_transform_) {
    return 0;
  }

  StatsFuture *old_future = gpu_context_->future();
  gpu_context_->set_future(future);

  const OdimRanges *input_odim_ranges = part_plan_->input_odim_ranges();
  ops::OpenCLPartBufferTransformer input_transformer(MemoryType::GPU_IMAGE,
                                                     MemoryType::GPU_BUFFER);
  input_transformer.Transform(gpu_context_.get(),
                              input_,
                              OpenCLBufferType::IN_OUT_CHANNEL,
                              MemoryType::GPU_BUFFER,
                              0,
                              *input_odim_ranges,
                              input_cpu_);
  
  gpu_context_->set_future(old_future);

  return 0;
}

int CodlFullyConnectedCpuGpuTestTask::EnqueueMapKernel(
    StatsFuture *future,
    StatsFuture *map_in_future) {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_compute_) {
    return 0;
  }

  StatsFuture *old_future = gpu_context_->future();
  gpu_context_->set_future(future);

  input_cpu_->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                        BlockFlag::BF_FALSE,
                        map_in_future);
  output_cpu_->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                         BlockFlag::BF_FALSE,
                         future);
  
  gpu_context_->set_future(old_future);

  return 0;
}

int CodlFullyConnectedCpuGpuTestTask::EnqueueGpuComputeKerenl(
    StatsFuture *future) {
  if (part_plan_->IsCpuOnly()) {
    return 0;
  }
  if (!do_compute_) {
    return 0;
  }

  StatsFuture *old_future = gpu_context_->future();
  gpu_context_->set_future(future);

  if (!part_plan_->IsGpuOnly()) {
    opencl_kernel_->Compute(
        gpu_context_.get(),
        input_gpu_, weight_gpu_, bias_,
        activation_, relux_max_limit_, leakyrelu_coefficient_,
        output_gpu_);
  } else {
    opencl_kernel_->Compute(
        gpu_context_.get(),
        input_, weight_, bias_,
        activation_, relux_max_limit_, leakyrelu_coefficient_,
        output_);
  }

  gpu_context_->set_future(old_future);

  return 0;
}

int CodlFullyConnectedCpuGpuTestTask::EnqueueUnmapKernel(
    cl::UserEvent **event,
    StatsFuture *unmap_in_future,
    StatsFuture *unmap_out_future) {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_compute_) {
    return 0;
  }

  OpenCLRuntime *opencl_runtime =
      GetDeviceContext()->GetGpuDevice()->gpu_runtime()->opencl_runtime();
  OpenCLEventManager *event_manager = opencl_runtime->event_manager();
  *event = event_manager->CreateSingleUserEvent(
      opencl_runtime->context(),
      EventActionType::WAIT,
      EventOpType::TRANSFORM_OUTPUT);
  input_cpu_->UnmapBuffer(unmap_in_future);
  event_manager->InsertNullEvent(EventActionType::WAIT);
  output_cpu_->UnmapBuffer(unmap_out_future);
  return 0;
}

int CodlFullyConnectedCpuGpuTestTask::EnqueueOutputDataTransformKernel(
    StatsFuture *future) {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_data_transform_) {
    return 0;
  }

  StatsFuture *old_future = gpu_context_->future();
  gpu_context_->set_future(future);

  const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
  ops::OpenCLPartBufferTransformer output_transformer(MemoryType::GPU_BUFFER,
                                                      MemoryType::GPU_IMAGE);
  output_transformer.Transform(gpu_context_.get(),
                               output_cpu_,
                               OpenCLBufferType::IN_OUT_CHANNEL,
                               MemoryType::GPU_IMAGE,
                               0,
                               *output_odim_ranges,
                               output_);
  
  gpu_context_->set_future(old_future);

  return 0;
}

int CodlFullyConnectedCpuGpuTestTask::RunCpuComputeKernel() {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_compute_) {
    return 0;
  }

  cpu_kernel_->Compute(cpu_context_.get(),
                       input_cpu_,
                       weight_cpu_,
                       bias_cpu_,
                       output_cpu_);
  return 0;
}

}  // namespace mace
