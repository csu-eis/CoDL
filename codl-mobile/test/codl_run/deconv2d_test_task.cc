
#include "test/codl_run/deconv2d_test_param.h"
#include "test/codl_run/deconv2d_test_task.h"

namespace mace {

int CodlDeconv2dCpuGpuTestTask::Prepare(TestParam *test_param) {
  Deconv2dTestParam *deconv2d_test_param =
      reinterpret_cast<Deconv2dTestParam *>(test_param);

  // Load parameters for testing.
  const int num_threads = deconv2d_test_param->num_threads;
  const CPUAffinityPolicy policy
      = static_cast<CPUAffinityPolicy>(deconv2d_test_param->cpu_affinity_policy);
  is_debug_on_ = deconv2d_test_param->is_debug_on;
  do_data_transform_ = deconv2d_test_param->do_data_transform;
  do_compute_ = deconv2d_test_param->do_compute;
  const MemoryType gpu_memory_type = deconv2d_test_param->gpu_memory_type;
  const PartitionDim part_dim = static_cast<PartitionDim>(deconv2d_test_param->part_dim);
  const float part_ratio = deconv2d_test_param->part_ratio;
  compute_unit_hint_     = deconv2d_test_param->compute_unit_hint;

  padding_type_ = static_cast<Padding>(static_cast<int>(VALID));
  paddings_     = {0, 0};
  dilations_    = {1, 1};
  activation_   = ops::ActivationType::NOOP;
  relux_max_limit_       = 0.0f;
  leakyrelu_coefficient_ = 0.0f;
  
  const std::vector<index_t> input_shape = deconv2d_test_param->input_shape;
  const std::vector<index_t> filter_shape = deconv2d_test_param->filter_shape;
  strides_ = deconv2d_test_param->strides;
  std::vector<index_t> output_shape;
  std::vector<index_t> bias_shape;
  const FrameworkType farmework_type = FrameworkType::TENSORFLOW;
  ops::Deconv2dPartPlanUtils::CalcOutputShape(
      nullptr, input_shape, filter_shape,
      strides_, padding_type_, paddings_,
      farmework_type, output_shape);
  ops::Deconv2dPartPlanUtils::CalcBiasShape(output_shape, bias_shape);

  // Create partition plan.
  ShowText("Create partition plan");
  ops::Deconv2dPartPlan *deconv2d_part_plan = new ops::Deconv2dPartPlan(
      part_dim, part_ratio, DataFormat::NCHW);
  ops::PartitionResult partition_result;
  do {
    partition_result = deconv2d_part_plan->Make(nullptr,
                                                input_shape,
                                                filter_shape,
                                                strides_,
                                                padding_type_,
                                                paddings_,
                                                farmework_type);
  } while (partition_result == ops::PartitionResult::PARTITION_REDO);

  if (partition_result == ops::PartitionResult::PARTITION_FAILED) {
    LOG(ERROR) << "Make deconv2d part plan failed";
    return -1;
  } else {
    deconv2d_part_plan->Show();
  }

  part_plan_.reset(deconv2d_part_plan);

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
  const DataType in_out_dt = DataType::DT_FLOAT;
  const DataType weights_dt = DataType::DT_FLOAT;

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
    filter_ = TensorUtils::CreateGPUImageTensor(
        dev_context, filter_shape, weights_dt, true,
        DataFormat::OIHW, OpenCLBufferType::CONV2D_FILTER,
        std::string("filter_image"));
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
    filter_ = TensorUtils::CreateBufferTensor(
        dev_context, filter_shape, weights_dt, true,
        DataFormat::OIHW, DeviceType::GPU,
        std::string("filter_buffer"), true);
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
      part_shape_filter_gpu = deconv2d_part_plan->gpu_filter_part_shape();
  const std::vector<index_t>
      part_shape_output_gpu = part_plan_->gpu_output_part_shape();
  const std::vector<index_t>
      part_shape_input_cpu  = part_plan_->cpu_input_part_shape();
  const std::vector<index_t>
      part_shape_filter_cpu = deconv2d_part_plan->cpu_filter_part_shape();
  const std::vector<index_t>
      part_shape_output_cpu = part_plan_->cpu_output_part_shape();

  tensor_manage_util_.reset(new TensorManageUtil(
      dev_context->GetCpuDevice()->allocator()));
  tensor_manage_util_->set_gpu_allocator(
      dev_context->GetGpuDevice()->allocator());

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
      part_length = part_shape_filter_gpu[O_OIHW];
      filter_gpu_ = tensor_manage_util_->CreateOrReshapePartTensor(
                      filter_, part_dim, 0, part_length, true);
      part_length = part_shape_output_gpu[in_out_dim_idx];
      output_gpu_ = tensor_manage_util_->CreateOrReshapePartTensor(
                      output_, part_dim, 0, part_length, true);
    }

    const DeviceType in_out_device_type = DeviceType::GPU;
    const DeviceType filter_device_type = DeviceType::GPU;
    bool in_out_mapping_hint = false;
    const bool weights_mapping_hint = true;

    input_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_input_cpu, DT_FLOAT, false,
        DataFormat::NCHW, in_out_device_type,
        std::string("input_cpu"), in_out_mapping_hint);
    filter_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_filter_cpu, DT_FLOAT, filter_->is_weight(),
        filter_->data_format(), filter_device_type,
        std::string("filter_cpu"), weights_mapping_hint);
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
  if (!part_plan_->IsGpuOnly()) {
    cpu_kernel_.reset(new Deconv2dCpuFloatKernel(
        filter_cpu_, strides_, paddings_, padding_type_, farmework_type,
        activation_, relux_max_limit_, leakyrelu_coefficient_));
  }

  ShowText("Create gpu computing kernel");
  if (!part_plan_->IsCpuOnly()) {
    const MemoryType opencl_mem_type
        = use_opencl_image ? MemoryType::GPU_IMAGE : MemoryType::GPU_BUFFER;
    opencl_kernel_ = CreateOpenCLDeconv2dKernel(opencl_mem_type);
  }

  // TODO(fucheng): Warming up.

  return 0;
}

int CodlDeconv2dCpuGpuTestTask::EnqueueInputDataTransformKernel(
    StatsFuture *future) {
  if (part_plan_->IsGpuOnly()) {
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

int CodlDeconv2dCpuGpuTestTask::EnqueueMapKernel(
    StatsFuture *future,
    StatsFuture *map_in_future) {
  if (part_plan_->IsGpuOnly()) {
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

int CodlDeconv2dCpuGpuTestTask::EnqueueGpuComputeKerenl(
    StatsFuture *future) {
  if (part_plan_->IsCpuOnly()) {
    return 0;
  }

  StatsFuture *old_future = gpu_context_->future();
  gpu_context_->set_future(future);

  if (!part_plan_->IsGpuOnly()) {
    opencl_kernel_->Compute(
        gpu_context_.get(),
        input_gpu_, filter_gpu_, bias_,
        strides_.data(), paddings_.data(),
        activation_, relux_max_limit_, leakyrelu_coefficient_,
        output_gpu_->shape(), output_gpu_);
  } else {
    opencl_kernel_->Compute(
        gpu_context_.get(),
        input_, filter_, bias_,
        strides_.data(), paddings_.data(),
        activation_, relux_max_limit_, leakyrelu_coefficient_,
        output_->shape(), output_);
  }

  gpu_context_->set_future(old_future);

  return 0;
}

int CodlDeconv2dCpuGpuTestTask::EnqueueUnmapKernel(
    cl::UserEvent **event,
    StatsFuture *unmap_in_future,
    StatsFuture *unmap_out_future) {
  if (part_plan_->IsGpuOnly()) {
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

int CodlDeconv2dCpuGpuTestTask::EnqueueOutputDataTransformKernel(
    StatsFuture *future) {
  if (part_plan_->IsGpuOnly()) {
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

int CodlDeconv2dCpuGpuTestTask::RunCpuComputeKernel() {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }

  cpu_kernel_->Compute(cpu_context_.get(),
                       input_cpu_,
                       filter_cpu_,
                       bias_cpu_,
                       output_cpu_);
  return 0;
}

}  // namespace mace
