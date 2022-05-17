
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#endif

#include "test/codl_run/pooling_test_param.h"
#include "test/codl_run/pooling_test_task.h"

namespace mace {

int CodlPoolingCpuGpuTestTask::Prepare(TestParam *test_param) {
  PoolingTestParam *pooling_test_param =
      reinterpret_cast<PoolingTestParam *>(test_param);

  // Load parameters for testing.
  const int num_threads = pooling_test_param->num_threads;
  const CPUAffinityPolicy policy
      = static_cast<CPUAffinityPolicy>(pooling_test_param->cpu_affinity_policy);
  is_debug_on_ = pooling_test_param->is_debug_on;
  do_data_transform_ = pooling_test_param->do_data_transform;
  do_compute_ = pooling_test_param->do_compute;
  const MemoryType gpu_memory_type = pooling_test_param->gpu_memory_type;
  const PartitionDim part_dim = static_cast<PartitionDim>(pooling_test_param->part_dim);
  const float part_ratio = pooling_test_param->part_ratio;
  compute_unit_hint_     = pooling_test_param->compute_unit_hint;
  const DataType cpu_dtype = pooling_test_param->cpu_dtype;
  const DataType gpu_dtype = pooling_test_param->gpu_dtype;

  pooling_type_ = static_cast<PoolingType>(pooling_test_param->pooling_type);
  padding_type_ = static_cast<Padding>(static_cast<int>(VALID));
  paddings_     = {0, 0};
  dilations_    = {1, 1};
  round_type_   = RoundType::CEIL;
  
  const std::vector<index_t> input_shape = pooling_test_param->input_shape;
  std::vector<index_t> filter_shape = pooling_test_param->filter_shape;
  if (filter_shape.size() == 2) {
    filter_shape = {input_shape[3], input_shape[3], filter_shape[0], filter_shape[1]};
  }
  MACE_CHECK(filter_shape.size() == 2 || filter_shape.size() == 4);
  kernels_ = std::vector<int>(2);
  kernels_[0] = static_cast<int>(filter_shape[2]);
  kernels_[1] = static_cast<int>(filter_shape[3]);
  strides_ = pooling_test_param->strides;
  
  std::vector<index_t> output_shape;
  ops::ConvPool2dPartPlanUtils::CalcConv2dOutputShape(
      input_shape, filter_shape, strides_,
      paddings_, padding_type_, output_shape);

#if 0
  LOG(INFO) << "input_shape " << VectorToString<index_t>(input_shape)
            << ", output_shape " << VectorToString<index_t>(output_shape);
#endif

  // Create partition plan.
  ShowText("Create partition plan");
  DataFormat cpu_in_out_data_format = DataFormat::NCHW;
  DataFormat cpu_filter_data_format = DataFormat::OIHW;
  if (cpu_dtype == DataType::DT_UINT8) {
    do_data_transform_ = false;
    cpu_in_out_data_format = DataFormat::NHWC;
    cpu_filter_data_format = DataFormat::OHWI;
  }
  
  ops::ConvPool2dPartPlan *pooling_part_plan = new ops::ConvPool2dPartPlan(
      part_dim, part_ratio, cpu_in_out_data_format, cpu_dtype);
  ops::PartitionResult partition_result;
  do {
    partition_result = pooling_part_plan->Make(input_shape,
                                               filter_shape,
                                               strides_,
                                               dilations_,
                                               padding_type_,
                                               paddings_);
  } while (partition_result == ops::PartitionResult::PARTITION_REDO);

  if (partition_result == ops::PartitionResult::PARTITION_FAILED) {
    LOG(ERROR) << "Make pooling part plan failed";
    return -1;
  }
  
#if 1
  pooling_part_plan->Show();
#endif

  part_plan_.reset(pooling_part_plan);

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
    output_ = TensorUtils::CreateGPUImageTensor(
        dev_context, output_shape, in_out_dt, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("output_image"));
  } else {
    // Create GPU (buffer) tensors.
    input_ = TensorUtils::CreateBufferTensor(
        dev_context, input_shape, in_out_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("input_buffer"), true);
    output_ = TensorUtils::CreateBufferTensor(
        dev_context, output_shape, in_out_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("output_buffer"), true);
  }

  const std::vector<index_t>
      part_shape_input_gpu = part_plan_->gpu_input_part_shape();
  const std::vector<index_t>
      part_shape_output_gpu = part_plan_->gpu_output_part_shape();
  const std::vector<index_t>
      part_shape_input_cpu  = part_plan_->cpu_input_part_shape();
  const std::vector<index_t>
      part_shape_output_cpu = part_plan_->cpu_output_part_shape();

  tensor_manage_util_.reset(new TensorManageUtil(
      dev_context->GetCpuDevice()->allocator()));
  tensor_manage_util_->set_gpu_allocator(
      dev_context->GetGpuDevice()->allocator());

  tensor_manage_util_->Manage(input_);
  tensor_manage_util_->Manage(output_);

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
      part_length = part_shape_output_gpu[in_out_dim_idx];
      output_gpu_ = tensor_manage_util_->CreateOrReshapePartTensor(
                      output_, part_dim, 0, part_length, true);
    }

    const DeviceType in_out_device_type = DeviceType::GPU;
    bool in_out_mapping_hint = false;

    input_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_input_cpu, DT_FLOAT, false,
        DataFormat::NCHW, in_out_device_type,
        std::string("input_cpu"), in_out_mapping_hint);
    output_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_output_cpu, DT_FLOAT, false,
        DataFormat::NCHW, in_out_device_type,
        std::string("output_cpu"), in_out_mapping_hint);
  }

  // Create computing kernel.
  ShowText("Create cpu computing kernel");
  if (!part_plan_->IsGpuOnly()) {
    if (cpu_dtype == DataType::DT_FLOAT) {
      cpu_kernel_.reset(new PoolingCpuFloatKernel(
          kernels_, strides_, dilations_, pooling_type_));
    } else if (cpu_dtype == DataType::DT_UINT8) {
      cpu_kernel_.reset(new PoolingCpuUint8Kernel(
          kernels_, strides_, pooling_type_));
      input_cpu_->SetScale(0.5);
      output_cpu_->SetScale(0.5);
    } else {
      LOG(ERROR) << "Unsupported cpu data type " << static_cast<int>(cpu_dtype);
      MACE_NOT_IMPLEMENTED;
    }
  }

  ShowText("Create gpu computing kernel");
  if (!part_plan_->IsCpuOnly()) {
    const MemoryType opencl_mem_type
        = use_opencl_image ? MemoryType::GPU_IMAGE : MemoryType::GPU_BUFFER;
    opencl_kernel_ = CreateOpenCLPoolingKernel(opencl_mem_type);
  }

  // TODO(fucheng): Warming up.

  return 0;
}

void CodlPoolingCpuGpuTestTask::UpdatePartTensors() {
  const ops::ConvPool2dPartPlan *part_plan
      = reinterpret_cast<const ops::ConvPool2dPartPlan *>(part_plan_.get());

  const std::vector<index_t>
      part_shape_input_gpu = part_plan->gpu_input_part_shape();
  const std::vector<index_t>
      part_shape_output_gpu = part_plan->gpu_output_part_shape();
  const std::vector<index_t>
      part_shape_input_cpu = part_plan->cpu_input_part_shape();
  const std::vector<index_t>
      part_shape_output_cpu = part_plan->cpu_output_part_shape();

  if (!part_plan->IsGpuOnly()) {
    if (!part_plan->IsCpuOnly()) {
      const PartitionDim part_dim = part_plan->dimension();
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
      part_length = part_shape_output_gpu[in_out_dim_idx];
      output_gpu_ = tensor_manage_util_->CreateOrReshapePartTensor(
                      output_, part_dim, 0, part_length, true);
    }

    const DeviceType in_out_device_type = DeviceType::GPU;
    bool in_out_mapping_hint = false;

    input_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_input_cpu, DT_FLOAT, false,
        DataFormat::NCHW, in_out_device_type,
        std::string("input_cpu"), in_out_mapping_hint);
    output_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_output_cpu, DT_FLOAT, false,
        DataFormat::NCHW, in_out_device_type,
        std::string("output_cpu"), in_out_mapping_hint);
  }
}

int CodlPoolingCpuGpuTestTask::EnqueueInputDataTransformKernel(
    StatsFuture *future) {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_data_transform_) {
    return 0;
  }

#if 0
  LOG(INFO) << "EnqueueInputDataTransformKernel";
#endif

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

int CodlPoolingCpuGpuTestTask::EnqueueMapKernel(
    StatsFuture *future,
    StatsFuture *map_in_future) {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_compute_) {
    return 0;
  }

#if 0
  LOG(INFO) << "EnqueueMapKernel";
#endif

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

int CodlPoolingCpuGpuTestTask::EnqueueGpuComputeKerenl(
    StatsFuture *future) {
  if (part_plan_->IsCpuOnly()) {
    return 0;
  }
  if (!do_compute_) {
    return 0;
  }

#if 0
  LOG(INFO) << "EnqueueGpuComputeKerenl";
#endif

  StatsFuture *old_future = gpu_context_->future();
  gpu_context_->set_future(future);

  Tensor *input_tensor, *output_tensor;
  if (!part_plan_->IsGpuOnly()) {
    input_tensor = input_gpu_;
    output_tensor = output_gpu_;
  } else {
    input_tensor = input_;
    output_tensor = output_;
  }

  opencl_kernel_->Compute(gpu_context_.get(),
                          input_tensor,
                          pooling_type_,
                          kernels_.data(),
                          strides_.data(),
                          padding_type_,
                          paddings_,
                          dilations_.data(),
                          round_type_,
                          output_tensor);

  gpu_context_->set_future(old_future);

  return 0;
}

int CodlPoolingCpuGpuTestTask::EnqueueUnmapKernel(
    cl::UserEvent **event,
    StatsFuture *unmap_in_future,
    StatsFuture *unmap_out_future) {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_compute_) {
    return 0;
  }

#if 0
  LOG(INFO) << "EnqueueUnmapKernel";
#endif

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

int CodlPoolingCpuGpuTestTask::EnqueueOutputDataTransformKernel(
    StatsFuture *future) {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_data_transform_) {
    return 0;
  }

#if 0
  LOG(INFO) << "EnqueueOutputDataTransformKernel";
#endif

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

int CodlPoolingCpuGpuTestTask::RunCpuComputeKernel() {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_compute_) {
    return 0;
  }

#if 0
  LOG(INFO) << "RunCpuComputeKernel";
#endif

  cpu_kernel_->Compute(cpu_context_.get(),
                       input_cpu_,
                       output_cpu_);
  return 0;
}

}  // namespace mace
