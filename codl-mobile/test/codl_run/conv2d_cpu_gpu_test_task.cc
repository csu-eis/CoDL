
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#endif

#include "test/codl_run/core/test_param.h"
#include "test/codl_run/conv2d_test_param.h"
#include "test/codl_run/conv2d_test_task.h"

namespace mace {

int CodlConv2dCpuGpuTestTask::Prepare(TestParam *test_param) {
  Conv2dTestParam *conv2d_test_param =
      reinterpret_cast<Conv2dTestParam *>(test_param);

  // Load parameters for testing.
  const int num_threads = conv2d_test_param->num_threads;
  const CPUAffinityPolicy policy
      = static_cast<CPUAffinityPolicy>(conv2d_test_param->cpu_affinity_policy);
  is_debug_on_ = conv2d_test_param->is_debug_on;
  do_data_transform_ = conv2d_test_param->do_data_transform;
  do_compute_ = conv2d_test_param->do_compute;
  const bool do_warmup = conv2d_test_param->do_warmup;
  const MemoryType gpu_memory_type = conv2d_test_param->gpu_memory_type;
  const PartitionDim part_dim = static_cast<PartitionDim>(conv2d_test_param->part_dim);
  const float part_ratio = conv2d_test_param->part_ratio;
  compute_unit_hint_     = conv2d_test_param->compute_unit_hint;
  const DataType cpu_dtype = conv2d_test_param->cpu_dtype;
  const DataType gpu_dtype = conv2d_test_param->gpu_dtype;

  paddings_     = {0, 0};
  padding_type_ = static_cast<Padding>(static_cast<int>(VALID));
  dilations_    = {1, 1};
  activation_   = ops::ActivationType::NOOP;
  relux_max_limit_       = 0.0f;
  leakyrelu_coefficient_ = 0.0f;
  wino_block_size_       = conv2d_test_param->wino_block_size;
  
  const std::vector<index_t> input_shape = conv2d_test_param->input_shape;
  const std::vector<index_t> filter_shape = conv2d_test_param->filter_shape;
  strides_ = conv2d_test_param->strides;
  std::vector<index_t> output_shape;
  std::vector<index_t> bias_shape;
  ops::ConvPool2dPartPlanUtils::CalcConv2dOutputShape(
      input_shape, filter_shape, strides_,
      paddings_, padding_type_, output_shape);
  ops::ConvPool2dPartPlanUtils::CalcConv2dBiasShape(output_shape, bias_shape);

  // Create partition plan.
  ShowText("Create partition plan");
  DataFormat cpu_in_out_data_format = DataFormat::NCHW;
  DataFormat cpu_filter_data_format = DataFormat::OIHW;
  if (cpu_dtype == DataType::DT_UINT8) {
    do_data_transform_ = false;
    cpu_in_out_data_format = DataFormat::NHWC;
    cpu_filter_data_format = DataFormat::OHWI;
  }

  ops::ConvPool2dPartPlan *conv2d_part_plan = new ops::ConvPool2dPartPlan(
      part_dim, part_ratio, cpu_in_out_data_format, cpu_dtype);
  ops::PartitionResult partition_result;
  do {
    partition_result = conv2d_part_plan->Make(input_shape,
                                              filter_shape,
                                              strides_,
                                              dilations_,
                                              padding_type_,
                                              paddings_);
  } while (partition_result == ops::PartitionResult::PARTITION_REDO);

  if (partition_result == ops::PartitionResult::PARTITION_FAILED) {
    LOG(ERROR) << "Make conv2d part plan failed";
    return -1;
  }

#if 1
  conv2d_part_plan->Show();
#endif
  
  part_plan_.reset(conv2d_part_plan);

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
  MACE_CHECK(gpu_memory_type == MemoryType::GPU_BUFFER ||
             gpu_memory_type == MemoryType::GPU_IMAGE);

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
      part_shape_filter_gpu = conv2d_part_plan->gpu_filter_part_shape();
  const std::vector<index_t>
      part_shape_output_gpu = part_plan_->gpu_output_part_shape();
  const std::vector<index_t>
      part_shape_input_cpu = part_plan_->cpu_input_part_shape();
  const std::vector<index_t>
      part_shape_filter_cpu = conv2d_part_plan->cpu_filter_part_shape();
  const std::vector<index_t>
      part_shape_output_cpu = part_plan_->cpu_output_part_shape();

  tensor_manage_util_.reset(new TensorManageUtil(
      dev_context->GetCpuDevice()->allocator()));
  tensor_manage_util_->set_gpu_allocator(
      dev_context->GetGpuDevice()->allocator());
  OpenCLRuntime *opencl_runtime =
      dev_context->GetGpuDevice()->gpu_runtime()->opencl_runtime();

  tensor_manage_util_->Manage(input_);
  tensor_manage_util_->Manage(filter_);
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
        cpu_in_out_data_format, in_out_device_type,
        std::string("input_cpu"), in_out_mapping_hint);
    filter_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_filter_cpu, DT_FLOAT, filter_->is_weight(),
        cpu_filter_data_format, filter_device_type,
        std::string("filter_cpu"), weights_mapping_hint);
    output_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_output_cpu, DT_FLOAT, false,
        cpu_in_out_data_format, in_out_device_type,
        std::string("output_cpu"), in_out_mapping_hint);
    bias_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        bias_->shape(), DT_FLOAT, false,
        bias_->data_format(), in_out_device_type,
        std::string("bias_cpu"), weights_mapping_hint);
  }

  // Create computing kernel.
  ShowText("Create cpu computing kernel");
  if (cpu_dtype == DataType::DT_FLOAT) {
    conv2d_cpu_kernel_.reset(new Conv2dCpuFloatKernel(
        input_cpu_, filter_cpu_,
        strides_, paddings_, padding_type_, dilations_,
        activation_, relux_max_limit_, leakyrelu_coefficient_));
  } else if (cpu_dtype == DataType::DT_UINT8) {
    conv2d_cpu_kernel_.reset(new Conv2dCpuUint8Kernel(strides_, paddings_));
    input_cpu_->SetScale(0.5);
    filter_cpu_->SetScale(0.5);
    output_cpu_->SetScale(0.5);
  } else {
    LOG(ERROR) << "Unsupported cpu data type " << static_cast<int>(cpu_dtype);
    MACE_NOT_IMPLEMENTED;
  }

  ShowText("Create gpu computing kernel");
  const MemoryType opencl_mtype = use_opencl_image ? MemoryType::GPU_IMAGE
                                                   : MemoryType::GPU_BUFFER;
  opencl_kernel_ = CreateOpenCLConv2dKernel(opencl_mtype);

  if (do_warmup) {
    // Warm up.
    cl::CommandQueue &queue = opencl_runtime->command_queue();

    ShowText("Warm up computing kernel");
    if (!part_plan_->IsCpuOnly()) {
      ShowText("Warm up GPU kernel");
      StatsFuture future;
      gpu_context_->set_future(&future);
      
      int64_t t0 = NowMicros();
      if (part_plan_->IsGpuOnly()) {
        opencl_kernel_->Compute(
            gpu_context_.get(), input_, filter_, bias_,
            strides_.data(), padding_type_, paddings_,
            dilations_.data(), activation_,
            relux_max_limit_, leakyrelu_coefficient_,
            wino_block_size_, output_);
      } else {
        opencl_kernel_->Compute(
            gpu_context_.get(), input_gpu_, filter_gpu_, bias_,
            strides_.data(), padding_type_, paddings_,
            dilations_.data(), activation_,
            relux_max_limit_, leakyrelu_coefficient_,
            wino_block_size_, output_gpu_);
      }
      CallStats call_stats;
      future.wait_fn(&call_stats);
      int64_t t1 = NowMicros();
      VLOG(1) << "Warming up OpenCL kernel takes "
              << ((t1 - t0) / 1000.0) << " ms"
              << " event "
              << (call_stats.end_micros - call_stats.start_micros) / 1000.0 << " ms";
      
      queue.finish();
      gpu_context_->set_future(nullptr);
    }

    if (!part_plan_->IsGpuOnly()) {
      ShowText("Warm up CPU kernel");
      const OdimRanges *input_odim_ranges = part_plan_->input_odim_ranges();
      const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
      ops::OpenCLPartBufferTransformer input_transformer(MemoryType::GPU_IMAGE,
                                                         MemoryType::GPU_BUFFER);
      ShowText("Enqueue transforming input kernel");
      input_transformer.Transform(gpu_context_.get(),
                                  input_,
                                  OpenCLBufferType::IN_OUT_CHANNEL,
                                  MemoryType::GPU_BUFFER,
                                  wino_block_size_,
                                  *input_odim_ranges,
                                  input_cpu_);
      ShowText("Enqueue mapping input and output kernel");
      input_cpu_->MapBuffer(AllocatorMapType::AMT_READ_ONLY, BlockFlag::BF_TRUE);
      output_cpu_->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY, BlockFlag::BF_TRUE);
      MACE_CHECK(input_cpu_->IsBufferMapped());
      MACE_CHECK(output_cpu_->IsBufferMapped());
      MACE_CHECK(filter_cpu_->IsBufferMapped());
      MACE_CHECK(bias_cpu_->IsBufferMapped());
      int64_t t0 = NowMicros();
      ShowText("Run CPU kernel");
      conv2d_cpu_kernel_->Compute(cpu_context_.get(),
                                  input_cpu_,
                                  filter_cpu_,
                                  bias_cpu_,
                                  output_cpu_);
      int64_t t1 = NowMicros();
      VLOG(1) << "Warming up CPU kernel takes " << (t1 - t0) / 1000.0 << " ms";
      ShowText("Enqueue unmapping input and output kernel");
      input_cpu_->UnmapBuffer();
      output_cpu_->UnmapBuffer();
      //queue.finish();
      ShowText("Enqueue transforming output kernel");
      ops::OpenCLPartBufferTransformer output_transformer(MemoryType::GPU_BUFFER,
                                                          MemoryType::GPU_IMAGE);
      output_transformer.Transform(gpu_context_.get(),
                                   output_cpu_,
                                   OpenCLBufferType::IN_OUT_CHANNEL,
                                   MemoryType::GPU_IMAGE,
                                   wino_block_size_,
                                   *output_odim_ranges,
                                   output_);
    }

    queue.finish();
  }

  return 1;
}

void CodlConv2dCpuGpuTestTask::UpdatePartTensors() {
  const ops::ConvPool2dPartPlan *conv2d_part_plan
      = reinterpret_cast<const ops::ConvPool2dPartPlan *>(
          part_plan_.get());

  const std::vector<index_t>
      part_shape_input_gpu = part_plan_->gpu_input_part_shape();
  const std::vector<index_t>
      part_shape_filter_gpu = conv2d_part_plan->gpu_filter_part_shape();
  const std::vector<index_t>
      part_shape_output_gpu = part_plan_->gpu_output_part_shape();
  const std::vector<index_t>
      part_shape_input_cpu = part_plan_->cpu_input_part_shape();
  const std::vector<index_t>
      part_shape_filter_cpu = conv2d_part_plan->cpu_filter_part_shape();
  const std::vector<index_t>
      part_shape_output_cpu = part_plan_->cpu_output_part_shape();

  if (!part_plan_->IsGpuOnly()) {
    if (!part_plan_->IsCpuOnly()) {
      const PartitionDim part_dim = part_plan_->dimension();
      index_t in_out_dim_idx = H_NHWC;
      index_t part_length;
      if (part_dim == DIM_INPUT_HEIGHT) {
        in_out_dim_idx = H_NHWC;
      } else if (part_dim == DIM_OUTPUT_CHANNEL) {
        in_out_dim_idx = C_NHWC;
      } else {
        MACE_NOT_IMPLEMENTED;
      }
      VLOG(1) << "Update GPU tensors";
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

    VLOG(1) << "Update CPU tensors";
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
}

#if 0
int CodlConv2dCpuGpuTestTask::RunCpu(
    mace::DurationCollector<double> *dura_collector) {
  if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
    int64_t t0;
    double wait_map_out_time = 0;
    double cpu_compute_time = 0;
    double wait_unmap_in_time = 0;
    double wait_transform_out_time = 0;
    StatsFuture future;
    StatsFuture transform_in_future;
    StatsFuture map_in_future;
    StatsFuture map_out_future;
    StatsFuture unmap_in_future;
    StatsFuture unmap_out_future;
    StatsFuture transform_out_future;
    cl::UserEvent *unmap_in_user_event = nullptr;
    
    OpenCLRuntime *opencl_runtime =
        GetDeviceContext()->GetGpuDevice()->gpu_runtime()->opencl_runtime();
    OpenCLEventManager *event_manager = opencl_runtime->event_manager();

    if (do_data_transform_) {
      gpu_context_->set_future(&future);
      const OdimRanges *input_odim_ranges = part_plan_->input_odim_ranges();
      ops::OpenCLPartBufferTransformer input_transformer(MemoryType::GPU_IMAGE,
                                                         MemoryType::GPU_BUFFER);
      input_transformer.Transform(gpu_context_.get(),
                                  input_.get(),
                                  OpenCLBufferType::IN_OUT_CHANNEL,
                                  MemoryType::GPU_BUFFER,
                                  wino_block_size_,
                                  *input_odim_ranges,
                                  input_cpu_);
      transform_in_future.wait_fn = future.wait_fn;
    }

    input_cpu_->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                          BlockFlag::BF_FALSE,
                          &future);
    map_in_future.wait_fn = future.wait_fn;
    output_cpu_->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                           BlockFlag::BF_FALSE,
                           &future);
    map_out_future.wait_fn = future.wait_fn;

    t0 = NowMicros();
    map_out_future.wait_fn(nullptr);
    wait_map_out_time = (NowMicros() - t0) / 1000.0;

    // Create a new user event.
    unmap_in_user_event = event_manager->CreateSingleUserEvent(
        opencl_runtime->context(),
        EventActionType::WAIT,
        EventOpType::TRANSFORM_OUTPUT);
    input_cpu_->UnmapBuffer(&future);
    unmap_in_future.wait_fn = future.wait_fn;
    event_manager->InsertNullEvent(EventActionType::WAIT);
    output_cpu_->UnmapBuffer(&future);
    unmap_out_future.wait_fn = future.wait_fn;

    if (do_data_transform_) {
      gpu_context_->set_future(&future);
      const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
      ops::OpenCLPartBufferTransformer output_transformer(MemoryType::GPU_BUFFER,
                                                          MemoryType::GPU_IMAGE);
      output_transformer.Transform(gpu_context_.get(),
                                   output_cpu_,
                                   OpenCLBufferType::IN_OUT_CHANNEL,
                                   MemoryType::GPU_IMAGE,
                                   wino_block_size_,
                                   *output_odim_ranges,
                                   output_.get());
      transform_out_future.wait_fn = future.wait_fn;
    }

    if (!do_data_transform_) {
      cpu_context_->set_dura_collector(dura_collector);
    } else {
      cpu_context_->set_dura_collector(nullptr);
    }
    t0 = NowMicros();
    if (do_compute_) {
      conv2d_cpu_kernel_->Compute(cpu_context_.get(),
                                  input_cpu_,
                                  filter_cpu_,
                                  bias_cpu_,
                                  output_cpu_);
    }
    cpu_compute_time = (NowMicros() - t0) / 1000.0;
    cpu_context_->set_dura_collector(nullptr);

    if (unmap_in_user_event != nullptr) {
      event_manager->SetUserEventComplete(unmap_in_user_event);
    }

    if (do_data_transform_) {
      t0 = NowMicros();
      transform_out_future.wait_fn(nullptr);
      wait_transform_out_time = (NowMicros() - t0) / 1000.0;
      gpu_context_->set_future(nullptr);
    }

    if (is_debug_on_) {
      const double tranform_in_time = FutureToMillis(&transform_in_future);
      const double map_in_time = FutureToMillis(&map_in_future);
      const double map_out_time = FutureToMillis(&map_out_future);
      const double unmap_in_time = FutureToMillis(&unmap_in_future);
      const double unmap_out_time = FutureToMillis(&unmap_out_future);
      const double transform_out_time = FutureToMillis(&transform_out_future);

      LOG(INFO) << "Latency:"
                << " tr_in " << tranform_in_time << " ms"
                << " map_in " << map_in_time << " ms"
                << " map_out " << map_out_time << " ms"
                << " wait_map_out " << wait_map_out_time << " ms"
                << " c_cpu " << cpu_compute_time << " ms"
                << " wait_unmap_in " << wait_unmap_in_time << " ms"
                << " unmap_in " << unmap_in_time << " ms"
                << " unmap_out " << unmap_out_time << " ms"
                << " tr_out " << transform_out_time << " ms"
                << " wait_tr_out " << wait_transform_out_time << " ms";
    }

    if (do_data_transform_ && dura_collector != nullptr) {
      // Data transfoming related latency.
      std::vector<double> dt_latencies;
      dt_latencies.push_back(FutureToMillis(&transform_in_future));  // 0
      dt_latencies.push_back(FutureToMillis(&map_in_future));        // 1
      dt_latencies.push_back(FutureToMillis(&map_out_future));       // 2
      dt_latencies.push_back(wait_map_out_time
          - dt_latencies[0] - dt_latencies[1] - dt_latencies[2]);    // 3
      dt_latencies.push_back(FutureToMillis(&unmap_in_future));      // 4
      dt_latencies.push_back(FutureToMillis(&unmap_out_future));     // 5
      dt_latencies.push_back(FutureToMillis(&transform_out_future)); // 6
      dt_latencies.push_back(wait_unmap_in_time - dt_latencies[4]);  // 7
      dura_collector->Add(dt_latencies);
    }
  }
  
  return 1;
}

int CodlConv2dCpuGpuTestTask::RunCpuGpu(
    mace::DurationCollector<double> *dura_collector) {
  int64_t t0;
  double enqueued_gpu_time;
  double wait_map_out_time;
  double cpu_compute_time;
  double wait_transform_out_time;
  StatsFuture future;
  StatsFuture transform_in_future;
  StatsFuture map_in_future;
  StatsFuture map_out_future;
  StatsFuture gpu_compute_future;
  StatsFuture unmap_in_future;
  StatsFuture unmap_out_future;
  StatsFuture transform_out_future;

  gpu_context_->set_future(&future);

  OpenCLRuntime *opencl_runtime =
      GetDeviceContext()->GetGpuDevice()->gpu_runtime()->opencl_runtime();
  OpenCLEventManager *event_manager = opencl_runtime->event_manager();

  t0 = NowMicros();
  if (do_data_transform_) {
    if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
      // Enqueue input transforming kernel.
      const OdimRanges *input_odim_ranges = part_plan_->input_odim_ranges();
      ops::OpenCLPartBufferTransformer input_transformer(MemoryType::GPU_IMAGE,
                                                         MemoryType::GPU_BUFFER);
      input_transformer.Transform(gpu_context_.get(),
                                  input_.get(),
                                  OpenCLBufferType::IN_OUT_CHANNEL,
                                  MemoryType::GPU_BUFFER,
                                  wino_block_size_,
                                  *input_odim_ranges,
                                  input_cpu_);
      transform_in_future.wait_fn = future.wait_fn;
    }
  }

  if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
    // Enqueue input/output mapping kernels.
    input_cpu_->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                          BlockFlag::BF_FALSE,
                          &future);
    map_in_future.wait_fn = future.wait_fn;
    output_cpu_->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                           BlockFlag::BF_FALSE,
                           &future);
    map_out_future.wait_fn = future.wait_fn;
  }

  if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_GPU, compute_unit_hint_)) {
    // Enqueue GPU computation kernel.
    opencl_kernel_->Compute(
        gpu_context_.get(), input_gpu_, filter_gpu_, bias_.get(),
        strides_.data(), padding_type_, paddings_,
        dilations_.data(), activation_,
        relux_max_limit_, leakyrelu_coefficient_,
        wino_block_size_, output_gpu_);
    gpu_compute_future.wait_fn = future.wait_fn;
  }

  cl::UserEvent *trans_out_user_event = nullptr;
  if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
    // Create a new user event.
    trans_out_user_event
        = event_manager->CreateSingleUserEvent(
            opencl_runtime->context(),
            EventActionType::WAIT,
            EventOpType::TRANSFORM_OUTPUT);
    // Enqueue input/output unmapping kernels.
    input_cpu_->UnmapBuffer(&future);
    unmap_in_future.wait_fn = future.wait_fn;
    event_manager->InsertNullEvent(EventActionType::WAIT);
    output_cpu_->UnmapBuffer(&future);
    unmap_out_future.wait_fn = future.wait_fn;
  }

  if (do_data_transform_) {
    if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
      // Enqueue output transforming kernel.
      const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
      ops::OpenCLPartBufferTransformer output_transformer(MemoryType::GPU_BUFFER,
                                                          MemoryType::GPU_IMAGE);
      output_transformer.Transform(gpu_context_.get(),
                                   output_cpu_,
                                   OpenCLBufferType::IN_OUT_CHANNEL,
                                   MemoryType::GPU_IMAGE,
                                   wino_block_size_,
                                   *output_odim_ranges,
                                   output_.get());
      transform_out_future.wait_fn = future.wait_fn;
    }
  }
  enqueued_gpu_time = (NowMicros() - t0) / 1000.0;

  t0 = NowMicros();
  // Synchronize for output mapping.
  map_out_future.wait_fn(nullptr);
  wait_map_out_time = (NowMicros() - t0) / 1000.0;

  if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
    cpu_context_->set_dura_collector(dura_collector);
    t0 = NowMicros();
    // Run CPU computation kernel.
    conv2d_cpu_kernel_->Compute(cpu_context_.get(),
                                input_cpu_,
                                filter_cpu_,
                                bias_cpu_,
                                output_cpu_);
    cpu_compute_time = (NowMicros() - t0) / 1000.0;
    cpu_context_->set_dura_collector(nullptr);
  }

  if (trans_out_user_event != nullptr) {
    // Synchronize for input unmapping.
    event_manager->SetUserEventComplete(trans_out_user_event);
  }

  t0 = NowMicros();
  future.wait_fn(nullptr);
  wait_transform_out_time = (NowMicros() - t0) / 1000.0;
  
  gpu_context_->set_future(nullptr);

  if (is_debug_on_) {
    const double tranform_in_time = FutureToMillis(&transform_in_future);
    const double map_in_time = FutureToMillis(&map_in_future);
    const double map_out_time = FutureToMillis(&map_out_future);
    const double gpu_compute_time = FutureToMillis(&gpu_compute_future);
    const double unmap_in_time = FutureToMillis(&unmap_in_future);
    const double unmap_out_time = FutureToMillis(&unmap_out_future);
    const double transform_out_time = FutureToMillis(&transform_out_future);

    LOG(INFO) << "Latency:"
              << " tr_in " << tranform_in_time << " ms"
              << " map_in " << map_in_time << " ms"
              << " map_out " << map_out_time << " ms"
              << " tr_out " << transform_out_time << " ms"
              << " c_cpu " << cpu_compute_time << " ms"
              << " c_gpu " << gpu_compute_time << " ms";
    LOG(INFO) << "CPU-side Latency:"
              << " enq " << enqueued_gpu_time << " ms"
              << " wait_map_out " << wait_map_out_time << " ms"
              << " c_cpu " << cpu_compute_time << " ms"
              << " wait_tr_out " << wait_transform_out_time << " ms";
    LOG(INFO) << "GPU-side Latency:"
              << " tr_in " << tranform_in_time << " ms"
              << " map_in " << map_in_time << " ms"
              << " map_out " << map_out_time << " ms"
              << " c_gpu " << gpu_compute_time << " ms"
              << " unmap_in " << unmap_in_time << " ms"
              << " unmap_out " << unmap_out_time << " ms"
              << " tr_out " << transform_out_time << " ms";
  }

  return 1;
}

int CodlConv2dCpuGpuTestTask::Run(
    mace::DurationCollector<double> *dura_collector) {
  if (part_plan_->IsCpuGpu()) {
    RunCpuGpu(dura_collector);
  } else if (part_plan_->IsGpuOnly()) {
    RunGpu(dura_collector);
  } else if (part_plan_->IsCpuOnly()) {
    RunCpu(dura_collector);
  } else {
    LOG(ERROR) << "Partition plan error.";
    return -1;
  }

  return 1;
}
#endif

int CodlConv2dCpuGpuTestTask::EnqueueInputDataTransformKernel(
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
                              wino_block_size_,
                              *input_odim_ranges,
                              input_cpu_);
  
  gpu_context_->set_future(old_future);

  return 0;
}

int CodlConv2dCpuGpuTestTask::EnqueueMapKernel(
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

int CodlConv2dCpuGpuTestTask::EnqueueGpuComputeKerenl(StatsFuture *future) {
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

  if (!part_plan_->IsGpuOnly()) {
    opencl_kernel_->Compute(
        gpu_context_.get(), input_gpu_, filter_gpu_, bias_,
        strides_.data(), padding_type_, paddings_,
        dilations_.data(), activation_,
        relux_max_limit_, leakyrelu_coefficient_,
        wino_block_size_, output_gpu_);
  } else {
    opencl_kernel_->Compute(
        gpu_context_.get(),
        input_, filter_, bias_,
        strides_.data(), padding_type_, paddings_,
        dilations_.data(), activation_,
        relux_max_limit_, leakyrelu_coefficient_,
        wino_block_size_, output_);
  }

  gpu_context_->set_future(old_future);

  return 0;
}

int CodlConv2dCpuGpuTestTask::EnqueueUnmapKernel(
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

int CodlConv2dCpuGpuTestTask::EnqueueOutputDataTransformKernel(
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
                               wino_block_size_,
                               *output_odim_ranges,
                               output_);
  
  gpu_context_->set_future(old_future);

  return 0;
}

int CodlConv2dCpuGpuTestTask::RunCpuComputeKernel() {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }
  if (!do_compute_) {
    return 0;
  }

#if 0
  LOG(INFO) << "RunCpuComputeKernel";
#endif

  conv2d_cpu_kernel_->Compute(cpu_context_.get(),
                              input_cpu_,
                              filter_cpu_,
                              bias_cpu_,
                              output_cpu_);
  return 0;
}

}  // namespace mace
