
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/transpose_image.h"
#endif

#include "test/fucheng/test_param.h"
#include "test/fucheng/conv2d_test_param.h"
#include "test/fucheng/conv2d_test_task.h"

constexpr int kMaxGpuEnqueuedKernels = 0;

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
  const int gpu_memory_type = conv2d_test_param->gpu_memory_type;
  const PartitionDim part_dim = static_cast<PartitionDim>(conv2d_test_param->part_dim);
  const float part_ratio = conv2d_test_param->part_ratio;
  padding_type_ = static_cast<Padding>(static_cast<int>(VALID));
  dilations_    = {1, 1};
  activation_   = ops::ActivationType::NOOP;
  relux_max_limit_       = 0.0f;
  leakyrelu_coefficient_ = 0.0f;
  wino_block_size_       = conv2d_test_param->wino_block_size;
  compute_unit_hint_     = conv2d_test_param->compute_unit_hint;
  
  const std::vector<index_t> input_shape = conv2d_test_param->input_shape;
  const std::vector<index_t> filter_shape = conv2d_test_param->filter_shape;
  strides_ = conv2d_test_param->strides;
  const std::vector<index_t> output_shape
      = ops::Conv2dPartPlanUtils::CalcConv2dOutputShape(
          input_shape, filter_shape, strides_, paddings_, padding_type_);
  const std::vector<index_t> bias_shape
      = ops::Conv2dPartPlanUtils::CalcConv2dBiasShape(output_shape);

  // Create partition plan.
  part_plan_.reset(new ops::Conv2dPartPlan(part_dim,
                                           part_ratio,
                                           DataFormat::NCHW));
  ops::PartitionResult partition_result;
  do {
    partition_result = part_plan_->Make(input_shape,
                                        filter_shape,
                                        strides_,
                                        dilations_,
                                        padding_type_,
                                        paddings_);
  } while (partition_result == ops::PartitionResult::PARTITION_REDO);

  if (partition_result == ops::PartitionResult::PARTITION_FAILED) {
    LOG(ERROR) << "Make conv2d part plan failed";
    return -1;
  } else {
    part_plan_->Show();
  }

  // Initialize device context.
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

  // Initialize data template.
  const std::vector<float> input_data_template{1.0f};
  const std::vector<float> filter_data_template{1.0f};
  const std::vector<float> bias_data_template{1.0f};

  MACE_CHECK(input_shape[3] % input_data_template.size() == 0);
  MACE_CHECK(filter_shape[1] % filter_data_template.size() == 0);
  MACE_CHECK(bias_shape[0] % bias_data_template.size() == 0);
  MACE_CHECK(input_data_template.size() == filter_data_template.size());
  MACE_CHECK(input_data_template.size() == bias_data_template.size());
  
  // Compute the value of result.
  float single_result_value = 0.0f;
  for (size_t i = 0; i < input_data_template.size(); ++i) {
    single_result_value += (input_data_template[i] * filter_data_template[i]);
  }
  single_result_value *= (input_shape[3] / input_data_template.size());
  single_result_value *= (filter_shape[2] * filter_shape[3]);
  for (size_t i = 0; i < input_data_template.size(); ++i) {
    single_result_value += bias_data_template[i];
  }

  const bool use_opencl_image = (gpu_memory_type == 0);
  const DataType in_out_dt = DataType::DT_FLOAT;
  const DataType weights_dt = DataType::DT_FLOAT;

  if (!use_opencl_image) {
    // No data transforming for gpu buffer.
    do_data_transform_ = false;
  }

  if (use_opencl_image) {
    // Create GPU (image) full tensors.
    input_.reset(TensorUtils::CreateGPUImageTensor(
        dev_context, input_shape, in_out_dt, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("input_image")));
    filter_.reset(TensorUtils::CreateGPUImageTensor(
        dev_context, filter_shape, weights_dt, true,
        DataFormat::OIHW, OpenCLBufferType::CONV2D_FILTER,
        std::string("filter_image")));
    output_.reset(TensorUtils::CreateGPUImageTensor(
        dev_context, output_shape, in_out_dt, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("output_image")));
    bias_.reset(TensorUtils::CreateGPUImageTensor(
        dev_context, bias_shape, weights_dt, false,
        DataFormat::NONE, OpenCLBufferType::ARGUMENT,
        std::string("bias_image")));

    const bool do_fill_tensor_data = true;
    if (do_fill_tensor_data) {
      // Fill tensor data.
      LOG(INFO) << "Fill and check tensor image data ...";
      TensorDataFiller data_filler;
      data_filler.FillImage(dev_context, gpu_context_.get(), input_.get(),
                            OpenCLBufferType::IN_OUT_CHANNEL,
                            wino_block_size_,
                            input_data_template);
      data_filler.FillImage(dev_context, gpu_context_.get(), filter_.get(),
                            OpenCLBufferType::CONV2D_FILTER,
                            wino_block_size_,
                            filter_data_template);
      data_filler.FillImage(dev_context, gpu_context_.get(), bias_.get(),
                            OpenCLBufferType::ARGUMENT,
                            wino_block_size_,
                            bias_data_template);

      // Check filled tensor data.
      bool is_equal = false;
      TensorDataChecker data_checker;
      is_equal = data_checker.Equal(dev_context, gpu_context_.get(), input_.get(),
                                    OpenCLBufferType::IN_OUT_CHANNEL,
                                    wino_block_size_,
                                    input_data_template[0]);
      MACE_CHECK(is_equal, "input data is wrong");
      is_equal = data_checker.Equal(dev_context, gpu_context_.get(), filter_.get(),
                                    OpenCLBufferType::CONV2D_FILTER,
                                    wino_block_size_,
                                    filter_data_template[0]);
      MACE_CHECK(is_equal, "filter data is wrong");
      is_equal = data_checker.Equal(dev_context, gpu_context_.get(), bias_.get(),
                                    OpenCLBufferType::ARGUMENT,
                                    wino_block_size_,
                                    bias_data_template[0]);
      MACE_CHECK(is_equal, "bias data is wrong");
    }
  } else {
    // Create GPU (buffer) tensors.
    input_.reset(TensorUtils::CreateBufferTensor(
        dev_context, input_shape, in_out_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("input_buffer"), true));
    filter_.reset(TensorUtils::CreateBufferTensor(
        dev_context, filter_shape, weights_dt, true,
        DataFormat::OIHW, DeviceType::GPU,
        std::string("filter_buffer"), true));
    output_.reset(TensorUtils::CreateBufferTensor(
        dev_context, output_shape, in_out_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("output_buffer"), true));
    bias_.reset(TensorUtils::CreateBufferTensor(
        dev_context, bias_shape, weights_dt, false,
        DataFormat::NONE, DeviceType::GPU,
        std::string("bias_buffer"), true));
  }

  const std::vector<index_t>
      part_shape_input_gpu = part_plan_->gpu_input_part_shape();
  const std::vector<index_t>
      part_shape_filter_gpu = part_plan_->gpu_filter_part_shape();
  const std::vector<index_t>
      part_shape_output_gpu = part_plan_->gpu_output_part_shape();
  const std::vector<index_t>
      part_shape_input_cpu  = part_plan_->cpu_input_part_shape();
  const std::vector<index_t>
      part_shape_filter_cpu = part_plan_->cpu_filter_part_shape();
  const std::vector<index_t>
      part_shape_output_cpu = part_plan_->cpu_output_part_shape();

  tensor_manage_util_.reset(new TensorManageUtil(
      dev_context->GetCpuDevice()->allocator()));
  tensor_manage_util_->set_gpu_allocator(
      dev_context->GetGpuDevice()->allocator());
  OpenCLRuntime *opencl_runtime =
      dev_context->GetGpuDevice()->gpu_runtime()->opencl_runtime();
  // Create cpu tensors.
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
                      input_.get(), part_dim, 0, part_length, true);
      part_length = part_shape_filter_gpu[O_OIHW];
      filter_gpu_ = tensor_manage_util_->CreateOrReshapePartTensor(
                      filter_.get(), part_dim, 0, part_length, true);
      part_length = part_shape_output_gpu[in_out_dim_idx];
      output_gpu_ = tensor_manage_util_->CreateOrReshapePartTensor(
                      output_.get(), part_dim, 0, part_length, true);
    }

    const DeviceType in_out_device_type = DeviceType::GPU;
    const DeviceType filter_device_type = DeviceType::GPU;
    bool in_out_mapping_hint = false;
    if (opencl_runtime->ion_type() != IONType::NONE_ION) {
      in_out_mapping_hint = true;
    }
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

  // Create cpu computing kernel.
  conv2d_cpu_kernel_.reset(new Conv2dKernel<DeviceType::CPU, float>(
      input_cpu_, filter_cpu_,
      strides_, paddings_, padding_type_, dilations_,
      activation_, relux_max_limit_, leakyrelu_coefficient_));

  // Create gpu computing kernel.
  const MemoryType opencl_mtype = use_opencl_image ? MemoryType::GPU_IMAGE
                                                   : MemoryType::GPU_BUFFER;
  opencl_kernel_ = CreateOpenCLConv2dKernel(opencl_mtype);

  Tensor *filter_buffer = nullptr;
  Tensor *filter_gpu_wino = nullptr;
  // Prepare winograd setting.
  if (wino_block_size_ == 2 || wino_block_size_ == 4) {
    MACE_CHECK(use_opencl_image == true);
    if (opencl_kernel_->CheckUseWinograd(
        opencl_runtime, filter_shape, output_->shape(),
        strides_.data(), dilations_.data(), &wino_block_size_)) {
      filter_buffer = TensorUtils::CreateBufferTensor(
          dev_context, filter_shape, DT_FLOAT,
          true, DataFormat::NCHW, DeviceType::GPU,
          std::string("filter_buffer"), true);

      filter_gpu_wino = GetWorkspace()->CreateTensor(
          "filter_gpu_wino", gpu_context_->device()->allocator(),
          filter_buffer->dtype(), true);

      ops::OpenCLBufferTransformer(filter_buffer->memory_type(), opencl_mtype).
          Transform(gpu_context_.get(), filter_buffer,
                    OpenCLBufferType::WINOGRAD_FILTER,
                    opencl_mtype, wino_block_size_, filter_gpu_wino);
    } else {
      LOG(INFO) << "Check winograd failed. Set block size to 0";
      wino_block_size_ = 0;
    }
  }

  if (wino_block_size_ != 0) {
    if (!part_plan_->IsCpuOnly()) {
      if (part_plan_->IsGpuOnly()) {
        filter_.reset(filter_gpu_wino);
      } else {
        filter_gpu_ = filter_gpu_wino;
      }
    }

    LOG(INFO) << "Use winograd (blk_size=" << wino_block_size_ << ")";
  }

  cl::CommandQueue &queue = opencl_runtime->command_queue();

  if (!part_plan_->IsGpuOnly()) {
    if (do_data_transform_) {
      LOG(INFO) << "Tranform tensor data to CPU buffer...";
      const OdimRanges *input_odim_ranges = part_plan_->input_odim_ranges();
      ops::TensorImageTransformer().Transform(gpu_context_.get(),
                                              input_.get(),
                                              input_odim_ranges,
                                              wino_block_size_,
                                              input_cpu_);
      TensorDataTransformer data_trasnsformer =
          TensorDataTransformer(MemoryType::GPU_IMAGE, MemoryType::GPU_BUFFER);
      const bool do_filter_data_transform = false;
      if (do_filter_data_transform) {
        data_trasnsformer.Transform(gpu_context_.get(), filter_.get(),
                                    OpenCLBufferType::CONV2D_FILTER,
                                    wino_block_size_, false, filter_cpu_);
      }
      const bool do_bias_data_transform = true;
      if (do_bias_data_transform) {
        data_trasnsformer.Transform(gpu_context_.get(), bias_.get(),
                                    OpenCLBufferType::ARGUMENT,
                                    wino_block_size_, false, bias_cpu_);
      }

      queue.finish();
    }

    const bool do_check_transform_data = false;
    if (do_check_transform_data) {
      if (input_data_template.size() == 1) {
        MACE_CHECK(input_cpu_->data_format() == DataFormat::NCHW);
        MACE_CHECK(TensorDataChecker().EqualBuffer(
            input_cpu_, input_data_template[0]));
      }
      if (filter_data_template.size() == 1) {
        MACE_CHECK(TensorDataChecker().EqualBuffer(
            filter_cpu_, filter_data_template[0]));
      }
      if (bias_data_template.size() == 1) {
        MACE_CHECK(TensorDataChecker().EqualBuffer(
            bias_cpu_, bias_data_template[0]));
      }
    }
  }

  // Warming up.
  const int num_warm_up_rounds = 1;
  LOG(INFO) << "Warm up rounds: " << num_warm_up_rounds;
  int64_t t0, t1;
  if (!part_plan_->IsCpuOnly()) {
    LOG(INFO) << "Warm up GPU kernel...";
    StatsFuture future;
    gpu_context_->set_future(&future);
    for (int i = 0; i < num_warm_up_rounds; i ++) {
      t0 = NowMicros();
      if (part_plan_->IsGpuOnly()) {
        opencl_kernel_->Compute(
            gpu_context_.get(), input_.get(), filter_.get(), bias_.get(),
            strides_.data(), padding_type_, paddings_,
            dilations_.data(), activation_,
            relux_max_limit_, leakyrelu_coefficient_,
            wino_block_size_, output_.get());
      } else {
        const OdimRanges *input_odim_ranges = part_plan_->input_odim_ranges();
        const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
        ops::TensorImageTransformer().Transform(gpu_context_.get(),
                                                input_.get(),
                                                input_odim_ranges,
                                                wino_block_size_,
                                                input_cpu_);
        opencl_kernel_->Compute(
            gpu_context_.get(), input_gpu_, filter_gpu_, bias_.get(),
            strides_.data(), padding_type_, paddings_,
            dilations_.data(), activation_,
            relux_max_limit_, leakyrelu_coefficient_,
            wino_block_size_, output_gpu_);
        ops::TensorImageTransformer().Transform(gpu_context_.get(),
                                                output_cpu_,
                                                output_odim_ranges,
                                                wino_block_size_,
                                                output_.get());
      }
      CallStats call_stats;
      future.wait_fn(&call_stats);
      t1 = NowMicros();
      LOG(INFO) << "Warming up OpenCL kernel takes "
                << ((t1 - t0) / 1000.0) << " ms"
                << " event "
                << (call_stats.end_micros
                    - call_stats.start_micros) / 1000.0 << " ms";
    }
    
    queue.finish();
    gpu_context_->set_future(nullptr);
  }

  const int num_cpu_enqueue = 1;
  if (!part_plan_->IsGpuOnly()) {
    LOG(INFO) << "Warm up CPU kernel...";
    if (opencl_runtime->ion_type() == IONType::NONE_ION) {
      input_cpu_->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                            BlockFlag::BF_TRUE);
      output_cpu_->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                             BlockFlag::BF_TRUE);
    }
    MACE_CHECK(input_cpu_->IsBufferMapped());
    MACE_CHECK(output_cpu_->IsBufferMapped());
    MACE_CHECK(filter_cpu_->IsBufferMapped());
    MACE_CHECK(bias_cpu_->IsBufferMapped());
    for (int i = 0; i < num_warm_up_rounds; i ++) {
      int64_t t0 = NowMicros();
      for (int i = 0; i < num_cpu_enqueue; i ++) {
        conv2d_cpu_kernel_->Compute(cpu_context_.get(),
                                    input_cpu_,
                                    filter_cpu_,
                                    bias_cpu_,
                                    output_cpu_);
      }
      int64_t t1 = NowMicros();
      LOG(INFO) << "Warming up CPU kernel takes "
                << (t1 - t0) / 1000.0 << " ms";
    }
    if (opencl_runtime->ion_type() == IONType::NONE_ION) {
      input_cpu_->UnMapBuffer();
      output_cpu_->UnMapBuffer();
    }

    queue.finish();
  }

  queue.finish();

  LOG(INFO) << "Conv2dTestTask: num_warm_up_rounds " << num_warm_up_rounds
            << " num_cpu_enqueue " << num_cpu_enqueue;
  LOG(INFO) << "Conv2dTestTask: is_debug_on " << is_debug_on_;

  const double gflops =
      MaceFLOPsStatistics::Compute("Conv2D", filter_shape, output_shape)
          / (1000.0 * 1000.0 * 1000.0);
  LOG(INFO) << "Conv2dTestTask: GFLOPs " << gflops;

  return 1;
}

int CodlConv2dCpuGpuTestTask::Run(
    mace::DurationCollector<double> *dura_collector) {
  StatsFuture future;
  
  if (part_plan_->IsCpuGpu()) {
    int64_t t0;
    double enqueued_gpu_time;
    double wait_map_out_time;
    double cpu_compute_time;
    double wait_transform_out_time;
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

    //if (do_data_transform_ && opencl_runtime->ion_type() == IONType::NONE_ION) {
    //  event_manager->CreateSingleEvent(EventActionType::SET,
    //                                   EventOpType::TRANSFORM_INPUT);
    //}

    t0 = NowMicros();
    if (do_data_transform_) {
      if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
        const OdimRanges *input_odim_ranges = part_plan_->input_odim_ranges();
        ops::OpenCLPartBufferTransformer input_transformer(MemoryType::GPU_IMAGE,
                                                           MemoryType::CPU_BUFFER);
        input_transformer.PartTransformFasterImageToBufferEnqueue<DataFormat::NCHW>(
            gpu_context_.get(),
            input_.get(),
            OpenCLBufferType::IN_OUT_CHANNEL,
            wino_block_size_,
            *input_odim_ranges,
            input_cpu_);
        transform_in_future.wait_fn = future.wait_fn;
      }
    }

    if (opencl_runtime->ion_type() == IONType::NONE_ION) {
      if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
        //if (do_data_transform_) {
        //  event_manager->CreateWaitEventFromSetEvent();
        //  event_manager->InsertNullEvent(EventActionType::SET);
        //}

        input_cpu_->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                              BlockFlag::BF_FALSE,
                              &future);
        map_in_future.wait_fn = future.wait_fn;

        //if (do_data_transform_) {
        //  event_manager->InsertNullEvent(EventActionType::WAIT);
        //}
      }
    }

    if (opencl_runtime->ion_type() == IONType::NONE_ION) {
      if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
        // Bug: Segment faults after many rounds?
        //event_manager->CreateSingleEvent(EventActionType::SET,
        //                                 EventOpType::TRANSFORM_INPUT);

        output_cpu_->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                               BlockFlag::BF_FALSE,
                               &future);
        map_out_future.wait_fn = future.wait_fn;

        //event_manager->CreateWaitEventFromSetEvent();
        //event_manager->InsertNullEvent(EventActionType::SET);
      }
    }

    if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_GPU, compute_unit_hint_)) {
      // Enqueue gpu kernel.
      opencl_kernel_->Compute(
          gpu_context_.get(), input_gpu_, filter_gpu_, bias_.get(),
          strides_.data(), padding_type_, paddings_,
          dilations_.data(), activation_,
          relux_max_limit_, leakyrelu_coefficient_,
          wino_block_size_, output_gpu_);
      gpu_compute_future.wait_fn = future.wait_fn;
    }

    //if (opencl_runtime->ion_type() == IONType::NONE_ION) {
    //  event_manager->InsertNullEvent(EventActionType::WAIT);
    //}

    cl::UserEvent *trans_out_user_event = nullptr;
    if (opencl_runtime->ion_type() == IONType::NONE_ION) {
      if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
        // Create a new user event.
        trans_out_user_event
            = event_manager->CreateSingleUserEvent(
                opencl_runtime->context(),
                EventActionType::WAIT,
                EventOpType::TRANSFORM_OUTPUT);
        // Enqueue unmap input tensor.
        input_cpu_->UnMapBuffer(&future);
        unmap_in_future.wait_fn = future.wait_fn;
        event_manager->InsertNullEvent(EventActionType::WAIT);
      }
    }

    if (opencl_runtime->ion_type() == IONType::NONE_ION) {
      if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
        // Enqueue unmap output tensor.
        output_cpu_->UnMapBuffer(&future);
        unmap_out_future.wait_fn = future.wait_fn;
      }
    }

    if (do_data_transform_) {
      if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
        if (opencl_runtime->ion_type() != IONType::NONE_ION) {
          // Create a new user event.
          trans_out_user_event = event_manager->CreateSingleUserEvent(
              opencl_runtime->context(),
              EventActionType::WAIT,
              EventOpType::TRANSFORM_OUTPUT);
        }
    
        // Enqueue transform output kernel.
        const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
        ops::TensorImageTransformer().Transform(gpu_context_.get(),
                                                output_cpu_,
                                                output_odim_ranges,
                                                wino_block_size_,
                                                output_.get());
        transform_out_future.wait_fn = future.wait_fn;
    
        if (opencl_runtime->ion_type() != IONType::NONE_ION) {
          event_manager->InsertNullEvent(EventActionType::WAIT);
        }
      }
    }
    enqueued_gpu_time = (NowMicros() - t0) / 1000.0;

    t0 = NowMicros();
    //if (part_plan_->ratio() != ops::kRatioCpuGpuFull) {
    //  map_out_future.wait_fn(nullptr);
    //}
    map_out_future.wait_fn(nullptr);
    wait_map_out_time = (NowMicros() - t0) / 1000.0;

    if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
      //opencl_runtime->command_queue().flush();
      cpu_context_->set_dura_collector(dura_collector);
      t0 = NowMicros();
      conv2d_cpu_kernel_->Compute(cpu_context_.get(),
                                  input_cpu_,
                                  filter_cpu_,
                                  bias_cpu_,
                                  output_cpu_);
      cpu_compute_time = (NowMicros() - t0) / 1000.0;
      cpu_context_->set_dura_collector(nullptr);
    }

    if (trans_out_user_event != nullptr) {
      event_manager->SetUserEventComplete(trans_out_user_event);
    }
    
    if (part_plan_->ratio() != ops::kRatioCpuGpuFull) {
      t0 = NowMicros();
      future.wait_fn(nullptr);
      wait_transform_out_time = (NowMicros() - t0) / 1000.0;
    } else {
      // NOTE(fucheng): Flush to avoid kernel blocking in queue.
      //                Limit maximum number of gpu enqueued kernels.
      opencl_runtime->command_queue().flush();
      num_gpu_enqueued_kernels_ ++;
      if (num_gpu_enqueued_kernels_ > kMaxGpuEnqueuedKernels) {
        //opencl_runtime->command_queue().finish();
        num_gpu_enqueued_kernels_ = 0;
      }
    }
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
  } else if (part_plan_->IsGpuOnly()) {
    if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_GPU, compute_unit_hint_)) {
      const int64_t t0 = NowMicros();
      gpu_context_->set_future(&future);
      opencl_kernel_->Compute(gpu_context_.get(),
                              input_.get(), filter_.get(), bias_.get(),
                              strides_.data(), padding_type_, paddings_,
                              dilations_.data(), activation_,
                              relux_max_limit_, leakyrelu_coefficient_,
                              wino_block_size_, output_.get());
      future.wait_fn(nullptr);
      if (is_debug_on_) {
        const double gpu_compute_time = (NowMicros() - t0) / 1000.0;
        const double gpu_compute_time_cl = FutureToMillis(&future);
        LOG(INFO) << "Latency:"
                  << " c_gpu " << gpu_compute_time << " ms"
                  << " c_gpu_cl " << gpu_compute_time_cl << " ms";
      }
      if (dura_collector != nullptr) {
        dura_collector->add(FutureToMillis(&future));
      }
      gpu_context_->set_future(nullptr);
    }
  } else if (part_plan_->IsCpuOnly()) {
    if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
      int64_t t0;
      double wait_map_out_time = 0;
      double cpu_compute_time = 0;
      double wait_unmap_in_time = 0;
      double wait_transform_out_time = 0;
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
        ops::TensorImageTransformer().Transform(gpu_context_.get(),
                                                input_.get(),
                                                input_odim_ranges,
                                                wino_block_size_,
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

      if (opencl_runtime->ion_type() != IONType::NONE_ION) {
        // Create a new user event.
        unmap_in_user_event = event_manager->CreateSingleUserEvent(
            opencl_runtime->context(),
            EventActionType::WAIT,
            EventOpType::TRANSFORM_OUTPUT);
      }
      input_cpu_->UnMapBuffer(&future);
      unmap_in_future.wait_fn = future.wait_fn;
      if (opencl_runtime->ion_type() != IONType::NONE_ION) {
        event_manager->InsertNullEvent(EventActionType::WAIT);
      }
      output_cpu_->UnMapBuffer(&future);
      unmap_out_future.wait_fn = future.wait_fn;

      if (do_data_transform_) {
        gpu_context_->set_future(&future);
        const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
        ops::TensorImageTransformer().Transform(gpu_context_.get(),
                                                output_cpu_,
                                                output_odim_ranges,
                                                wino_block_size_,
                                                output_.get());
        transform_out_future.wait_fn = future.wait_fn;
        
      }

      //t0 = NowMicros();
      //map_out_future.wait_fn(nullptr);
      //wait_map_out_time = (NowMicros() - t0) / 1000.0;

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
      t0 = NowMicros();
      //unmap_in_future.wait_fn(nullptr);
      wait_unmap_in_time = (NowMicros() - t0) / 1000.0;

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
        dura_collector->add(dt_latencies);
      }
    }
  } else {
    LOG(ERROR) << "Partition plan error.";
    return -1;
  }

  return 1;
}

} // namespace mace
