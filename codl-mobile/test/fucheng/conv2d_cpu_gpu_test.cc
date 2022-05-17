
#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/core/tensor_manage_util.h"
#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/common/transpose.h"
#include "mace/ops/ref/conv_2d.h"
#include "mace/ops/conv_2d_part_plan.h"
#include "mace/utils/statistics.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/transpose_image.h"
#include "mace/ops/opencl/buffer_transformer.h"
#endif // MACE_ENABLE_OPENCL

#include "test/fucheng/io_util.h"
#include "test/fucheng/device_util.h"
#include "test/fucheng/tensor_buffer_util.h"
#include "test/fucheng/statistics_util.h"
#include "test/fucheng/conv_2d_util.h"
#include "test/fucheng/conv2d_test_param.h"
#include "test/fucheng/conv2d_test.h"

#define FUCHENG_ENABLE_CPU_COMPUTE
#define FUCHENG_ENABLE_GPU_COMPUTE
#define FUCHENG_ENABLE_TRANSFORM_INPUT
#define FUCHENG_ENABLE_TRANSFORM_OUTPUT

#define FUCHENG_ENABLE_PDT_V4_FASTER

//#define FUCHENG_ENABLED_MAP_IN_WAIT_TRANFORM_IN_EVENT
//#define FUCHENG_ENABLED_GPU_COMPUTE_WAIT_MAP_OUT_EVENT

#define FUCHENG_ENABLE_DEBUG_INFO_CPU
#define FUCHENG_ENABLE_DEBUG_INFO_GPU

namespace mace {

class ThreadUtils {
public:
  static inline void SpinEventWait(cl::Event *event) {
    cl_int event_status;
    for (;;) {
      for (int k = 0; k < 1000000; k ++);
      event->getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &event_status);
      if (event_status == CL_COMPLETE) {
        break;
      }
    }
  }
};

class OpenCLQueueUtils {
public:
  static inline void EnqueueOpenCLMarker(OpContext *context) {
    auto runtime = context->device()->gpu_runtime()->opencl_runtime();
    cl::Event event;
    runtime->command_queue().enqueueMarkerWithWaitList(NULL, &event);
    if (context->future() != nullptr) {
      context->future()->wait_fn = [runtime, event](CallStats *stats) {
        event.wait();
        if (stats != nullptr) {
          runtime->GetCallStats(event, stats);
        }
      };
    }
  }
};

MaceStatus Conv2dCpuGpuTest(TestDeviceContext *device_context,
                            const TestParam *test_param) {
  OpContext *gpu_context = new OpContext(GetWorkspace(),
                                         device_context->GetGpuDevice());
  OpContext *cpu_context = new OpContext(GetWorkspace(),
                                         device_context->GetCpuDevice());
  /**
   * BUG(fucheng): We must set cpu_device in cpu context,
   *               otherwise some errors are caused in cpu compute kernels.
   */
  cpu_context->set_cpu_device(device_context->GetCpuDevice());
  //context->set_device(GetGpuDevice());

  // Load parameters for testing.
  const Conv2dTestParam *conv2d_test_param =
      reinterpret_cast<const Conv2dTestParam *>(test_param);
  const bool do_data_transform = conv2d_test_param->do_data_transform;
  const PartitionDim part_dim = PartitionDim::DIM_INPUT_HEIGHT;
  const float part_ratio = conv2d_test_param->part_ratio;
  std::vector<int> paddings;
  Padding padding_type = static_cast<Padding>(static_cast<int>(VALID));
  std::vector<int> dilations = {1, 1};
  const ops::ActivationType activation = ops::ActivationType::NOOP;
  const float relux_max_limit = 0.0f;
  const float leakyrelu_coefficient = 0.0f;
  int wino_block_size = conv2d_test_param->wino_block_size;
  
  const std::vector<index_t> input_shape = conv2d_test_param->input_shape;
  const std::vector<index_t> filter_shape = conv2d_test_param->filter_shape;
  const std::vector<int>     strides = conv2d_test_param->strides;
  const std::vector<index_t> output_shape =
      ops::Conv2dPartPlanUtils::CalcConv2dOutputShape(
          input_shape, filter_shape, strides, paddings, padding_type);
  const std::vector<index_t> bias_shape =
      ops::Conv2dPartPlanUtils::CalcConv2dBiasShape(output_shape);
  
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
  //const float single_result_value =
  //    filter_shape[2] * filter_shape[3]
  //    * (input_shape[3] / input_data_template.size())
  //    * (input_data_template[0] + input_data_template[1])
  //    * filter_data_template[0];
  for (size_t i = 0; i < input_data_template.size(); ++i) {
    single_result_value += (input_data_template[i] * filter_data_template[i]);
  }
  single_result_value *= (input_shape[3] / input_data_template.size());
  single_result_value *= (filter_shape[2] * filter_shape[3]);
  for (size_t i = 0; i < input_data_template.size(); ++i) {
    single_result_value += bias_data_template[i];
  }

  // Set true if we want to use OpenCL image.
  const bool use_opencl_image = true;
  const DataType in_out_dt = DataType::DT_FLOAT;
  const DataType weights_dt = DataType::DT_FLOAT;
  std::unique_ptr<Tensor> input;
  std::unique_ptr<Tensor> filter;
  std::unique_ptr<Tensor> output;
  std::unique_ptr<Tensor> bias;
  if (use_opencl_image) {
    // Create GPU (image) full tensors.
    input.reset(TensorUtils::CreateGPUImageTensor(
        device_context, input_shape, in_out_dt, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("input_image")));
    filter.reset(TensorUtils::CreateGPUImageTensor(
        device_context, filter_shape, weights_dt, true,
        DataFormat::OIHW, OpenCLBufferType::CONV2D_FILTER,
        std::string("filter_image")));
    output.reset(TensorUtils::CreateGPUImageTensor(
        device_context, output_shape, in_out_dt, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("output_image")));
    bias.reset(TensorUtils::CreateGPUImageTensor(
        device_context, bias_shape, weights_dt, false,
        DataFormat::NONE, OpenCLBufferType::ARGUMENT,
        std::string("bias_image")));

    // Fill tensor data.
    LOG(INFO) << "Fill and check tensor image data ...";
    TensorDataFiller data_filler;
    data_filler.FillImage(device_context, gpu_context, input.get(),
                          OpenCLBufferType::IN_OUT_CHANNEL,
                          wino_block_size,
                          input_data_template);
    data_filler.FillImage(device_context, gpu_context, filter.get(),
                          OpenCLBufferType::CONV2D_FILTER,
                          wino_block_size,
                          filter_data_template);
    data_filler.FillImage(device_context, gpu_context, bias.get(),
                          OpenCLBufferType::ARGUMENT,
                          wino_block_size,
                          bias_data_template);

    // Check filled tensor data.
    bool is_equal = false;
    TensorDataChecker data_checker;
    is_equal = data_checker.Equal(device_context, gpu_context, input.get(),
                                  OpenCLBufferType::IN_OUT_CHANNEL,
                                  wino_block_size,
                                  input_data_template[0]);
    MACE_CHECK(is_equal, "input data is wrong");
    is_equal = data_checker.Equal(device_context, gpu_context, filter.get(),
                                  OpenCLBufferType::CONV2D_FILTER,
                                  wino_block_size,
                                  filter_data_template[0]);
    MACE_CHECK(is_equal, "filter data is wrong");
    is_equal = data_checker.Equal(device_context, gpu_context, bias.get(),
                                  OpenCLBufferType::ARGUMENT,
                                  wino_block_size,
                                  bias_data_template[0]);
    MACE_CHECK(is_equal, "bias data is wrong");
  } else {
    // Create GPU (buffer) tensors.
    input.reset(TensorUtils::CreateBufferTensor(
        device_context, input_shape, in_out_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("input_buffer"), true));
    filter.reset(TensorUtils::CreateBufferTensor(
        device_context, filter_shape, weights_dt, true,
        DataFormat::OIHW, DeviceType::GPU,
        std::string("filter_buffer"), true));
    output.reset(TensorUtils::CreateBufferTensor(
        device_context, output_shape, in_out_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("output_buffer"), true));
    MACE_CHECK(bias != nullptr);
  }

  //LOG(INFO) << "Create gpu full tensor success";
  
  // Fill tensor data.
  //FillTensorDataFloat(input, input_data_template);
  //FillTensorDataFloat(filter, filter_data_template);
  //LOG(INFO) << "Fill gpu full tensor data success";
  
  // Make partition plan.
  ops::Conv2dPartPlan plan(PartitionDim::DIM_INPUT_HEIGHT,
                           part_ratio,
                           DataFormat::NCHW);
  int ret = plan.Make(input->shape(), filter->shape(),
                      strides, dilations,
                      padding_type, paddings);
  if (!ret) {
    LOG(ERROR) << "Make conv2d part plan failed";
    return MaceStatus::MACE_RUNTIME_ERROR;
  } else {
    plan.Show();
  }

  // Check partition plan.
  if (plan.IsCpuGpu()) {
    MACE_CHECK(use_opencl_image, "CPU/GPU test must use OpenCL image");
  }
  
  // Press enter to continue.
  PressEnterKeyToContinue();
  
  // Make partation plan.
  //index_t input_cpu_h_dim_start  = 4;
  //index_t output_cpu_h_dim_start = 4;
  //std::vector<index_t> part_shape_input_gpu{1, 4, 7, 3};
  //std::vector<index_t> part_shape_output_gpu{1, 4, 7, 4};
  //std::vector<index_t> part_shape_input_cpu{1, 3, 3, 7};
  //std::vector<index_t> part_shape_output_cpu{1, 4, 3, 7};
  
  //index_t input_cpu_h_dim_start  = plan.input_cpu_h_dim_start();
  //index_t output_cpu_h_dim_start = plan.output_cpu_h_dim_start();
  const std::vector<index_t>
      part_shape_input_gpu = plan.gpu_input_part_shape();
  const std::vector<index_t>
      part_shape_output_gpu = plan.gpu_output_part_shape();
  const std::vector<index_t>
      part_shape_input_cpu  = plan.cpu_input_part_shape();
  const std::vector<index_t>
      part_shape_output_cpu = plan.cpu_output_part_shape();
    
  //MACE_CHECK(plan.input_cpu_h_dim_start()  == input_cpu_h_dim_start);
  //MACE_CHECK(plan.output_cpu_h_dim_start() == output_cpu_h_dim_start);
  //MACE_CHECK(plan.part_shape_input_gpu()   == part_shape_input_gpu);
  //MACE_CHECK(plan.part_shape_output_gpu()  == part_shape_output_gpu);
  //MACE_CHECK(plan.part_shape_input_cpu()   == part_shape_input_cpu);
  //MACE_CHECK(plan.part_shape_output_cpu()  == part_shape_output_cpu);
  
  int64_t t0, t1;
  Tensor *input_gpu = nullptr, *filter_gpu = nullptr, *output_gpu = nullptr;
  Tensor *input_cpu = nullptr, *filter_cpu = nullptr, *output_cpu = nullptr;
  Tensor *bias_cpu = nullptr;
  //Tensor *input_gpu_for_cpu  = nullptr;
  //Tensor *filter_gpu_for_cpu = nullptr;
  //Tensor *output_gpu_for_cpu = nullptr;
  TensorManageUtil tensor_manage_util(
      device_context->GetCpuDevice()->allocator());
  tensor_manage_util.set_gpu_allocator(
      device_context->GetGpuDevice()->allocator());
  OpenCLRuntime *opencl_runtime =
      device_context->GetGpuDevice()->gpu_runtime()->opencl_runtime();
  if (!plan.IsGpuOnly()) {
    if (!plan.IsCpuOnly()) {
      index_t part_height;
      /**
       * Create GPU temporary tensors and GPU temporary tensors
       *   reuse GPU full tensors' buffer.
       */
      //input_gpu  = CreatePartTensorV2(device_context, input,
      //                                0, part_shape_input_gpu.at(1));
      part_height = part_shape_input_gpu[1];
      input_gpu  = tensor_manage_util.CreateOrReshapePartTensor(
                      input.get(), part_dim, 0, part_height, true);
      filter_gpu = filter.get();
      //output_gpu = CreatePartTensorV2(device_context, output,
      //                                0, part_shape_output_gpu.at(1));
      part_height = part_shape_output_gpu[1];
      output_gpu = tensor_manage_util.CreateOrReshapePartTensor(
                      output.get(), part_dim, 0, part_height, false);
      
      //fprintf(stderr, "Info: Create GPU part tensors success\n");
      //LOG(INFO) << "Create GPU part tensors success";
    }

    /**
     * Create CPU temporary tensors and CPU temporary tensors will
     *   get GPU tensors' data by transpose procedure.
     */
#ifdef FUCHENG_ENABLE_PDT_V4_FASTER
    const DeviceType in_out_device_type = DeviceType::GPU;
    const DeviceType filter_device_type = DeviceType::GPU;
    bool in_out_mapping_hint = false;
    if (opencl_runtime->ion_type() != IONType::NONE_ION) {
      in_out_mapping_hint = true;
    }
    const bool gpu_buf_mapping_hint = true;
#else // FUCHENG_ENABLE_PDT_V4_FASTER
    const DeviceType in_out_device_type = DeviceType::CPU;
    const DeviceType filter_device_type = DeviceType::CPU;
    const bool gpu_buf_mapping_hint = false;
#endif // FUCHENG_ENABLE_PDT_V4_FASTER

    input_cpu = tensor_manage_util.CreateOrReshapeTensor(
                    part_shape_input_cpu, DT_FLOAT, false,
                    DataFormat::NCHW, in_out_device_type,
                    std::string("input_cpu"), in_out_mapping_hint);
    filter_cpu = tensor_manage_util.CreateOrReshapeTensor(
                    filter->shape(),  DT_FLOAT, filter->is_weight(),
                    filter->data_format(), filter_device_type,
                    std::string("filter_cpu"), gpu_buf_mapping_hint);
    output_cpu = tensor_manage_util.CreateOrReshapeTensor(
                    part_shape_output_cpu, DT_FLOAT, false,
                    DataFormat::NCHW, in_out_device_type,
                    std::string("output_cpu"), in_out_mapping_hint);
    bias_cpu = tensor_manage_util.CreateOrReshapeTensor(
                    bias->shape(), DT_FLOAT, false,
                    bias->data_format(), in_out_device_type,
                    std::string("bias_cpu"), gpu_buf_mapping_hint);
                                  
    //LOG(INFO) << "Create CPU part tensors success";
    
    // TODO: Create GPU temporary tensors
    //       respond to CPU temporary tensors
    //input_gpu_for_cpu= CreatePartTensorV1(
    //    device_context, input, input_cpu_h_dim_start);
    //input_gpu_for_cpu = tensor_manage_util.CreatePartTensor(
    //    input, input_cpu_h_dim_start, true);
    //PrintTensorInfo(input_cpu);
    //PrintTensorInfo(input_gpu_for_cpu);
    //filter_gpu_for_cpu = filter;
    //filter_cpu->ReuseTensorBuffer(*filter);
    //Tensor::MappingGuard guard(filter_cpu);
    //output_gpu_for_cpu = CreatePartTensorV1(
    //    device_context, output, output_cpu_h_dim_start);
    //output_gpu_for_cpu = tensor_manage_util.CreatePartTensor(
    //    output, output_cpu_h_dim_start, true);
    
    //LOG(INFO) << "Create GPU part tensors for CPU tensors success";
  }
    
  double call_gpu_millis, event_gpu_millis, wait_gpu_millis;
  double run_gpu_millis, run_cpu_millis;
  double enqueue_trans_millis, wait_trans_millis;
  double event_trans_millis;
  double run_trans_millis, run_trans_total_millis;
  double run_op_millis, run_total_rounds_millis;
  double wait_thread_millis;
  TestDelayStatistics delay_statis;

  LOG(INFO) << "Create NEON conv2d delegator";
  std::shared_ptr<Conv2dKernel<DeviceType::CPU, float>>
      conv2d_cpu_kernel(
          new Conv2dKernel<DeviceType::CPU, float>(
              input_cpu, filter_cpu,
              strides, paddings, padding_type, dilations,
              activation, relux_max_limit, leakyrelu_coefficient));
  
  const int num_cpu_enqueue = 1;
  const int num_opencl_enqueue = 1;
  const int num_warm_up_rounds = 1;
  
  CallStats call_stats;
  StatsFuture future;
  StatsFuture gpu_compute_future[2];
  StatsFuture trans_in_future, trans_out_future[2];
  StatsFuture map_in_future, map_output_future;
  StatsFuture unmap_in_future[2], unmap_out_future[2];
  StatsFuture futures[num_opencl_enqueue];

  gpu_context->set_future(&future);

  LOG(INFO) << "Create OpenCL conv2d kernel";
  const MemoryType
      opencl_mtype = use_opencl_image ? MemoryType::GPU_IMAGE
                                      : MemoryType::GPU_BUFFER;
  std::unique_ptr<ops::OpenCLConv2dKernel>
      opencl_kernel = CreateOpenCLConv2dKernel(opencl_mtype);

  // Check if we could use winograd.
  LOG(INFO) << "Check winograd...";
  Tensor *filter_buffer = nullptr;
  Tensor *filter_gpu_wino = nullptr;
  const MemoryType mem_type = MemoryType::GPU_IMAGE;
  if (wino_block_size == 2 || wino_block_size == 4) {
    MACE_CHECK(use_opencl_image == true);
    if (opencl_kernel->CheckUseWinograd(
        opencl_runtime, filter_shape, output->shape(),
        strides.data(), dilations.data(), &wino_block_size)) {
      filter_buffer = TensorUtils::CreateBufferTensor(
                          device_context, filter_shape, DT_FLOAT,
                          true, DataFormat::NCHW, DeviceType::GPU,
                          std::string("filter_buffer"), true);

      filter_gpu_wino = GetWorkspace()->CreateTensor(
          "filter_gpu_wino", gpu_context->device()->allocator(),
          filter_buffer->dtype(), true);

      ops::OpenCLBufferTransformer(filter_buffer->memory_type(), mem_type).
          Transform(gpu_context, filter_buffer,
                    OpenCLBufferType::WINOGRAD_FILTER,
                    mem_type, wino_block_size, filter_gpu_wino);
    } else {
      LOG(INFO) << "Check winograd failed. Set block size to 0";
      wino_block_size = 0;
    }
  }

  if (wino_block_size != 0) {
    if (!plan.IsCpuOnly()) {
      if (plan.IsGpuOnly()) {
        filter.reset(filter_gpu_wino);
      } else {
        filter_gpu = filter_gpu_wino;
      }
    }

    LOG(INFO) << "Use winograd (blk_size=" << wino_block_size << ")";
  }

  cl::CommandQueue &queue = opencl_runtime->command_queue();

  // Data transform for filter and bias tensor.
  if (!plan.IsGpuOnly()) {
    LOG(INFO) << "Tranform tensor data to CPU buffer...";
    const OdimRanges *input_odim_ranges = plan.input_odim_ranges();
    ops::TensorImageTransformer().Transform(gpu_context,
                                            input.get(),
                                            input_odim_ranges,
                                            wino_block_size,
                                            input_cpu);
    TensorDataTransformer data_trasnsformer =
        TensorDataTransformer(MemoryType::GPU_IMAGE,
                              MemoryType::GPU_BUFFER);
    data_trasnsformer.Transform(gpu_context, filter.get(),
                                OpenCLBufferType::CONV2D_FILTER,
                                wino_block_size, false, filter_cpu);
    data_trasnsformer.Transform(gpu_context, bias.get(),
                                OpenCLBufferType::ARGUMENT,
                                wino_block_size, false, bias_cpu);
    queue.finish();

    if (input_data_template.size() == 1) {
      MACE_CHECK(input_cpu->data_format() == DataFormat::NCHW);
      MACE_CHECK(TensorDataChecker().EqualBuffer(
          input_cpu, input_data_template[0]));
    }
    if (filter_data_template.size() == 1) {
      MACE_CHECK(TensorDataChecker().EqualBuffer(
          filter_cpu, filter_data_template[0]));
    }
    if (bias_data_template.size() == 1) {
      MACE_CHECK(TensorDataChecker().EqualBuffer(
          bias_cpu, bias_data_template[0]));
    }
  }
                                                
  // Warm up running for OpenCL kernel.
  if (!plan.IsCpuOnly()) {
    LOG(INFO) << "Warm up GPU kernel...";
    for (int i = 0; i < num_warm_up_rounds; i ++) {
      t0 = NowMicros();
      if (plan.IsGpuOnly()) {
        opencl_kernel->Compute(gpu_context, input.get(), filter.get(), bias.get(),
                               strides.data(), padding_type, paddings,
                               dilations.data(), activation,
                               relux_max_limit, leakyrelu_coefficient,
                               wino_block_size, output.get());
      } else {
        const OdimRanges *input_odim_ranges
            = plan.input_odim_ranges();
        const OdimRanges *output_odim_ranges
            = plan.output_odim_ranges();
        ops::TensorImageTransformer().Transform(gpu_context,
                                                input.get(),
                                                input_odim_ranges,
                                                wino_block_size,
                                                input_cpu);
        opencl_kernel->Compute(gpu_context, input_gpu, filter_gpu, bias.get(),
                               strides.data(), padding_type, paddings,
                               dilations.data(), activation,
                               relux_max_limit, leakyrelu_coefficient,
                               wino_block_size, output_gpu);
        ops::TensorImageTransformer().Transform(gpu_context,
                                                output_cpu,
                                                output_odim_ranges,
                                                wino_block_size,
                                                output.get());
      }
      future.wait_fn(&call_stats);
      t1 = NowMicros();
      LOG(INFO) << "Warming up OpenCL kernel takes "
                << ((t1 - t0) / 1000.0) << " ms"
                << " event "
                << (call_stats.end_micros
                    - call_stats.start_micros) / 1000.0 << " ms";
    }
    queue.finish();
  }

#ifdef FUCHENG_ENABLE_CPU_COMPUTE
  mace::utils::ThreadPool &thread_pool
      = cpu_context->device()->cpu_runtime()->thread_pool();
  thread_pool.SetSpinWaitTime(200 * 1000 * 1000);

  if (!plan.IsGpuOnly()) {
    LOG(INFO) << "Warm up CPU thread...";
    if (opencl_runtime->ion_type() == IONType::NONE_ION) {
      input_cpu->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                           BlockFlag::BF_TRUE);
      output_cpu->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                            BlockFlag::BF_TRUE);
    }
    MACE_CHECK(input_cpu->IsBufferMapped());
    MACE_CHECK(output_cpu->IsBufferMapped());
    MACE_CHECK(filter_cpu->IsBufferMapped());
    MACE_CHECK(bias_cpu->IsBufferMapped());
    for (int i = 0; i < num_warm_up_rounds; i ++) {
      int64_t t0 = NowMicros();
      for (int i = 0; i < num_cpu_enqueue; i ++) {
        thread_pool.Compute1DSample();
        conv2d_cpu_kernel->Compute(cpu_context,
                                   input_cpu,
                                   filter_cpu,
                                   bias_cpu,
                                   output_cpu);
      }
      int64_t t1 = NowMicros();
      LOG(INFO) << "Warming up CPU thread takes "
                << (t1 - t0) / 1000.0 << " ms";
    }
    if (opencl_runtime->ion_type() == IONType::NONE_ION) {
      input_cpu->UnMapBuffer();
      output_cpu->UnMapBuffer();
    }
  }
#endif  // FUCHENG_ENABLE_CPU_COMPUTE
    
  ops::OpenCLPartBufferTransformer buffer_transformer(MemoryType::GPU_IMAGE,
                                                      MemoryType::CPU_BUFFER);
  OpenCLEventManager *event_manager = opencl_runtime->event_manager();

  const int64_t t0_total_rounds = NowMicros();
  const int kCpuGpuStartRound = 1;
  const int kCpuGpuRoundMod = 1;
  std::vector<std::shared_ptr<Conv2dCpuTask>> conv2d_cpu_task_list;

  // Run
  for (int r = 0; r < conv2d_test_param->total_rounds; r++) {
#ifdef FUCHENG_ENABLE_DEBUG_INFO_CPU
    LOG(INFO) << "Round " << r;
#endif
    // Optional condition:
    // 1. r >= kCpuGpuStartRound
    // 2. (r % 2)
    if (plan.IsCpuGpu() &&
        r >= kCpuGpuStartRound &&
        r % kCpuGpuRoundMod == 0) {
      /*****************************
       * CPU-GPU collaboration case.
       *****************************/
      run_cpu_millis = 0.0f;
      run_gpu_millis = 0.0f;
      call_gpu_millis = 0.0f;
      event_gpu_millis = 0.0f;
      run_trans_total_millis = 0.0f;
      run_op_millis = 0.0f;
      wait_thread_millis = 0.0f;
      //output_gpu->Clear();
      //output_gpu_for_cpu->Clear();
      //input_cpu->Clear();
      //filter_cpu->Clear();
      //output_cpu->Clear();
      const int64_t op_t0 = NowMicros();
            
#ifdef FUCHENG_ENABLE_TRANSFORM_INPUT
      gpu_context->set_future(&future);

      // Transpose GPU partital tensors' data to CPU partitial tensors.
      t0 = NowMicros();
      //TensorTranspose(&thread_pool,
      //                input_gpu_for_cpu /* src */,
      //                input_cpu         /* dst */);
      //filter_cpu->Copy(*filter_gpu_for_cpu, BlockFlag::BF_FALSE);
      
#define BUFFER_TRANSFORM_INPUT_ENQUEUE \
      buffer_transformer.PartTransformFasterImageToBufferEnqueue<DataFormat::NCHW>( \
          gpu_context,                      \
          input.get(),                      \
          OpenCLBufferType::IN_OUT_CHANNEL, \
          wino_block_size,                  \
          *input_odim_ranges,               \
          input_cpu);

#ifdef FUCHENG_ENABLED_MAP_IN_WAIT_TRANFORM_IN_EVENT
      if (do_data_transform && opencl_runtime->ion_type() == IONType::NONE_ION) {
        event_manager->CreateSingleEvent(EventActionType::SET,
                                         EventOpType::TRANSFORM_INPUT);
      }
#endif
      
      if (do_data_transform) {
        // New version of tensor image transform.
        const OdimRanges *input_odim_ranges = plan.input_odim_ranges();
        //TensorImageTranspose(gpu_context, input, input_cpu,
        //                     input_odim_ranges, wino_block_size, false);
        //trans_in_future.wait_fn = future.wait_fn;
        //for (int i = 0; i < 3; i ++) {
        //  TensorImageTranspose(gpu_context, input, input_cpu,
        //                       input_odim_ranges, wino_block_size, false);
        //}

        BUFFER_TRANSFORM_INPUT_ENQUEUE
      }
      trans_in_future.wait_fn = future.wait_fn;
      enqueue_trans_millis = (NowMicros() - t0) / 1000.0;
      //queue.flush();
      //cl::Event event;
      //queue.enqueueBarrierWithWaitList(NULL, &event);
      //queue.enqueueBarrierWithWaitList(NULL, NULL);
      //queue.flush();
      //for (int i = 0; i < 1; i ++) {
      //  queue.enqueueMarkerWithWaitList(NULL, NULL);
      //}

      t0 = NowMicros();
      //future.wait_fn(&call_stats);
      t1 = NowMicros();
      wait_trans_millis = (t1 - t0) / 1000.0;
      run_trans_millis = enqueue_trans_millis + wait_trans_millis;
      run_trans_total_millis += run_trans_millis;
      delay_statis.AddTransposeDelay(
          run_trans_millis, TransposeType::TT_GPU_TO_CPU);

      if (opencl_runtime->ion_type() == IONType::NONE_ION) {
#ifdef FUCHENG_ENABLED_MAP_IN_WAIT_TRANFORM_IN_EVENT
        if (do_data_transform) {
          event_manager->CreateWaitEventFromSetEvent();
          event_manager->InsertNullEvent(EventActionType::SET);
        }
#endif

        input_cpu->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                             BlockFlag::BF_FALSE,
                             &future);
        map_in_future.wait_fn = future.wait_fn;

#ifdef FUCHENG_ENABLED_MAP_IN_WAIT_TRANFORM_IN_EVENT
        if (do_data_transform) {
          event_manager->InsertNullEvent(EventActionType::WAIT);
        }
#endif
      }

      //LOG(INFO) << "Transpose GPU to CPU"
      //          << " Host " << run_trans_millis << " ms"
      //          << " Enq " << enqueue_trans_millis << " ms"
      //          << " Wait " << wait_trans_millis << " ms"
      //          << " Device " << (call_stats.end_micros
      //              - call_stats.start_micros) / 1000.0 << " ms";
  
      //LOG(INFO) << "Transform data from gpu part tensors"
      //          << " to cpu part tensors success";
#else  // FUCHENG_ENABLE_TRANSFORM_INPUT
      MACE_UNUSED(event_trans_millis);
      MACE_UNUSED(enqueue_trans_millis);
      MACE_UNUSED(wait_trans_millis);
#endif // FUCHENG_ENABLE_TRANSFORM_INPUT

#ifdef FUCHENG_ENABLE_TRANSFORM_OUTPUT
      if (opencl_runtime->ion_type() == IONType::NONE_ION) {
#ifdef FUCHENG_ENABLED_GPU_COMPUTE_WAIT_MAP_OUT_EVENT
        event_manager->CreateSingleEvent(EventActionType::SET,
                                         EventOpType::TRANSFORM_INPUT);
#endif
        output_cpu->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                              BlockFlag::BF_FALSE,
                              &future);
#ifdef FUCHENG_ENABLED_GPU_COMPUTE_WAIT_MAP_OUT_EVENT
        event_manager->CreateWaitEventFromSetEvent();
        event_manager->InsertNullEvent(EventActionType::SET);
#endif  // FUCHENG_ENABLED_GPU_COMPUTE_WAIT_MAP_OUT_EVENT
      }
#endif  // FUCHENG_ENABLE_TRANSFORM_OUTPUT

#ifdef FUCHENG_ENABLE_TRANSFORM_INPUT
      map_output_future.wait_fn = future.wait_fn;
      //trans_in_future.wait_fn = future.wait_fn;
      //queue.enqueueBarrierWithWaitList(NULL, NULL);
      t0 = NowMicros();
      //if (r == kCpuGpuStartRound) {
      //  map_output_future.wait_fn(nullptr);
      //}
      //queue.flush();
      t1 = NowMicros();
#ifdef FUCHENG_ENABLE_DEBUG_INFO_CPU
      //LOG(INFO) << "CPU ItoB Wait"
      //          << " t0 " << t0
      //          << " t1 " << t1
      //          << " t " << ((t1 - t0) / 1000.0) << " ms";
#endif  // FUCHENG_ENABLE_DEBUG_INFO_CPU
#endif  // FUCHENG_ENABLE_TRANSFORM_INPUT

#ifdef FUCHENG_ENABLE_GPU_COMPUTE
      /*********************************
       * GPU (OpenCL) compute procedure.
       *********************************/
      //future.wait_fn(nullptr);

#ifdef FUCHENG_ENABLED_GPU_COMPUTE_WAIT_MAP_OUT_EVENT
      if (opencl_runtime->ion_type() == IONType::NONE_ION) {
        event_manager->PrintLastEventInfo(EventActionType::WAIT);
      }
#endif
            
      t0 = NowMicros();
      for (int i = 0; i < num_opencl_enqueue; i ++) {
        gpu_context->set_future(&futures[i]);
#define ENQUEUE_GPU_KERNEL                                                     \
        opencl_kernel->Compute(gpu_context, input_gpu, filter_gpu, bias.get(), \
           strides.data(), padding_type, paddings,   \
           dilations.data(), activation,             \
           relux_max_limit, leakyrelu_coefficient,   \
           wino_block_size, output_gpu);
                
        //queue.flush();
        ENQUEUE_GPU_KERNEL
        //OpenCLQueueUtils::EnqueueOpenCLMarker(gpu_context);
        future.wait_fn = futures[i].wait_fn;
        //queue.flush();
      }
      t1 = NowMicros();
      call_gpu_millis = (t1 - t0) / 1000.0;

#ifdef FUCHENG_ENABLED_GPU_COMPUTE_WAIT_MAP_OUT_EVENT
      if (opencl_runtime->ion_type() == IONType::NONE_ION) {
        event_manager->InsertNullEvent(EventActionType::WAIT);
      }
#endif
            
      // Wait for GPU compute kernel.
      //t0 = NowMicros();
      //if (gpu_context->future() != nullptr)
      //    future.wait_fn(&call_stats);
      //t1 = NowMicros();
      //run_gpu_millis += (t1-t0) / 1000.0;

      gpu_compute_future[1].wait_fn = future.wait_fn;
      
      //LOG(INFO) << "Compute on GPU success";
#endif // FUCHENG_ENABLE_GPU_COMPUTE

      cl::UserEvent *trans_out_user_event = nullptr;
      
#ifdef FUCHENG_ENABLE_TRANSFORM_INPUT
      if (opencl_runtime->ion_type() == IONType::NONE_ION) {
        // Create a new user event.
        trans_out_user_event
            = event_manager->CreateSingleUserEvent(
                opencl_runtime->context(),
                EventActionType::WAIT,
                EventOpType::TRANSFORM_OUTPUT);
        // Enqueue unmap input tensor.
        input_cpu->UnMapBuffer(&future);
        unmap_in_future[1].wait_fn = future.wait_fn;
        event_manager->InsertNullEvent(EventActionType::WAIT);
      }
#endif 

#ifdef FUCHENG_ENABLE_TRANSFORM_OUTPUT
      if (opencl_runtime->ion_type() == IONType::NONE_ION) {
        // Enqueue unmap output tensor.
        output_cpu->UnMapBuffer(&future);
        unmap_out_future[1].wait_fn = future.wait_fn;
      }
      if (do_data_transform) {
        if (opencl_runtime->ion_type() != IONType::NONE_ION) {
          // Create a new user event.
          trans_out_user_event
              = event_manager->CreateSingleUserEvent(
                  opencl_runtime->context(),
                  EventActionType::WAIT,
                  EventOpType::TRANSFORM_OUTPUT);
        }
      
        // Enqueue transform output kernel.
        gpu_context->set_future(&future);
        const OdimRanges *output_odim_ranges = plan.output_odim_ranges();
        event_manager->PrintLastEventInfo(EventActionType::WAIT);
        ops::TensorImageTransformer().Transform(gpu_context,
                                                output_cpu,
                                                output_odim_ranges,
                                                wino_block_size,
                                                output.get());
        trans_out_future[1].wait_fn = future.wait_fn;
      
        if (opencl_runtime->ion_type() != IONType::NONE_ION) {
          event_manager->InsertNullEvent(EventActionType::WAIT);
        }
      }
      //queue.flush();
#endif  // FUCHENG_ENABLE_TRANSFORM_OUTPUT

#ifdef FUCHENG_ENABLE_CONV2D_CPU_TASK_MODE
      // Create CPU conv2d task.
      std::shared_ptr<Conv2dCpuTask> conv2d_cpu_task
          = std::make_shared<Conv2dCpuTask>(conv2d_cpu_kernel,
                                            input_cpu,
                                            filter_cpu,
                                            bias_cpu,
                                            output_cpu);
      conv2d_cpu_task->set_in_transform_future(&trans_in_future);
      conv2d_cpu_task->set_out_transform_user_event(trans_out_user_event);
      conv2d_cpu_task_list.push_back(conv2d_cpu_task);
#endif

#ifdef FUCHENG_ENABLE_TRANSFORM_INPUT
      // Wait for completion of transform input tensor kernel.
      t0 = NowMicros();
      map_output_future.wait_fn(nullptr);
      //buffer_transformer.PartTransformNCHWFasterImageToBufferWaitAndCopy(
      //    gpu_context);
      //event.wait();
      //ThreadUtils::SpinEventWait(&event);
      t1 = NowMicros();
#ifdef FUCHENG_ENABLE_DEBUG_INFO_CPU
      LOG(INFO) << "CPU ItoB Wait"
                << " t0 " << t0
                << " t1 " << t1
                << " t " << ((t1 - t0) / 1000.0) << " ms";
#endif
      //t0 = NowMicros();
#ifdef FUCHENG_ENABLE_DEBUG_INFO_GPU
      if (opencl_runtime->ion_type() == IONType::NONE_ION) {
        LOG(INFO) << "Event Profiling Time MOut";
        map_output_future.wait_fn(&call_stats);
        LOG(INFO) << "Event Profiling Time MIn";
        map_in_future.wait_fn(&call_stats);
      }
      LOG(INFO) << "Event Profiling Time ItoB";
      trans_in_future.wait_fn(&call_stats);
#endif
      t1 = NowMicros();
      wait_trans_millis = (t1 - t0) / 1000.0;
      //LOG(INFO) << "Transpose GPU to CPU"
      //          << " Wait " << wait_trans_millis << " ms"
      //          << " Device " << (call_stats.end_micros
      //              - call_stats.start_micros) / 1000.0 << " ms";
#ifdef FUCHENG_ENABLE_DEBUG_INFO_GPU
      LOG(INFO) << "Event Profiling Time GPU Compute";
      gpu_compute_future[0].wait_fn(&call_stats);
      event_gpu_millis = (call_stats.end_micros
                              - call_stats.start_micros) / 1000.0;
      if (opencl_runtime->ion_type() == IONType::NONE_ION) {
        LOG(INFO) << "Event Profiling Time UIn";
        unmap_in_future[0].wait_fn(&call_stats);
        LOG(INFO) << "Event Profiling Time UOut";
        unmap_out_future[0].wait_fn(&call_stats);
      }
      LOG(INFO) << "Event Profiling Time BtoI";
      trans_out_future[0].wait_fn(&call_stats);
      event_trans_millis = (call_stats.end_micros
                              - call_stats.start_micros) / 1000.0;
      //LOG(INFO) << "Device"
      //          << " GPU " << event_gpu_millis << " ms"
      //          << " TransOut " << event_trans_millis << " ms";
#else
      MACE_UNUSED(event_gpu_millis);
      MACE_UNUSED(event_trans_millis);
#endif  // FUCHENG_ENABLE_DEBUG_INFO_GPU
#endif  // FUCHENG_ENABLE_TRANSFORM_INPUT
            
#ifdef FUCHENG_ENABLE_CPU_COMPUTE
      //MACE_CHECK(input_cpu->IsBufferMapped());
      //MACE_CHECK(output_cpu->IsBufferMapped());

      t0 = NowMicros();
      if (r >= kCpuGpuStartRound + 1) {
        for (int i = 0; i < num_cpu_enqueue; i ++) {
          conv2d_cpu_kernel->Compute(cpu_context,
                                     input_cpu,
                                     filter_cpu,
                                     bias_cpu,
                                     output_cpu);
        }
      } else {
        thread_pool.Compute1DSample();
      }

      t1 = NowMicros();
      run_cpu_millis = (t1 - t0) / 1000.0;
      delay_statis.AddCPUDelay(run_cpu_millis);
#ifdef FUCHENG_ENABLE_DEBUG_INFO_CPU
      LOG(INFO) << "CPU Compute"
                << " t0 " << t0
                << " t1 " << t1
                << " t " << run_cpu_millis << " ms";
#endif

      //LOG(INFO) << "Compute on CPU success";
#endif // FUCHENG_ENABLE_CPU_COMPUTE
            
#ifdef FUCHENG_ENABLE_GPU_COMPUTE
      // Wait for GPU compute kernel.
      t0 = NowMicros();
      //future.wait_fn(&call_stats);
      /*****
      for (int i = num_opencl_enqueue - 1; i >= 0; i --) {
        futures[i].wait_fn(&call_stats);
        LOG(INFO) << "Kernel " << i
                  << " event "
                  << (call_stats.end_micros
                      - call_stats.start_micros) / 1000.0
                  << " ms";
      }*****/
      for (int i = 0; i < num_opencl_enqueue; i ++) {
        //futures[i].wait_fn(&call_stats);
        //LOG(INFO) << "Kernel " << i
        //          << " event "
        //          << (call_stats.end_micros
        //              - call_stats.start_micros) / 1000.0
        //          << " ms";
        event_gpu_millis += (call_stats.end_micros
            - call_stats.start_micros) / 1000.0;
      }
      t1 = NowMicros();
      wait_gpu_millis = (t1 - t0) / 1000.0;
      run_gpu_millis = call_gpu_millis + wait_gpu_millis;
      
      //event_gpu_millis
      //    = (call_stats.end_micros - call_stats.start_micros) / 1000.0;
      delay_statis.AddGPUDelay(run_gpu_millis);
#else // FUCHENG_ENABLE_GPU_COMPUTE
      MACE_UNUSED(call_stats);
#endif // FUCHENG_ENABLE_GPU_COMPUTE

#ifdef FUCHENG_ENABLE_TRANSFORM_OUTPUT
      // Output data is copied back with transpose procedure.
      t0 = NowMicros();
      //PrintTensorInfo(output_cpu);
      //PrintTensorInfo(output_gpu_for_cpu);
      //TensorDebugUtils::PrintTensorData(output_gpu_for_cpu);
      //TensorTranspose(&thread_pool,
      //                output_cpu,
      //                output_gpu_for_cpu);
      //gpu_context->set_future(&future);
      //const OdimRanges *output_odim_ranges = plan.output_odim_ranges();
      //TensorImageTranspose(gpu_context, output_cpu, output,
      //                     output_odim_ranges, wino_block_size, false);
      enqueue_trans_millis = (NowMicros() - t0) / 1000.0;
      //trans_out_future.wait_fn = future.wait_fn;
      //queue.enqueueMarkerWithWaitList(NULL, NULL);
      //t0 = NowMicros();
      //queue.flush();
      //LOG(INFO) << "Flush " << (NowMicros() - t0) / 1000.0 << " ms";
      t0 = NowMicros();
      //future.wait_fn(&call_stats);
      t1 = NowMicros();
      wait_trans_millis = (t1 - t0) / 1000.0;
      run_trans_millis = enqueue_trans_millis + wait_trans_millis;
      run_trans_total_millis += run_trans_millis;
      delay_statis.AddTransposeDelay(
          run_trans_millis, TransposeType::TT_CPU_TO_GPU);
      
      //fprintf(stderr,"Info: Copy data from CPU part tensors"
      //        " to GPU part tensors success\n");
      //LOG(INFO) << "Copy data from CPU to GPU output tensor success";
      //LOG(INFO) << "Transpose CPU to GPU"
      //          << " Host " << run_trans_millis << " ms"
      //          << " Enq " << enqueue_trans_millis << " ms"
      //          << " Wait " << wait_trans_millis << " ms"
      //          << " Device " << (call_stats.end_micros
      //              - call_stats.start_micros) / 1000.0 << " ms";

      // Set user event status to CL_COMPLETE.
      //trans_out_user_event->setStatus(CL_COMPLETE);
      if (trans_out_user_event != nullptr) {
        event_manager->SetUserEventComplete(trans_out_user_event);
      }
      //MACE_UNUSED(trans_out_user_event);
#endif // FUCHENG_ENABLE_TRANSFORM_OUTPUT
            
      run_op_millis += (NowMicros() - op_t0) / 1000.0;

      // Should we try to enqueue other kernels?
      //for (int i = 0; i < 3; i ++) {
      //  TensorImageTranspose(gpu_context, output_cpu, output,
      //                       output_odim_ranges, wino_block_size, false);
      //}
      //for (int i = 0; i < 10; i ++) {
      //  ENQUEUE_GPU_KERNEL
      //  queue.enqueueMarkerWithWaitList(NULL, NULL);
      //}

      gpu_compute_future[0].wait_fn = gpu_compute_future[1].wait_fn;
      trans_out_future[0].wait_fn = trans_out_future[1].wait_fn;
      unmap_in_future[0].wait_fn = unmap_in_future[1].wait_fn;
      unmap_out_future[0].wait_fn = unmap_out_future[1].wait_fn;
    } else if (plan.IsGpuOnly() ||
               r < kCpuGpuStartRound ||
               r % kCpuGpuRoundMod != 0) {
      /****************
       * GPU only case.
       ****************/
      //r < kCpuGpuStartRound
      //(r % 2 == 0)

      event_gpu_millis = 0.0f;
      run_trans_total_millis = 0.0f;
      run_op_millis = 0.0f;
      //output->Clear();

      CallStats call_stats;
      StatsFuture future;
      gpu_context->set_future(&future);
      t0 = NowMicros();
      for (int i = 0; i < num_opencl_enqueue; i ++) {
        //queue.enqueueBarrierWithWaitList(NULL, NULL);
        gpu_context->set_future(&futures[i]);
        opencl_kernel->Compute(gpu_context, input.get(), filter.get(), bias.get(),
                               strides.data(), padding_type, paddings,
                               dilations.data(), activation,
                               relux_max_limit, leakyrelu_coefficient,
                               wino_block_size, output.get());
        futures[i].wait_fn(&call_stats);
      }
      t1 = NowMicros();
      call_gpu_millis = (t1 - t0) / 1000.0;
      
      // Wait for GPU compute kernel?
      t0 = NowMicros();
      /*****
      for (int i = num_opencl_enqueue - 1;
              i >= num_opencl_enqueue / 2; i --) {
          futures[i].wait_fn(&call_stats);
          LOG(INFO) << "Kernel " << i
                    << " event "
                    << (call_stats.end_micros
                        - call_stats.start_micros) / 1000.0
                    << " ms";
      }
      for (int i = num_opencl_enqueue / 2 - 1; i >= 0; i --) {
          futures[i].wait_fn(&call_stats);
          LOG(INFO) << "Kernel " << i
                    << " event "
                    << (call_stats.end_micros
                        - call_stats.start_micros) / 1000.0
                    << " ms";
      }*****/
      /*****
      for (int i = 0; i < num_opencl_enqueue / 2; i ++) {
          futures[i].wait_fn(&call_stats);
          LOG(INFO) << "Kernel " << i << " event "
                    << (call_stats.end_micros
                        - call_stats.start_micros) / 1000.0 << " ms";
          event_gpu_millis += (call_stats.end_micros
                                    - call_stats.start_micros) / 1000.0;
      }
      for (int i = num_opencl_enqueue / 2; i < num_opencl_enqueue; i ++) {
          futures[i].wait_fn(&call_stats);
          LOG(INFO) << "Kernel " << i << " event "
                    << (call_stats.end_micros
                        - call_stats.start_micros) / 1000.0 << " ms";
          event_gpu_millis += (call_stats.end_micros
                                    - call_stats.start_micros) / 1000.0;
      }*****/
      MACE_UNUSED(call_stats);
      t1 = NowMicros();
      
      wait_gpu_millis = (t1 - t0) / 1000.0;
      run_gpu_millis = call_gpu_millis + wait_gpu_millis;
      
      //event_gpu_millis =
      //    (call_stats.end_micros - call_stats.start_micros) / 1000.0;
      delay_statis.AddGPUDelay(run_gpu_millis);
      run_cpu_millis = 0.0f;
      run_op_millis += run_gpu_millis;
    } else if (plan.IsCpuOnly()) {
      /****************
       * CPU only case.
       ****************/
      MACE_CHECK(use_opencl_image);

      run_trans_total_millis = 0.0f;
      run_op_millis = 0.0f;
      //output_gpu_for_cpu->Clear();
      //input_cpu->Clear();
      //filter_cpu->Clear();
      //output_cpu->Clear();
      
      // Transpose GPU partitial tensors' data to CPU partitial tensors.
      t0 = NowMicros();
#ifdef FUCHENG_ENABLE_TRANSFORM_INPUT
      //TensorTranspose(&thread_pool,
      //                input_gpu_for_cpu /* src */,
      //                input_cpu         /* dst */);
      //filter_cpu->Copy(*filter_gpu_for_cpu, BlockFlag::BF_TRUE);
      if (do_data_transform) {
        // Transpose GPU partitial tensors' data to CPU partitial tensors.
        gpu_context->set_future(&future);
        const OdimRanges *input_odim_ranges = plan.input_odim_ranges();
        ops::TensorImageTransformer().Transform(gpu_context,
                                           input.get(),
                                           input_odim_ranges,
                                           wino_block_size,
                                           input_cpu);
        trans_in_future.wait_fn = future.wait_fn;
        input_cpu->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                             BlockFlag::BF_FALSE);
        output_cpu->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                              BlockFlag::BF_FALSE,
                              &future);
        LOG(INFO) << "Transform In";
        trans_in_future.wait_fn(&call_stats);
        LOG(INFO) << "Transform Out (Last Round)";
        trans_out_future[0].wait_fn(&call_stats);
      }
#endif // FUCHENG_ENABLE_TRANSFORM_INPUT
      t1 = NowMicros();
      run_trans_millis = (t1-t0) / 1000.0;
      run_trans_total_millis += run_trans_millis;
      delay_statis.AddTransposeDelay(run_trans_millis,
                                     TransposeType::TT_GPU_TO_CPU);
      //LOG(INFO) << "Transpose GPU to CPU " << run_trans_millis << " ms";
      
      // Run CPU compute kernel.
      t0 = NowMicros();
#ifdef FUCHENG_ENABLE_CPU_COMPUTE
      for (int i = 0; i < num_cpu_enqueue; i ++) {
        conv2d_cpu_kernel->Compute(cpu_context,
                                   input_cpu,
                                   filter_cpu,
                                   bias_cpu,
                                   output_cpu);
      }
#else // FUCHENG_ENABLE_CPU_COMPUTE
      MACE_UNUSED(num_cpu_enqueue);
#endif  // FUCHENG_ENABLE_CPU_COMPUTE
      t1 = NowMicros();
      run_gpu_millis = 0.0f;
      run_cpu_millis = (t1 - t0) / 1000.0;
      delay_statis.AddCPUDelay(run_cpu_millis);
      LOG(INFO) << "CPU compute " << run_cpu_millis << " ms";
      
      // Output data is copied back to GPU with transpose procedure.
      t0 = NowMicros();
#ifdef FUCHENG_ENABLE_TRANSFORM_OUTPUT
      if (do_data_transform) {
        input_cpu->UnMapBuffer();
        output_cpu->UnMapBuffer();
        gpu_context->set_future(&future);
        //TensorTranspose(&thread_pool, output_cpu, output_gpu_for_cpu);
        const OdimRanges *output_odim_ranges = plan.output_odim_ranges();
        ops::TensorImageTransformer().Transform(gpu_context,
                                                output_cpu,
                                                output_odim_ranges,
                                                wino_block_size,
                                                output.get());
        //future.wait_fn(&call_stats);
        trans_out_future[0].wait_fn = future.wait_fn;
      }
#endif // FUCHENG_ENABLE_TRANSFORM_OUTPUT
      t1 = NowMicros();
      run_trans_millis = (t1-t0) / 1000.0;
      run_trans_total_millis += run_trans_millis;
      delay_statis.AddTransposeDelay(run_trans_millis,
                                     TransposeType::TT_CPU_TO_GPU);
      //LOG(INFO) << "Transpose CPU to GPU " << run_trans_millis << " ms";
      
      run_op_millis += run_cpu_millis;
      run_op_millis += run_trans_total_millis;

      if (do_data_transform) {
        // Check output result.
        const bool is_equal = TensorDataChecker().EqualBuffer(
            output_cpu, single_result_value);
        //MACE_CHECK(is_equal, "CPU output data is not correct");
        if (!is_equal) {
          LOG(WARNING) << "CPU output data is not correct";
        }
      }
    } else {
      LOG(ERROR) << "Not support part ratio " << part_ratio;
      break;
    }
  
    // QUESTION(fucheng): OpenCL queue finish every round?
    t0 = NowMicros();
    //queue.finish();
    //if (gpu_context->future() != nullptr)
    //    future.wait_fn(&call_stats);
    double run_millis_queue_finish = (NowMicros() - t0) / 1000.0;
    run_op_millis += run_millis_queue_finish;
    
    // Check output data
    //MACE_CHECK(CheckTensorDataFloat(output, single_result_value) == true);
    //if (plan.IsCpuGpu()) {
        //CheckTensorDataFloat(output_cpu, single_result_value);
        //CheckTensorDataFloat(output_gpu, single_result_value);
    //}
    //CheckTensorDataFloat(output, single_result_value);
    //MACE_UNUSED(single_result_value);
    
    //fprintf(stderr, "Info: Delay CPU %.2f ms GPU %.2f ms\n",
    //        run_cpu_millis, run_gpu_millis);
    //LOG(INFO) << "Delay CPU " << (run_cpu_millis + run_trans_total_millis)
    //          << " ms GPU " << run_gpu_millis << " ms";
    
    //TensorDebugUtils::PrintTensorData(output_gpu);
    
    // Sleep
    //usleep(1000 * 1000); // ms
  }

  for (size_t i = 0; i < conv2d_cpu_task_list.size(); ++i) {
    LOG(INFO) << "Round " << i;
    conv2d_cpu_task_list[i]->Run(cpu_context, gpu_context);
  }

  // Command queue finish after all rounds.
  queue.finish();
  // Check output data.
  const bool is_equal = TensorDataChecker().Equal(
                          device_context, gpu_context, output.get(),
                          OpenCLBufferType::IN_OUT_CHANNEL,
                          wino_block_size, single_result_value);
  //MACE_CHECK(is_equal, "Output data is not correct");
  LOG(INFO) << "Check output values: " << is_equal;
  // Show total latency of all rounds.
  run_total_rounds_millis = (NowMicros() - t0_total_rounds) / 1000.0;
  LOG(INFO) << "Total rounds " << conv2d_test_param->total_rounds
            << " latency " << run_total_rounds_millis << " ms";
  // Delete temporary tensors.
  tensor_manage_util.DeleteTensors();
  LOG(INFO) << "Delete temporary tensors successfully";
  // Show statistics delay.
  delay_statis.Show();
  
  return MaceStatus::MACE_SUCCESS;
}

} // namespace mace
