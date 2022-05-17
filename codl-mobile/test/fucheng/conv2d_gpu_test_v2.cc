
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

namespace mace {

MaceStatus Conv2dGpuTestV2(TestDeviceContext *dev_context,
                           const TestParam *test_param) {
  OpContext *gpu_context =
      new OpContext(GetWorkspace(), dev_context->GetGpuDevice());
  
  // Define paramters.
  const Padding padding_type = static_cast<Padding>(static_cast<int>(SAME));
  const std::vector<int> paddings{0, 0};
  const std::vector<int> dilations{1, 1};
  const ops::ActivationType activation   = ops::ActivationType::RELU;
  const float relux_max_limit       = 0.0f;
  const float leakyrelu_coefficient = 0.0f;
  const int wino_block_size         = 0;

  // Get parameters.
  const Conv2dTestParam *conv2d_test_param =
      reinterpret_cast<const Conv2dTestParam *>(test_param);
  const int total_rounds = conv2d_test_param->total_rounds;
  const int gpu_memory_type = conv2d_test_param->gpu_memory_type;
  const std::vector<index_t> input_shape  = conv2d_test_param->input_shape;
  const std::vector<index_t> filter_shape = conv2d_test_param->filter_shape;
  const std::vector<int>     strides      = conv2d_test_param->strides;
  const std::vector<index_t> output_shape
      = ops::Conv2dPartPlanUtils::CalcConv2dOutputShape(
          input_shape, filter_shape, strides, paddings, padding_type);
  const std::vector<index_t> bias_shape
      = ops::Conv2dPartPlanUtils::CalcConv2dBiasShape(output_shape);

  // Create tensors.
  const MemoryType mem_type =
      gpu_memory_type == 0 ? MemoryType::GPU_IMAGE : MemoryType::GPU_BUFFER;
  std::unique_ptr<Tensor> input;
  std::unique_ptr<Tensor> filter;
  std::unique_ptr<Tensor> bias;
  std::unique_ptr<Tensor> output;
  std::unique_ptr<ops::OpenCLConv2dKernel> kernel;
  if (mem_type == MemoryType::GPU_BUFFER) {
    LOG(INFO) << "GPU memory type: Buffer";
    // Create GPU (buffer) full tensors.
    Tensor *input_ptr  = TensorUtils::CreateBufferTensor(
                                dev_context, input_shape,
                                DT_FLOAT, false,
                                DataFormat::NHWC, DeviceType::GPU,
                                std::string("input"));
    Tensor *filter_ptr = TensorUtils::CreateBufferTensor(
                                dev_context, filter_shape,
                                DT_FLOAT, true,
                                DataFormat::NCHW, DeviceType::GPU,
                                std::string("filter"));
    Tensor *bias_ptr    = TensorUtils::CreateBufferTensor(
                                dev_context, bias_shape,
                                DT_FLOAT, false,
                                DataFormat::NONE, DeviceType::GPU,
                                std::string("bias"));
    Tensor *output_ptr  = TensorUtils::CreateBufferTensor(
                                dev_context, output_shape,
                                DT_FLOAT, false,
                                DataFormat::NHWC, DeviceType::GPU,
                                std::string("output"));

    input.reset(input_ptr);
    filter.reset(filter_ptr);
    bias.reset(bias_ptr);
    output.reset(output_ptr);
    
    // Fill tensor buffer.
    TensorDataFiller data_filler;
    data_filler.Fill(input.get(),  std::vector<float>{1.0f});
    data_filler.Fill(filter.get(), std::vector<float>{2.0f});
    data_filler.Fill(bias.get(),   std::vector<float>{3.0f});

    // Create conv2d kernel.
    kernel = make_unique<ops::opencl::buffer::Conv2dKernel>();
  } else if (mem_type == MemoryType::GPU_IMAGE) {
    LOG(INFO) << "GPU memory type: Image";
    // Create GPU (image) full tensors.
    Tensor *input_ptr = TensorUtils::CreateGPUImageTensor(
        dev_context, input_shape, DT_FLOAT, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("input_image"));
    Tensor *filter_ptr = TensorUtils::CreateGPUImageTensor(
        dev_context, filter_shape, DT_FLOAT, true,
        DataFormat::OIHW, OpenCLBufferType::CONV2D_FILTER,
        std::string("filter_image"));
    Tensor *output_ptr = TensorUtils::CreateGPUImageTensor(
        dev_context, output_shape, DT_FLOAT, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("output_image"));
    Tensor *bias_ptr = TensorUtils::CreateGPUImageTensor(
        dev_context, bias_shape, DT_FLOAT, false,
        DataFormat::NONE, OpenCLBufferType::ARGUMENT,
        std::string("bias_image"));

    input.reset(input_ptr);
    filter.reset(filter_ptr);
    bias.reset(bias_ptr);
    output.reset(output_ptr);

    // Fill tensor data.
    const std::vector<float> input_data_template{1.0f};
    const std::vector<float> filter_data_template{1.0f};
    const std::vector<float> bias_data_template{1.0f};

    MACE_CHECK(input_shape[3] % input_data_template.size() == 0);
    MACE_CHECK(filter_shape[1] % filter_data_template.size() == 0);
    MACE_CHECK(bias_shape[0] % bias_data_template.size() == 0);
    MACE_CHECK(input_data_template.size() == filter_data_template.size());
    MACE_CHECK(input_data_template.size() == bias_data_template.size());

    LOG(INFO) << "Fill and check tensor image data ...";
    TensorDataFiller data_filler;
    data_filler.FillImage(dev_context, gpu_context, input.get(),
                          OpenCLBufferType::IN_OUT_CHANNEL,
                          wino_block_size,
                          input_data_template);
    data_filler.FillImage(dev_context, gpu_context, filter.get(),
                          OpenCLBufferType::CONV2D_FILTER,
                          wino_block_size,
                          filter_data_template);
    data_filler.FillImage(dev_context, gpu_context, bias.get(),
                          OpenCLBufferType::ARGUMENT,
                          wino_block_size,
                          bias_data_template);

    // Check filled tensor data.
    bool is_equal = false;
    TensorDataChecker data_checker;
    is_equal = data_checker.Equal(dev_context, gpu_context, input.get(),
                                  OpenCLBufferType::IN_OUT_CHANNEL,
                                  wino_block_size,
                                  input_data_template[0]);
    MACE_CHECK(is_equal, "input data is wrong");
    is_equal = data_checker.Equal(dev_context, gpu_context, filter.get(),
                                  OpenCLBufferType::CONV2D_FILTER,
                                  wino_block_size,
                                  filter_data_template[0]);
    MACE_CHECK(is_equal, "filter data is wrong");
    is_equal = data_checker.Equal(dev_context, gpu_context, bias.get(),
                                  OpenCLBufferType::ARGUMENT,
                                  wino_block_size,
                                  bias_data_template[0]);
    MACE_CHECK(is_equal, "bias data is wrong");

    // Create conv2d kernel.
    kernel = make_unique<ops::opencl::image::Conv2dKernel>();
  } else {
    LOG(ERROR) << "Unknown memory type";
    return MaceStatus::MACE_RUNTIME_ERROR;
  }

  // OpenCL queue.
  OpenCLRuntime *opencl_runtime =
      dev_context->GetGpuDevice()->gpu_runtime()->opencl_runtime();
  cl::CommandQueue &queue = opencl_runtime->command_queue();
  
  // Run.
  for (int i = 0; i < total_rounds; i ++) {
    const int64_t t0 = NowMicros();

    kernel->Compute(gpu_context, input.get(), filter.get(), bias.get(),
                    strides.data(), padding_type, paddings,
                    dilations.data(), activation, relux_max_limit,
                    leakyrelu_coefficient, wino_block_size, output.get());

    // Finish.
    queue.finish();

    const double run_millis = (NowMicros() - t0) / 1000.0;
    LOG(INFO) << "Round " << i << " Time " << run_millis << " ms";
  }

  return MaceStatus::MACE_SUCCESS;
}

} // namespace
