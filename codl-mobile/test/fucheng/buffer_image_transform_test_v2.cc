
#include <unistd.h>

#include "mace/ops/ref/conv_2d.h"
#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/conv_pool_2d_util.h"

#include "test/fucheng/io_util.h"
#include "test/fucheng/device_util.h"
#include "test/fucheng/tensor_buffer_util.h"
#include "test/fucheng/conv_2d_util.h"
#include "test/fucheng/conv2d_test_param.h"
#include "test/fucheng/conv2d_test.h"

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/bias_add.h"
#include "mace/ops/arm/fp32/activation.h"
#else // MACE_ENABLE_NEON
#include "mace/ops/ref/bias_add.h"
#include "mace/ops/ref/activation.h"
#endif // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/transpose_image.h"
#include "mace/ops/opencl/image/conv_2d.h"
#endif

namespace mace {

#if 0

enum TransformWay {
  WAY_CPU_TO_GPU = 0,
  WAY_GPU_TO_CPU = 1
};

MaceStatus BufferImageTransformTestV2Mace(TestDeviceContext *device_context,
                                          const TestParam *test_param,
                                          const TransformWay transform_way) {

  OpContext *op_context =
      new OpContext(GetWorkspace(), device_context->GetGpuDevice());
  const int wino_block_size = 0;

  const Conv2dTestParam *conv2d_test_param =
      reinterpret_cast<const Conv2dTestParam *>(test_param);
  const int total_rounds = conv2d_test_param->total_rounds;
  const std::vector<index_t> input_shape = conv2d_test_param->input_shape;
  // Create input shape with NCHW.
  std::vector<index_t> input_shape_nchw =
      ops::ShapeTransposeUtil::ShapeTranspose(input_shape, DST_DIMS_NHWC_TO_NCHW);
  const std::vector<index_t> filter_shape = conv2d_test_param->filter_shape;
  const std::vector<int>     strides = conv2d_test_param->strides;
  const std::vector<int>     paddings;
  const Padding padding_type = static_cast<Padding>(static_cast<int>(VALID));
  const std::vector<index_t> output_shape =
      ops::Conv2dPartPlanUtils::CalcConv2dOutputShape(
          input_shape, filter_shape, strides, paddings, padding_type);
  const std::vector<index_t> bias_shape = std::vector<index_t>{output_shape[3]};
  
  LOG(INFO) << "Input shape " << ShapeToString(input_shape);
  LOG(INFO) << "Filter shape " << ShapeToString(filter_shape);
  LOG(INFO) << "Output shape " << ShapeToString(output_shape);
  LOG(INFO) << "Bias shape " << ShapeToString(bias_shape);

  // Create CPU buffer tensor.
  std::unique_ptr<Tensor> input_cpu_nchw(TensorUtils::CreateBufferTensor(
      device_context,
      input_shape_nchw,
      DT_FLOAT, false,
      DataFormat::NCHW, DeviceType::CPU,
      std::string("input_cpu_nchw")));

  // Create GPU image tensors.
  std::unique_ptr<Tensor> input_image(TensorUtils::CreateGPUImageTensor(
      device_context, input_shape,
      DT_FLOAT, false,
      DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
      std::string("input_image")));

  const int rounds = total_rounds;
  Tensor *input, *output;
  ops::TensorImageTransformer transformer;
  if (transform_way == TransformWay::WAY_CPU_TO_GPU) {
    input = input_cpu_nchw.get();
    output = input_image.get();
  } else if (transform_way == TransformWay::WAY_GPU_TO_CPU) {
    input = input_image.get();
    output = input_cpu_nchw.get();
  }
  
  const int numItems = 3;
  transformer.SetEnableDebug(true, numItems);
  for (int i = 0; i < rounds; ++i) {
    transformer.TransformV1(op_context, input, wino_block_size, output);
  }
  transformer.PrintDebugInfo(ops::TDI_TYPE_AVGERAGE);

  return MaceStatus::MACE_SUCCESS;
}

void BuildOdimRangesImageToNCHWBuffer(OdimRanges &odim_ranges,
                                      index_t cpu_len,
                                      index_t total_len) {
  const size_t h_dim_idx = H_NCHW;
  const index_t cpu_off_start = 0;
  const index_t cpu_off_end   = cpu_len;
  const index_t gpu_off_start = total_len - cpu_len;
  odim_ranges[h_dim_idx].push_back(cpu_off_start);
  odim_ranges[h_dim_idx].push_back(cpu_off_end);
  odim_ranges[h_dim_idx].push_back(gpu_off_start - cpu_off_start);
}

MaceStatus BufferImageTransformTestV2Ours(TestDeviceContext *device_context,
                                          const TestParam *test_param,
                                          const TransformWay transform_way) {

  OpContext *op_context =
      new OpContext(GetWorkspace(), device_context->GetGpuDevice());
  const int wino_block_size = 0;

  const Conv2dTestParam *conv2d_test_param =
      reinterpret_cast<const Conv2dTestParam *>(test_param);
  const int total_rounds = conv2d_test_param->total_rounds;
  const float part_ratio = conv2d_test_param->part_ratio;
  const std::vector<index_t> input_shape = conv2d_test_param->input_shape;
  // Create input shape with NCHW.
  std::vector<index_t> input_shape_nchw =
      ops::ShapeTransposeUtil::ShapeTranspose(input_shape, DST_DIMS_NHWC_TO_NCHW);
  const std::vector<index_t> filter_shape = conv2d_test_param->filter_shape;
  const std::vector<int>     strides = conv2d_test_param->strides;
  const std::vector<int>     paddings;
  const Padding padding_type = static_cast<Padding>(static_cast<int>(VALID));
  const std::vector<index_t> output_shape =
      ops::Conv2dPartPlanUtils::CalcConv2dOutputShape(
          input_shape, filter_shape, strides, paddings, padding_type);
  const std::vector<index_t> bias_shape = std::vector<index_t>{output_shape[3]};
  
  LOG(INFO) << "Input shape " << ShapeToString(input_shape);
  LOG(INFO) << "Filter shape " << ShapeToString(filter_shape);
  LOG(INFO) << "Output shape " << ShapeToString(output_shape);
  LOG(INFO) << "Bias shape " << ShapeToString(bias_shape);

  // Create CPU buffer tensor.
  std::unique_ptr<Tensor> input_cpu_nchw(TensorUtils::CreateBufferTensor(
      device_context,
      input_shape_nchw,
      DT_FLOAT, false,
      DataFormat::NCHW, DeviceType::GPU,
      std::string("input_cpu_nchw")));

  // Create GPU image tensors.
  std::unique_ptr<Tensor> input_image(TensorUtils::CreateGPUImageTensor(
      device_context, input_shape,
      DT_FLOAT, false,
      DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
      std::string("input_image")));

  const float p_ratio = part_ratio;
  const index_t cpu_len = static_cast<index_t>(input_shape[1] * p_ratio);
  OdimRanges odim_ranges(4);
  Tensor *input, *output;
  ops::TensorImageTransformer transformer;
  const int rounds = total_rounds;
  
  BuildOdimRangesImageToNCHWBuffer(odim_ranges, cpu_len, input_shape[1]);
  
  if (transform_way == TransformWay::WAY_CPU_TO_GPU) {
    input = input_cpu_nchw.get();
    output = input_image.get();
  } else if (transform_way == TransformWay::WAY_GPU_TO_CPU) {
    input = input_image.get();
    output = input_cpu_nchw.get();
  }
  
  const int numItems = 2;
  transformer.SetEnableDebug(true, numItems);
  for (int i = 0; i < rounds; ++i) {
    transformer.TransformV4(op_context,
                            input,
                            &odim_ranges,
                            wino_block_size,
                            output);
  }

  transformer.PrintDebugInfo();

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BufferImageTransformTestV2(TestDeviceContext *context,
                                      const TestParam *test_param) {
  BufferImageTransformTestV2Mace(
      context, test_param, TransformWay::WAY_CPU_TO_GPU);
  BufferImageTransformTestV2Mace(
      context, test_param, TransformWay::WAY_GPU_TO_CPU);
  BufferImageTransformTestV2Ours(
      context, test_param, TransformWay::WAY_CPU_TO_GPU);
  BufferImageTransformTestV2Ours(
      context, test_param, TransformWay::WAY_GPU_TO_CPU);
  return MaceStatus::MACE_SUCCESS;
}

#else

MaceStatus BufferImageTransformTestV2(TestDeviceContext *context,
                                      const TestParam *test_param) {
  MACE_UNUSED(context);
  MACE_UNUSED(test_param);

  return MaceStatus::MACE_SUCCESS;
}

#endif

} // namespace mace
