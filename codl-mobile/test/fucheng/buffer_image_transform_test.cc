
#include <unistd.h>

#include "mace/ops/ref/conv_2d.h"
#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/conv_pool_2d_util.h"

//#undef MACE_ENABLE_NEON

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/bias_add.h"
#include "mace/ops/arm/fp32/activation.h"
#else
#include "mace/ops/ref/bias_add.h"
#include "mace/ops/ref/activation.h"
#endif // MACE_ENABLE_NEON

#include "test/fucheng/io_util.h"
#include "test/fucheng/device_util.h"
#include "test/fucheng/tensor_buffer_util.h"
#include "test/fucheng/tensor_transpose_util.h"
#include "test/fucheng/conv_2d_util.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/conv_2d.h"

namespace mace {

MaceStatus SwitchInTransformTest() {
  const int size = 3;

  if (size < 4) {
    switch(size) {
      case 3:
        LOG(INFO) << "case 3";
      case 2:
        LOG(INFO) << "case 2";
      case 1:
        LOG(INFO) << "case 1";
    }
  } else {
    LOG(INFO) << "size >= 4";
  }

  return MaceStatus::MACE_SUCCESS;
}

static int mul24(int v1, int v2) {
  return v1 * v2;
}

static int mad24(int v1, int v2, int v3) {
  return v1 * v2 + v3;
}

MaceStatus FilterBufferToImageTest(const float *input,
                                   const std::vector<size_t> image_shape,
                                   const int input_offset,
                                   const int out_channel,
                                   const int filter_h,
                                   const int filter_w,
                                   const int inner_size) {
  MACE_UNUSED(input);
  for (size_t w = 0; w < image_shape[0]; w ++) {
    for (size_t h = 0; h < image_shape[1]; h ++) {
      const int in_channel_idx = w;
      const int hw_size = mul24(filter_w, filter_h);
      int out_channel_idx = h / hw_size;
      const int hw_idx = h - mul24(out_channel_idx, hw_size);
      out_channel_idx = out_channel_idx << 2;
      const int h_idx = hw_idx / filter_w;
      const int w_idx = hw_idx - mul24(h_idx, filter_w);
      const int offset = input_offset +
          mad24(out_channel_idx, inner_size,
            mad24(mad24(in_channel_idx, filter_h, h_idx), filter_w, w_idx));
            
      const std::vector<size_t> image_cord = {w, h};
      const std::vector<int> buffer_cord
          = {out_channel_idx, in_channel_idx, h_idx, w_idx};
      LOG(INFO) << "Image Coordinate " << VectorToString(image_cord)
                << " Buffer Coordinate " << VectorToString(buffer_cord)
                << " Buffer Offset " << offset;

      float values[4];
      if (out_channel_idx < out_channel) {
        const int size = out_channel - out_channel_idx;
        if (size < 4) {
          switch (size) {
            case 3:
              //values[2] = *(input + offset + 2 * inner_size);
              ;
            case 2:
              //values[1] = *(input + offset + 1 * inner_size);
              ;
            case 1:
              //values[0] = *(input + offset);
              ;
          }
        } else {
          //values[3] = *(input + offset + 3 * inner_size);
          //values[2] = *(input + offset + 2 * inner_size);
          //values[1] = *(input + offset + 1 * inner_size);
          //values[0] = *(input + offset);
        }
      }
      
      MACE_UNUSED(values);

      //int2 coord = (int2)(w, h);
      //WRITE_IMAGET(output, coord, values);
    }
  }
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BufferToImageComputeTest(const Tensor *input,
                                    const OpenCLBufferType type,
                                    const int wino_blk_size,
                                    Tensor *output) {
    
  MACE_UNUSED(output);
  
  std::vector<index_t> formatted_buffer_shape
      = ops::FormatBufferShape(input->shape(), type);
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(
      formatted_buffer_shape, type, &image_shape, wino_blk_size);
  
  switch (type) {
    case OpenCLBufferType::CONV2D_FILTER:
      FilterBufferToImageTest(
          nullptr, image_shape,
          static_cast<uint32_t>(
              input->buffer_offset() / GetEnumTypeSize(input->dtype())),
          input->dim(0), input->dim(2), input->dim(3),
          input->dim(1) * input->dim(2) * input->dim(3));
      break;
    case OpenCLBufferType::IN_OUT_CHANNEL:
      break;
    default:
      break;
  }
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus ImageToBufferComputeTest(const Tensor *input,
                                    const OpenCLBufferType type,
                                    const int wino_blk_size,
                                    Tensor *output) {
  return BufferToImageComputeTest(input, type, wino_blk_size, output);
}

MaceStatus TensorImageMappingTest(const Tensor *tensor) {
  MACE_CHECK(tensor->has_opencl_image());
  TensorDebugUtils::PrintTensorData(tensor);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus TensorTransformTest(OpContext *op_context,
                               const Tensor *input,
                               const MemoryType in_mem_type,
                               const MemoryType out_mem_type,
                               const OpenCLBufferType buffer_type,
                               const int wino_block_size,
                               Tensor *output) {
  return ops::OpenCLBufferTransformer(in_mem_type, out_mem_type)
      .Transform(op_context, input, buffer_type,
                 out_mem_type, wino_block_size, output);
}

MaceStatus TensorImageMappingAndTransposeTest(
    OpContext *op_context,
    Tensor *input,
    Tensor *output,
    const DataFormat out_data_format,
    OdimRanges *odim_ranges = nullptr) {
  //LOG(INFO) << "Phase: Before clear output tensor";
  //TensorDebugUtils::PrintTensorData(output);
  
  if (!output->has_opencl_image()) {
    TensorDataClearer clearer;
    clearer.ClearTensor(output);
  }
  //LOG(INFO) << "Phase: Clear output tensor";
  //TensorDebugUtils::PrintTensorData(output);
  
  // NOTE: Must set out data format before Transpose,
  //       otherwise Transpose can not get correct out data format.
  if (output->data_format() != out_data_format) {
    output->set_data_format(out_data_format);
  }
  
  const int64_t t0 = NowMicros();
  TensorTransposeUtil tensor_transpose_util;
  tensor_transpose_util.Transpose(op_context, input, output, odim_ranges, false);
  double time_millis = (NowMicros() - t0) / 1000.0;
  LOG(INFO) << "Map and transpose " << time_millis << " ms";
  
  //LOG(INFO) << "Phase: Transpose input tensor to output tensor";
  //TensorDebugUtils::PrintTensorData(output);
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dComputeUsingPartTensorImageTest(
    OpContext *context,
    const Tensor *input, const Tensor *filter, const Tensor *bias, Tensor *output,
    std::vector<int> strides, Padding padding_type, std::vector<int> paddings,
    std::vector<int> dilations, const ops::ActivationType activation,
    const float relux_max_limit, const float leakyrelu_coefficient,
    int wino_block_size) {
  std::unique_ptr<ops::OpenCLConv2dKernel> kernel(
      new ops::opencl::image::Conv2dKernel());
  
  // HINT: It does not work for GPU_IMAGE tensor.
  //output->Clear();
  
  return kernel->Compute(context, input, filter, bias,
                         strides.data(), padding_type, paddings,
                         dilations.data(), activation, relux_max_limit,
                         leakyrelu_coefficient, wino_block_size, output);
}

MaceStatus Conv2dComputeUsingPartTensorCPUBufferTest(
    OpContext *op_context,
    Tensor *input,
    Tensor *filter,
    Tensor *bias,
    const std::vector<int> strides,
    const std::vector<int> paddings,
    const Padding padding_type,
    const std::vector<int> dilations,
    const ops::ActivationType activation_type,
    const float max_limit,
    const float leakyrelu_coefficient,
    Tensor *output) {
#ifdef MACE_ENABLE_NEON
  std::unique_ptr<ops::arm::fp32::Conv2dBase> conv2d_delegator(
      CreateNEONConv2dDelegator(input, filter, strides, paddings,
                                padding_type, dilations));
  LOG(INFO) << "NEON Conv2d Delegator";
  
  ops::arm::fp32::BiasAdd bias_add_delegator;
  ops::arm::fp32::Activation activation_delegator(
      activation_type, max_limit, leakyrelu_coefficient);
  
  conv2d_delegator->Compute(op_context, input, filter, output);
#else
  std::unique_ptr<ref::Conv2d<float>> ref_conv2d_delegator(
      new ref::Conv2d<float>(strides, dilations, paddings, padding_type));
  LOG(INFO) << "Ref Conv2d Delegator";
  
  ref::BiasAdd bias_add_delegator;
  ref::Activation activation_delegator(
      activation_type, max_limit, leakyrelu_coefficient);
  
  ref_conv2d_delegator->Compute(op_context, input, filter, output);
#endif // MACE_ENABLE_NEON

  bias_add_delegator.Compute(op_context, output, bias, output);
  activation_delegator.Compute(op_context, output, output);
  //MACE_UNUSED(bias);
  //MACE_UNUSED(bias_add_delegator);
  //MACE_UNUSED(activation_delegator);

  LOG(INFO) << "Conv2d Computing Result";
  TensorDebugUtils::PrintTensorData(output);
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BufferImageTransformTest(TestDeviceContext *device_context) {
  //SwitchInTransformTest();
  //LOG(INFO) << "Switch in transform test success";
  //PressEnterKeyToContinue();
  
  OpContext *op_context
      = new OpContext(GetWorkspace(), device_context->GetGpuDevice());
  int wino_block_size = 0;
  
  const std::vector<index_t> input_shape{1, 5, 5, 4};
  // Create input shape with NCHW.
  std::vector<index_t> input_shape_nchw;
  input_shape_nchw.push_back(input_shape[0]);
  input_shape_nchw.push_back(input_shape[3]);
  input_shape_nchw.push_back(input_shape[1]);
  input_shape_nchw.push_back(input_shape[2]);
  const std::vector<index_t> filter_shape{1, 4, 3, 3};
  const std::vector<int> strides{1, 1};
  const std::vector<int> paddings;
  const Padding padding_type = static_cast<Padding>(static_cast<int>(VALID));
  const std::vector<index_t> output_shape =
      ops::Conv2dPartPlanUtils::CalcConv2dOutputShape(
          input_shape, filter_shape, strides, paddings, padding_type);
  const std::vector<index_t> bias_shape = std::vector<index_t>{output_shape[3]};
  
  LOG(INFO) << "Input shape " << ShapeToString(input_shape);
  LOG(INFO) << "Filter shape " << ShapeToString(filter_shape);
  LOG(INFO) << "Output shape " << ShapeToString(output_shape);
  LOG(INFO) << "Bias shape " << ShapeToString(bias_shape);
  
  std::vector<float> input_data_template{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> filter_data_template{1.0f};
  std::vector<float> bias_data_template{1.0f};
  
  // Create GPU buffer tensors.
  Tensor *input  = TensorUtils::CreateBufferTensor(
                                      device_context,
                                      input_shape,
                                      DT_FLOAT, false,
                                      DataFormat::NHWC, DeviceType::CPU,
                                      std::string("input"));
  Tensor *filter = TensorUtils::CreateBufferTensor(
                                      device_context,
                                      filter_shape,
                                      DT_FLOAT, true,
                                      DataFormat::OIHW, DeviceType::CPU,
                                      std::string("filter"));
  Tensor *output = TensorUtils::CreateBufferTensor(
                                      device_context,
                                      output_shape,
                                      DT_FLOAT, false,
                                      DataFormat::NHWC, DeviceType::CPU,
                                      std::string("output"));
  Tensor *bias   = TensorUtils::CreateBufferTensor(
                                      device_context,
                                      bias_shape,
                                      DT_FLOAT, false,
                                      DataFormat::NONE, DeviceType::CPU,
                                      std::string("bias"));
                                
  Tensor *input_cpu_nchw = TensorUtils::CreateBufferTensor(
                                              device_context,
                                              input_shape_nchw,
                                              DT_FLOAT, false,
                                              DataFormat::NCHW, DeviceType::CPU,
                                              std::string("input_cpu_nchw"));
  
  // Fill template data.
  TensorDataFiller data_filler;
  data_filler.Fill(input, input_data_template);
  data_filler.Fill(filter, filter_data_template);
  data_filler.Fill(bias, bias_data_template);
  
  // Buffer to image compute test.
  //BufferToImageComputeTest(filter,
  //                         OpenCLBufferType::CONV2D_FILTER,
  //                         wino_block_size,
  //                         output);
  
  // Create GPU image tensors.
  Tensor *input_image  = TensorUtils::CreateGPUImageTensor(
                            device_context, input_shape,
                            DT_HALF, false,
                            DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
                            std::string("input_image"));
  Tensor *filter_image = TensorUtils::CreateGPUImageTensor(
                            device_context, filter_shape,
                            DT_FLOAT, true,
                            DataFormat::OIHW, OpenCLBufferType::CONV2D_FILTER,
                            std::string("filter_image"));
  Tensor *output_image = TensorUtils::CreateGPUImageTensor(
                            device_context, output_shape,
                            DT_HALF, false,
                            DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
                            std::string("output_image"));
  Tensor *bias_image   = TensorUtils::CreateGPUImageTensor(
                            device_context, bias_shape,
                            DT_FLOAT, false,
                            DataFormat::NONE, OpenCLBufferType::ARGUMENT,
                            std::string("bias_image"));
  
  // Transform CPU buffer tensors to GPU image tensors.
  MemoryType in_mem_type  = MemoryType::CPU_BUFFER;
  MemoryType out_mem_type = MemoryType::GPU_IMAGE;
  TensorTransformTest(op_context, input, in_mem_type, out_mem_type,
                      OpenCLBufferType::IN_OUT_CHANNEL,
                      wino_block_size, input_image);
  TensorTransformTest(op_context, filter, in_mem_type, out_mem_type,
                      OpenCLBufferType::CONV2D_FILTER,
                      wino_block_size, filter_image);
  TensorTransformTest(op_context, bias, in_mem_type, out_mem_type,
                      OpenCLBufferType::ARGUMENT,
                      wino_block_size, bias_image);
  
  // Wait for OpenCL command finishing.
  device_context->GetGpuDevice()->gpu_runtime()->opencl_runtime()
      ->command_queue().finish();
  LOG(INFO) << "Transform CPU buffer to GPU image success";
  
  //TensorDebugUtils::PrintTensorData(input_image);
  //TensorDebugUtils::PrintTensorData(filter_image);
  //TensorDebugUtils::PrintTensorData(bias_image);
  
  //LOG(INFO) << "Please check if the values of tensor is correct";
  //PressEnterKeyToContinue();
  
  // Clear source CPU tensors and transform back.
  /*****
  LOG(INFO) << "Phase: Before Clear";
  TensorDebugUtils::PrintTensorData(input);
  TensorDebugUtils::PrintTensorData(filter);
  TensorDataClearer clearer;
  clearer.ClearTensor(input);
  clearer.ClearTensor(filter);
  LOG(INFO) << "Phase: After Clear";
  TensorDebugUtils::PrintTensorData(input);
  TensorDebugUtils::PrintTensorData(filter);
  *****/
  // Image to buffer compute test.
  //ImageToBufferComputeTest(filter_image,
  //                         OpenCLBufferType::CONV2D_FILTER,
  //                         wino_block_size,
  //                         output);
  
  int test_rounds = (0 + 1);
  in_mem_type  = MemoryType::GPU_IMAGE;
  out_mem_type = MemoryType::CPU_BUFFER;
  
  int64_t t0, t1;
  double time_millis;
  // Measure delay of transform.
  // This process uses default implementation in MACE.
  for (int i = 0; i < test_rounds; i ++) {
    t0 = NowMicros();
    TensorTransformTest(op_context, input_image, in_mem_type, out_mem_type,
                        OpenCLBufferType::IN_OUT_CHANNEL,
                        wino_block_size, input);
    t1 = NowMicros();
    time_millis = (t1 - t0) / 1000.0;
    LOG(INFO) << "Transform input from GPU_IMAGE to CPU_BUFFER " << time_millis << " ms";
    
    t0 = NowMicros();
    TensorTransposeUtil tensor_transpose_util;
    tensor_transpose_util.Transpose(op_context, input, input_cpu_nchw);
    t1 = NowMicros();
    time_millis = (t1 - t0) / 1000.0;
    LOG(INFO) << "Transpose input from NCHW to NHWC " << time_millis << " ms";
    
    t0 = NowMicros();
    TensorTransformTest(op_context, filter_image, in_mem_type, out_mem_type,
                        OpenCLBufferType::CONV2D_FILTER,
                        wino_block_size, filter);
    t1 = NowMicros();
    time_millis = (t1 - t0) / 1000.0;
    LOG(INFO) << "Transform filter from GPU_IMAGE to CPU_BUFFER " << time_millis << " ms";
    
    // Wait for OpenCL command finishing.
    device_context->GetGpuDevice()->gpu_runtime()->opencl_runtime()
        ->command_queue().finish();
  }
  /*****
  LOG(INFO) << "Phase: After Transform";
  TensorDebugUtils::PrintTensorData(input);
  TensorDebugUtils::PrintTensorData(filter);
  *****/
  
  // Tensor image mapping test.
  //TensorImageMappingTest(input_image);
  //TensorImageMappingTest(filter_image);
  
  // Test conv2d computation using part tensor with image type.
  const int gpu_h_dim_start = 0;
  const int gpu_h_dim_end   = 4;
  Tensor *input_gpu_image = TensorUtils::CreatePartTensorV2(device_context,
                                                            input_image,
                                                            gpu_h_dim_start,
                                                            gpu_h_dim_end);
  Tensor *filter_gpu_image = filter_image;
  Tensor *bias_gpu_image   = bias_image;
  Tensor *output_gpu_image = output_image;
  Conv2dComputeUsingPartTensorImageTest(
      op_context, input_gpu_image, filter_gpu_image,
      bias_gpu_image, output_gpu_image,
      strides, static_cast<Padding>(static_cast<int>(VALID)),
      std::vector<int>{0, 0}, std::vector<int>{1, 1},
      ops::ActivationType::NOOP, 0.0f, 0.0f, wino_block_size);
      
  // Wait for OpenCL command finishing.
  device_context->GetGpuDevice()->gpu_runtime()->opencl_runtime()
      ->command_queue().finish();
  
  //TensorDebugUtils::PrintTensorData(input_gpu_image);
  //TensorDebugUtils::PrintTensorData(filter_gpu_image);
  TensorTransformTest(op_context, output_image, in_mem_type, out_mem_type,
                      OpenCLBufferType::IN_OUT_CHANNEL,
                      wino_block_size, output);
  TensorDebugUtils::PrintTensorData(output);
  LOG(INFO) << "Conv2D computation using part tensor image test success";
  PressEnterKeyToContinue();
  
  // Tensor image mapping and transpose test.
  // Height is from GPU (4,7) to CPU (0,3).
  index_t cpu_off_start = 0;
  index_t cpu_off_end   = 4;
  index_t cpu_height    = cpu_off_end - cpu_off_start;
  MACE_CHECK(cpu_height <= input_shape_nchw[2]);
  index_t gpu_off_start = input_image->shape()[1] - cpu_height;
  // Index of output height in NCHW.
  size_t h_dim_index = 2;
  
  OdimRanges odim_ranges(4);
  odim_ranges[h_dim_index].push_back(cpu_off_start);
  odim_ranges[h_dim_index].push_back(cpu_off_end);
  odim_ranges[h_dim_index].push_back(gpu_off_start - cpu_off_start);
  // Input.
  for (int i = 0; i < test_rounds; i ++) {
    // Please set &odim_ranges or nullptr.
    TensorImageMappingAndTransposeTest(op_context, input_image, input,
                                       DataFormat::NCHW, &odim_ranges);
  }
  TensorDebugUtils::PrintTensorData(input);
  // Filter.
  //TensorImageMappingAndTransposeTest(
  //    op_context, filter_image, filter, DataFormat::NCHW);
  // Bias.
  //TensorTransformTest(op_context, bias_image,
  //                    MemoryType::GPU_IMAGE, MemoryType::CPU_BUFFER,
  //                    OpenCLBufferType::ARGUMENT, wino_block_size, bias);
                                     
  // NOTE: No need to transpose filter to filter_image.
  //TensorImageMappingAndTransposeTest(op_context, filter, filter_image,
  //                                   DataFormat::NONE, nullptr);
  
  LOG(INFO) << "Tensor image mapping and transpose test success";
  PressEnterKeyToContinue();
  
  // Test CPU buffer to part GPU image.
  input_data_template = std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f};
  data_filler.Fill(input, input_data_template);
  
  h_dim_index = 1; // Output data format is NHWC.
  odim_ranges = OdimRanges(4);
  odim_ranges[h_dim_index].push_back(gpu_off_start);
  odim_ranges[h_dim_index].push_back(input_image->shape()[1]);
  odim_ranges[h_dim_index].push_back(cpu_off_start - gpu_off_start);
  
  TensorImageMappingAndTransposeTest(op_context, input, input_image,
                                     DataFormat::NONE, &odim_ranges);
                                     
  // Transform data back to CPU for checking.
  h_dim_index = 2;
  odim_ranges = OdimRanges(4);
  odim_ranges[h_dim_index].push_back(cpu_off_start);
  odim_ranges[h_dim_index].push_back(cpu_off_end);
  odim_ranges[h_dim_index].push_back(gpu_off_start - cpu_off_start);
  
  TensorImageMappingAndTransposeTest(op_context, input_image, input,
                                     DataFormat::NCHW, &odim_ranges);
  TensorDebugUtils::PrintTensorData(input);
  
  // Reshape input tensor.
  std::vector<index_t> part_shape;
  part_shape.push_back(input_shape_nchw[0]);
  part_shape.push_back(input_shape_nchw[1]);
  part_shape.push_back(cpu_height);
  part_shape.push_back(input_shape_nchw[3]);
  if (input->shape() != part_shape) {
    input->Reshape(part_shape);
  }
  
  // TODO: Compute using GPU image tensors.
  // TODO: Create CPU buffer tensors.
  // TODO: Transform GPU image tensors to CPU buffer tensors.
  
  // TODO: Compute using CPU buffer tensors.
  // BUG: Ensure CPU Delegator use cpu_device.
  op_context->set_cpu_device(device_context->GetCpuDevice());

  TensorDebugUtils::PrintTensorData(input);
  TensorDebugUtils::PrintTensorData(filter);
  TensorDebugUtils::PrintTensorData(bias);

  Conv2dComputeUsingPartTensorCPUBufferTest(
      op_context, input, filter, bias,
      strides, std::vector<int>{0, 0},
      static_cast<Padding>(static_cast<int>(VALID)),
      std::vector<int>{1, 1},
      ops::ActivationType::NOOP, 0.0f, 0.0f,
      output);
  LOG(INFO) << "Conv2D compute using part tensor CPU buffer test success";
  
  LOG(INFO) << "";
  LOG(INFO) << "Press Ctrl+C to exit";
  getchar();
  
  return MaceStatus::MACE_SUCCESS;
}
#endif // MACE_ENABLE_OPENCL

} // namcespace mace
