
namespace mace {

#ifdef MACE_ENABLE_OPENCL
MaceStatus Conv2dGpuTestV1(TestDeviceContext *device_context) {
  OpContext *context = new OpContext(GetWorkspace(),
                                     device_context->GetGpuDevice());
  // NOTE: Input shape format is NHWC.
  Tensor *input  = TensorUtils::CreateBufferTensor(
      device_context,
      std::vector<index_t>{1, 7, 7, 3},
      DT_FLOAT, false,
      DataFormat::NHWC, DeviceType::GPU,
      std::string("input"));
  // NOTE: Filter shape format is OIHW.
  Tensor *filter = TensorUtils::CreateBufferTensor(
      device_context,
      std::vector<index_t>{4, 3, 1, 1},
      DT_FLOAT, true,
      DataFormat::NCHW, DeviceType::GPU,
      std::string("filter"));
  Tensor *bias   = nullptr;
  std::vector<int> strides{1, 1};
  Padding padding_type = static_cast<Padding>(static_cast<int>(SAME));
  std::vector<int> paddings{0, 0};
  std::vector<int> dilations{1, 1};
  const ops::ActivationType activation = ops::ActivationType::NOOP;
  const float relux_max_limit = 0.0f;
  const float leakyrelu_coefficient = 0.0f;
  int wino_block_size = 0;
  // NOTE: Output shape format is NHWC.
  const std::vector<index_t> output_shape
      = ops::Conv2dPartPlanUtils::CalcConv2dOutputShape(
          input->shape(), filter->shape(), strides, paddings, padding_type);
  Tensor *output = TensorUtils::CreateBufferTensor(
      device_context, output_shape,
      DT_FLOAT, false,
      DataFormat::NHWC, DeviceType::GPU,
      std::string("output"));
  
  // Fill tensor buffer.
  //input->Copy<float>(
  //    CreateFloatSampleBuffer(input->shape()), input->size());
  //filter->Copy<float>(
  //    CreateFloatSampleBuffer(filter->shape()), filter->size());
  TensorDataFiller data_filler;
  data_filler.Fill(input, std::vector<float>{0.0f, 1.0f, 2.0f});
  data_filler.Fill(filter, std::vector<float>{1.0f});
  
  MemoryType mem_type = MemoryType::GPU_BUFFER;
  
  std::unique_ptr<ops::OpenCLConv2dKernel> kernel;
  
  if (mem_type == MemoryType::GPU_BUFFER) {
    kernel = make_unique<ops::opencl::buffer::Conv2dKernel>();
  }
  
  kernel->Compute(context, input, filter, bias,
                  strides.data(), padding_type, paddings,
                  dilations.data(), activation, relux_max_limit,
                  leakyrelu_coefficient, wino_block_size, output);
  
  // Print output data.
  TensorDebugUtils::PrintTensorData(input);
  TensorDebugUtils::PrintTensorData(filter);
  TensorDebugUtils::PrintTensorData(output);
  
  return MaceStatus::MACE_SUCCESS;
}
#endif // MACE_ENABLE_OPENCL

} // namespace mace
