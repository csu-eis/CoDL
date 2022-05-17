
#include "test/fucheng/tensor_buffer_util.h"

namespace mace {

#define MACE_MAP_BUFFER_ONLY_ONCE true

MaceStatus BufferMapUnmapTestReadInternal(OpContext *context,
                                          Tensor *ta,
                                          Tensor *tb,
                                          Tensor *tc,
                                          const std::vector<float> &data) {
  StatsFuture future;
  context->set_future(&future);

  // Set data of tensor B
  if (!MACE_MAP_BUFFER_ONLY_ONCE) {
    tb->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY);
  }
  //TensorDataFiller().Fill(ta, {0});
  TensorDataFiller().Fill(tb, data);
  if (!MACE_MAP_BUFFER_ONLY_ONCE) {
    tb->UnMapBuffer();
  }

  // Data transform
  // Tensor B to tensor C
  TensorDataTransformer(MemoryType::GPU_BUFFER, MemoryType::GPU_IMAGE)
      .Transform(context, tb, OpenCLBufferType::IN_OUT_CHANNEL, 0, true, tc);

  // Data transform
  // Tensor C to tensor A
  TensorDataTransformer(MemoryType::GPU_IMAGE, MemoryType::GPU_BUFFER)
      .Transform(context, tc, OpenCLBufferType::IN_OUT_CHANNEL, 0, true, ta);

  future.wait_fn(nullptr);

  // Check data of tensor A and tensor B
  //ta->MapBuffer(AllocatorMapType::AMT_READ_ONLY);
  const bool result = TensorDataChecker().Equal(ta, tb);
  if (result) {
    LOG(INFO) << "Good, same data";
  } else {
    LOG(INFO) << "Bad, different data";
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BufferMapUnmapTestRead(OpContext *context,
                                  Tensor *ta,
                                  Tensor *tb,
                                  Tensor *tc) {
  MACE_CHECK(ta->has_opencl_buffer());
  MACE_CHECK(tc->has_opencl_image());

  // Map buffer of tensor A once
  if (MACE_MAP_BUFFER_ONLY_ONCE) {
    ta->MapBuffer(AllocatorMapType::AMT_READ_ONLY);
    //Tensor::MappingGuard guard(ta);
  }

  const std::vector<float> data_0 = {1.0f};
  const std::vector<float> data_1 = {10.0f};
  const std::vector<float> data_2 = {100.0f};
  const std::vector<float> data_3 = {1000.0f};

  BufferMapUnmapTestReadInternal(context, ta, tb, tc, data_0);
  BufferMapUnmapTestReadInternal(context, ta, tb, tc, data_1);
  BufferMapUnmapTestReadInternal(context, ta, tb, tc, data_2);
  BufferMapUnmapTestReadInternal(context, ta, tb, tc, data_3);

  // Unmap buffer of tensor A
  if (MACE_MAP_BUFFER_ONLY_ONCE) {
    ta->UnMapBuffer();
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BufferMapUnmapTestWriteInternal(OpContext *context,
                                           Tensor *ta,
                                           Tensor *tb,
                                           Tensor *tc,
                                           const std::vector<float> &data) {
  StatsFuture future;
  context->set_future(&future);

  if (!MACE_MAP_BUFFER_ONLY_ONCE) {
    ta->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY);
  }
  TensorDataFiller().Fill(ta, data);
  if (!MACE_MAP_BUFFER_ONLY_ONCE) {
    ta->UnMapBuffer();
  }

  // Data transform
  // Tensor A to tensor C
  TensorDataTransformer(MemoryType::GPU_BUFFER, MemoryType::GPU_IMAGE)
      .Transform(context, ta, OpenCLBufferType::IN_OUT_CHANNEL, 0, false, tc);

  // Data transform
  // Tensor C to tensor B
  TensorDataTransformer(MemoryType::GPU_IMAGE, MemoryType::GPU_BUFFER)
      .Transform(context, tc, OpenCLBufferType::IN_OUT_CHANNEL, 0, false, tb);

  future.wait_fn(nullptr);

  // Check data of tensor A and tensor B
  const bool result = TensorDataChecker().Equal(ta, tb);
  if (result) {
    LOG(INFO) << "Good, same data";
  } else {
    LOG(INFO) << "Bad, different data";
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BufferMapUnmapTestWrite(OpContext *context,
                                   Tensor *ta,
                                   Tensor *tb,
                                   Tensor *tc) {
  MACE_CHECK(ta->has_opencl_buffer());
  MACE_CHECK(tc->has_opencl_image());

  // Map buffer of tensor A once
  if (MACE_MAP_BUFFER_ONLY_ONCE) {
    ta->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY);
    //Tensor::MappingGuard guard(ta);
  }

  // Set data of tensor A
  const std::vector<float> data_0 = {2.0f};
  const std::vector<float> data_1 = {20.0f};
  const std::vector<float> data_2 = {200.0f};
  const std::vector<float> data_3 = {2000.0f};

  BufferMapUnmapTestWriteInternal(context, ta, tb, tc, data_0);
  BufferMapUnmapTestWriteInternal(context, ta, tb, tc, data_1);
  BufferMapUnmapTestWriteInternal(context, ta, tb, tc, data_2);
  BufferMapUnmapTestWriteInternal(context, ta, tb, tc, data_3);

  // Unmap buffer of tensor A
  if (MACE_MAP_BUFFER_ONLY_ONCE) {
    ta->UnMapBuffer();
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BufferMapUnmapTest(TestDeviceContext *device_context) {

  LOG(INFO) << "Test name: Buffer Map and Unmap Test";
  LOG(INFO) << "Map once: " << MACE_MAP_BUFFER_ONLY_ONCE;

  std::shared_ptr<OpContext> op_context(
      new OpContext(GetWorkspace(), device_context->GetGpuDevice()));

  const std::vector<index_t> shape = {1, 7, 7, 64};

  Tensor *tensor_buffer[3];
  Tensor *tensor_image;

  tensor_buffer[0] = TensorUtils::CreateTensor<MemoryType::GPU_BUFFER>(
                        device_context, shape, DT_FLOAT, false,
                        DataFormat::NHWC, DeviceType::GPU,
                        OpenCLBufferType::IN_OUT_CHANNEL,
                        std::string("tensor_buffer_a"));

  tensor_buffer[1] = TensorUtils::CreateTensor<MemoryType::GPU_BUFFER>(
                        device_context, shape, DT_FLOAT, false,
                        DataFormat::NHWC, DeviceType::GPU,
                        OpenCLBufferType::IN_OUT_CHANNEL,
                        std::string("tensor_buffer_b"));

  tensor_buffer[2] = TensorUtils::CreateTensor<MemoryType::GPU_BUFFER>(
                        device_context, shape, DT_FLOAT, false,
                        DataFormat::NHWC, DeviceType::GPU,
                        OpenCLBufferType::IN_OUT_CHANNEL,
                        std::string("tensor_buffer_c"));

  tensor_image = TensorUtils::CreateTensor<MemoryType::GPU_IMAGE>(
                        device_context, shape, DT_HALF, false,
                        DataFormat::NHWC, DeviceType::GPU,
                        OpenCLBufferType::IN_OUT_CHANNEL,
                        std::string("tensor_image"));

  BufferMapUnmapTestRead(op_context.get(),
                         tensor_buffer[0],
                         tensor_buffer[2],
                         tensor_image);
  BufferMapUnmapTestWrite(op_context.get(),
                          tensor_buffer[1],
                          tensor_buffer[2],
                          tensor_image);

  return MaceStatus::MACE_SUCCESS;
}

#undef MACE_MAP_BUFFER_ONLY_ONCE

} // namespace mace
