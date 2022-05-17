
#include "test/fucheng/device_util.h"
#include "test/fucheng/tensor_buffer_util.h"
#include "test/fucheng/conv_2d_util.h"

#define UNIT_LEVEL_B  0
#define UNIT_LEVEL_KB 1
#define UNIT_LEVEL_MB 2

namespace mace {

std::string DataSizeToString(index_t nbytes) {
  double size = nbytes;
  int unit_level = 0;
  while (size > 1024) {
    size = size / 1024;
    unit_level ++;
  }
  
  std::stringstream stream;
  stream << size << " ";
  switch (unit_level) {
    case UNIT_LEVEL_B:
      stream << "B";
      break;
    case UNIT_LEVEL_KB:
      stream << "KB";
      break;
    case UNIT_LEVEL_MB:
      stream << "MB";
      break;
  }

  return stream.str();
}

#ifdef MACE_ENABLE_OPENCL
MaceStatus BufferMapAndCopyTestInternal(
    TestDeviceContext *device_context,
    const std::vector<index_t> *shape) {
  // Create Tensor
  Tensor *tensor  = TensorUtils::CreateBufferTensor(
      device_context, *shape, DT_FLOAT, false,
      DataFormat::NHWC, DeviceType::GPU,
      std::string("tensor"), true /* alloc_buf */);

  LOG(INFO) << "Tensor Data Size: "
            << DataSizeToString(tensor->raw_size());

  // Start Test
  const int num_rounds = 20;

  LOG(INFO) << "Test: Read Buffer";

  float *tensor_data = reinterpret_cast<float*>(
                           malloc(tensor->raw_size()));
  for (int i = 0; i < num_rounds; i ++) {
    int64_t t0 = NowMicros();

    tensor->Read(tensor_data, tensor->raw_size());

    MACE_UNUSED(tensor_data);

    int64_t t1 = NowMicros();
    double copy_millis = (t1 - t0) / 1000.0;
    LOG(INFO) << "Round " << i << " Read " << copy_millis << " ms";
  }

  free(tensor_data);

  LOG(INFO) << "Test: Map Buffer";

  tensor_data = nullptr;

  for (int i = 0; i < num_rounds; i ++) {
    int64_t t0 = NowMicros();

    Tensor::MappingGuard guard(tensor);
    //tensor_data = tensor->data<float>();

    MACE_UNUSED(guard);

    int64_t t1 = NowMicros();
    double map_millis = (t1 - t0) / 1000.0;
    LOG(INFO) << "Round " << i << " Map " << map_millis << " ms";
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus BufferMapAndCopyTest(TestDeviceContext *device_context) {
  std::vector<index_t> shape;

  shape = std::vector<index_t>{1, 64, 64, 256};
  BufferMapAndCopyTestInternal(device_context, &shape);
  
  shape = std::vector<index_t>{1, 128, 128, 256};
  BufferMapAndCopyTestInternal(device_context, &shape);
  
  shape = std::vector<index_t>{1, 256, 256, 256};
  BufferMapAndCopyTestInternal(device_context, &shape);

  shape = std::vector<index_t>{1, 512, 512, 256};
  BufferMapAndCopyTestInternal(device_context, &shape);

  return MaceStatus::MACE_SUCCESS;
}

#endif // MACE_ENABLE_OPENCL

} // namespace mace
