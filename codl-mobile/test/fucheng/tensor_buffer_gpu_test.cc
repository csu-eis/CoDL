
#include <stdio.h>
#include <stdbool.h>
#include <memory>
#include <string.h>
#include <vector>

#include "mace/ops/common/transpose.h"
#include "test/fucheng/device_util.h"
#include "test/fucheng/tensor_buffer_util.h"

#ifdef MACE_ENABLE_OPENCL

using namespace mace;
using namespace mace::ops;

TestDeviceContext *GetDeviceContext() {
  static TestDeviceContext context(4, CPUAffinityPolicy::AFFINITY_BIG_ONLY);
  return &context;
}

Device *GetGPUDevice() {
  return GetDeviceContext()->GetGpuDevice();
}

void InitDeviceContext() {
  TestDeviceContext *context = GetDeviceContext();
  if (!context->is_initialized()) {
    context->InitGpuDevice();
    context->set_is_initialized(true);
  }
}

int TensorBufferGPUTest() {
  std::vector<index_t> src_tensor_shape{1, 2, 2, 3};
  std::vector<index_t> dst_tensor_shape{1, 1, 2, 3};
  
  // Create tensor.
  Tensor *src_tensor = TensorUtils::CreateBufferTensor(
      GetDeviceContext(),
      src_tensor_shape,
      DT_FLOAT,
      false,
      DataFormat::NHWC,
      DeviceType::GPU,
      std::string("src_tensor"));
                     
  Tensor *dst_tensor = TensorUtils::CreateBufferTensor(
      GetDeviceContext(),
      dst_tensor_shape,
      DT_FLOAT,
      false,
      DataFormat::NHWC,
      DeviceType::CPU,
      std::string("dst_tensor"));
  
  // Fill tensor data.
  float *sam_buf = TensorBufferUtils::CreateSampleBuffer<float>(src_tensor_shape);
  src_tensor->Copy<float>(sam_buf, src_tensor->size());
  
  // CPU tensor reuse GPU tensor buffer by mapping.
  index_t reuse_data_offset = 1 * src_tensor_shape[2] * src_tensor_shape[3] * sizeof(float);
  index_t reuse_data_size   = 1 * src_tensor_shape[2] * src_tensor_shape[3] * sizeof(float);
  dst_tensor->ReuseTensorBuffer(*src_tensor, reuse_data_offset, reuse_data_size);
  
  // Mapping tensor before access its mapped buffer.
  Tensor::MappingGuard guard(dst_tensor);
  
  // Print tensor data.
  TensorDebugUtils::PrintTensorData(dst_tensor);
  
  LOG(INFO) << "Tensor buffer on GPU test success";
  
  return 0;
}

int GPUTensorTransposeTest() {
  // Full and part shape.
  std::vector<index_t> full_shape = {1, 4, 2, 3};
  std::vector<index_t> gpu_part_shape = {1, 2, 2, 3};
  std::vector<index_t> cpu_part_shape = {1, 3, 2, 2};
  
  // Create tensors.
  Tensor *gpu_full_tensor = TensorUtils::CreateBufferTensor(
      GetDeviceContext(),
      full_shape,
      DT_FLOAT,
      false,
      DataFormat::NHWC,
      DeviceType::GPU,
      std::string("gpu_full_tensor"));
  Tensor *gpu_part_tensor = TensorUtils::CreateBufferTensor(
      GetDeviceContext(),
      gpu_part_shape,
      DT_FLOAT,
      false,
      DataFormat::NHWC,
      DeviceType::GPU,
      std::string("gpu_part_tensor"));
  Tensor *cpu_part_tensor = TensorUtils::CreateBufferTensor(
      GetDeviceContext(),
      cpu_part_shape,
      DT_FLOAT,
      false,
      DataFormat::NCHW,
      DeviceType::CPU,
      std::string("cpu_part_tensor"));
  
  // Fill GPU full tensor data.
  const float *sam_buf = TensorBufferUtils::CreateSampleBuffer<float>(full_shape);
  gpu_full_tensor->Copy<float>(sam_buf, gpu_full_tensor->size());
  
  // GPU part tensor reuses GPU full tensor data with offset and size.
  index_t reuse_dim_start = 2;
  index_t reuse_data_offset = reuse_dim_start *
                              full_shape.at(2) *
                              full_shape.at(3) *
                              sizeof(float);
  index_t reuse_data_size = (full_shape.at(1) - reuse_dim_start) *
                            full_shape.at(2) *
                            full_shape.at(3) *
                            sizeof(float);
  gpu_part_tensor->ReuseTensorBuffer(*gpu_full_tensor,
                                     reuse_data_offset,
                                     reuse_data_size);
  
  // Print CPU part tensor data.
  TensorDebugUtils::PrintTensorData(cpu_part_tensor);
  
  // Transpose GPU part tensor to CPU part tensor.
  TensorTranspose(&GetGPUDevice()->cpu_runtime()->thread_pool(),
                  gpu_part_tensor,
                  cpu_part_tensor);
  
  // Print CPU part tensor data.
  TensorDebugUtils::PrintTensorData(cpu_part_tensor);
  
  LOG(INFO) << "GPU tensor transpose test success";
  
  return 0;
}

#endif // MACE_ENABLE_OPENCL

int main(void) {
#ifdef MACE_ENABLE_OPENCL
  InitDeviceContext();

  TensorBufferGPUTest();
  
  GPUTensorTransposeTest();
#else
  LOG(ERROR) << "You should enable OpenCL while compile this test program";
#endif // MACE_ENABLE_OPENCL
  return 0;
}
