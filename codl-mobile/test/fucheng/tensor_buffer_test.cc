
#include <stdio.h>
#include <stdbool.h>
#include <memory>
#include <numeric>
#include <string.h>
#include <vector>

#include "mace/core/allocator.h"
#include "mace/core/buffer.h"
#include "mace/core/device.h"
#include "mace/core/preallocated_pooled_allocator.h"
#include "mace/core/tensor.h"
#include "mace/ops/common/transpose.h"
#include "mace/public/mace.h"
#include "mace/utils/thread_pool.h"
#include "test/fucheng/device_util.h"
#include "test/fucheng/tensor_buffer_util.h"

using namespace mace;
using namespace mace::ops;

TestDeviceContext *GetDeviceContext() {
  static TestDeviceContext dev_context(4, CPUAffinityPolicy::AFFINITY_BIG_ONLY);
  return &dev_context;
}

Device *GetCPUDevice() {
  TestDeviceContext *dev_context = GetDeviceContext();
  if (!dev_context.is_initialized()) {
    dev_context.InitCpuDevice();
    dev_context.set_is_initialized(true);
  }
  return dev_context->GetCpuDevice();
}

int TensorBufferTest() {
  index_t buf_data_size = 1024;
  const unsigned char *buf_data = new unsigned char[buf_data_size];
  
  Buffer *cpu_buf = new Buffer(GetCPUDevice()->allocator(),
                               const_cast<unsigned char*>(buf_data),
                               buf_data_size);
  
  MACE_UNUSED(cpu_buf);
  
  LOG(INFO) << "Tensor buffer test success";
  
  return 0;
}

int PreallocatedAllocatorTest() {
  index_t buf_data_size = 1024;
  
  // Create and allocate tensor buffer.
  std::unique_ptr<BufferBase> tensor_buf(new Buffer(GetCPUDevice()->allocator()));
  tensor_buf->Allocate(buf_data_size);
  
  // Bind allocator with tensor buffer.
  PreallocatedPooledAllocator preallocated_allocator;
  preallocated_allocator.SetBuffer(0, std::move(tensor_buf));
  LOG(INFO) << "Preallocated allocator test success";
  
  // Create tensor using buffer allocated by preallocated allocator.
  std::vector<index_t> tensor_shape{1, 7, 7, 3};
  index_t tensor_size = std::accumulate(tensor_shape.begin(),
                                        tensor_shape.end(),
                                        1, std::multiplies<int64_t>());
  index_t     tensor_data_size  = tensor_size * sizeof(float);
  DataType    tensor_data_type  = DT_FLOAT;
  bool        tensor_is_weight  = false;
  std::string tensor_name       = "tensor";
  
  std::unique_ptr<Tensor> tensor(new Tensor(
      BufferSlice(preallocated_allocator.GetBuffer(0), 0, tensor_data_size),
      tensor_data_type,
      tensor_is_weight,
      tensor_name));
  
  tensor->Reshape(tensor_shape);
  
  // Create a new tensor reusing below tensor data.
  std::vector<index_t> tensor2_shape{1, 3, 7, 3};
  index_t tensor2_size = std::accumulate(tensor2_shape.begin(),
                                         tensor2_shape.end(),
                                         1, std::multiplies<int64_t>());
  index_t tensor2_data_size   = tensor2_size * sizeof(float);
  index_t tensor2_data_offset = tensor_shape[0] * 4 * tensor_shape[2] * tensor_shape[3] * sizeof(float);
  
  std::unique_ptr<Tensor> tensor2(new Tensor(
      GetCPUAllocator(),
      tensor->dtype(),
      false,
      std::string("tensor2")));
  tensor2->ReuseTensorBuffer(*tensor, tensor2_data_offset, tensor2_data_size);
  tensor2->Reshape(tensor2_shape);
  
  LOG(INFO) << "Create new tensor and reusing data test success";
  
  return 0;
}

int TensorDataFormatTransposeTest() {
  std::vector<index_t> src_tensor_shape{1, 2, 2, 3};
  std::vector<index_t> dst_tensor_shape{1, 3, 2, 2};
  
  // Create source and target tensor.
  Tensor *src_tensor = TensorUtils::CreateBufferTensor(GetDeviceContext(),
                                                       src_tensor_shape,
                                                       DT_FLOAT,
                                                       false);
  
  Tensor *dst_tensor = TensorUtils::CreateBufferTensor(GetDeviceContext(),
                                                       dst_tensor_shape,
                                                       DT_FLOAT,
                                                       false);
  // Fill tensor data.
  float *sam_buf = TensorBufferUtils::CreateSampleBuffer<float>(src_tensor_shape);
  src_tensor->Copy<float>(sam_buf, src_tensor->size());
  
  // Print tensor data.
  TensorDebugUtils::PrintTensorData(src_tensor);
  
  // Transpose.
  std::vector<int> dst_dims{0, 3, 1, 2};  // NHWC to NCHW
  Tensor *input  = src_tensor;
  Tensor *output = dst_tensor;
  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  
  Transpose(&GetCPUDevice()->cpu_runtime()->thread_pool(),
            input_data,
            input->shape(),
            dst_dims,
            output_data);
  
  // Print tensor data.
  TensorDebugUtils::PrintTensorData(dst_tensor);
  
  LOG(INFO) << "Tensor data format transpose test success";
  
  return 0;
}

int main(void) {
  TensorBufferTest();
  
  PreallocatedAllocatorTest();
  
  TensorDataFormatTransposeTest();
  
  return 0;
}
