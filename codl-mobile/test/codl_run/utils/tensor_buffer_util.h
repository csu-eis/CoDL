
#ifndef TEST_CODL_RUN_UTILS_TENSOR_BUFFER_UTIL_H_
#define TEST_CODL_RUN_UTILS_TENSOR_BUFFER_UTIL_H_

#include <stdbool.h>
#include <string.h>
#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/core/memory_optimizer.h"
#include "mace/ops/conv_2d_part_plan.h"
#include "mace/public/mace.h"
#include "mace/utils/thread_pool.h"
#include "test/codl_run/utils/device_util.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_util.h"
#endif

#define EXTRA_BUFFER_PAD_SIZE 64

typedef mace::ops::OdimRanges OdimRanges;

using namespace mace;

const std::string ShapeToString(const std::vector<index_t> &shape);

const std::string DataFormatToString(const DataFormat df);

class TensorBufferUtils {
public:
  template<typename T>
  static T* CreateSampleBuffer(const std::vector<index_t> &shape);
};

class TensorUtils {
 public:
  static Tensor* CreateBufferTensor(
      TestDeviceContext *device_context,
      const std::vector<index_t> &shape,
      const DataType dt,
      const bool is_weight,
      const DataFormat data_format = DataFormat::NCHW,
      const DeviceType device_type = DeviceType::CPU,
      const std::string name = "",
      const bool alloc_buffer_hint = true,
      const bool map_buffer_hint = false);
      
#ifdef MACE_ENABLE_OPENCL
  static Tensor* CreateGPUImageTensor(
      TestDeviceContext *device_context,
      const std::vector<index_t> &shape,
      const DataType dt,
      const bool is_weight,
      const DataFormat data_format,
      const OpenCLBufferType buffer_type,
      const std::string name = "");
#endif

  static Tensor* CreatePartTensorV1(TestDeviceContext *device_context,
                                    Tensor *src,
                                    const index_t h_dim_start);

  static Tensor* CreatePartTensorV2(TestDeviceContext *device_context,
                                    Tensor *src,
                                    const index_t h_dim_start,
                                    const index_t h_dim_length);

  template<MemoryType mem_type>
  static Tensor *CreateTensor(TestDeviceContext *dev_context,
                              const std::vector<index_t> &shape,
                              const DataType dt,
                              const bool is_weight,
                              const DataFormat data_format,
                              const DeviceType dev_type,
                              const OpenCLBufferType buffer_type,
                              const std::string name = "",
                              const bool alloc_buffer_hint = true,
                              const bool map_buffer_hint = false);

  static void DeleteTensorMemory(TestDeviceContext *dev_context) {
    dev_context->GetCpuDevice()->allocator()->DeleteAll();
    dev_context->GetGpuDevice()->allocator()->DeleteAll();
  }
};

class TensorDataFiller {
 public:
  void Fill(Tensor *tensor, const std::vector<float> &template_data);

  void FillImage(TestDeviceContext *dev_context,
                 OpContext *op_context,
                 Tensor *tensor,
                 const OpenCLBufferType buffer_type,
                 const int wino_blk_size,
                 const std::vector<float> &template_data);
};

class TensorDataClearer {
 public:
  int ClearTensor(Tensor *tensor);
};

class TensorDataTransformer {
 public:
  TensorDataTransformer(const MemoryType in_mem_type,
                        const MemoryType out_mem_type)
    : in_mem_type_(in_mem_type), out_mem_type_(out_mem_type) {}

  MaceStatus Transform(OpContext *context,
                       const Tensor *input,
                       const OpenCLBufferType buffer_type,
                       const int wino_block_size,
                       const bool use_internal_tensor,
                       Tensor *output);

 private:
  MemoryType in_mem_type_;
  MemoryType out_mem_type_;
};

class TensorDataChecker {
 public:
  bool Equal(const Tensor *a, const Tensor *b);

  bool EqualImage(TestDeviceContext *dev_context,
                  OpContext *op_context,
                  const Tensor *tensor,
                  const OpenCLBufferType buffer_type,
                  const int wino_block_size,
                  const float value);

  bool EqualBuffer(const Tensor *tesnor, const float value);

 private:
  template<class T>
  bool EqualInternal(const Tensor *a, const Tensor *b);
};

class TensorDebugUtils {
 public:
  static void PrintTensorInfo(const Tensor *tensor);

  static void PrintTensorData(TestDeviceContext *dev_context,
                              OpContext *op_context,
                              const Tensor *tensor,
                              const OpenCLBufferType buffer_type,
                              const int wino_blk_size);

  static void PrintTensorBufferData(const Tensor *tensor);

  static void PrintTensorImageData(TestDeviceContext *dev_context,
                                   OpContext *op_context,
                                   const Tensor *tensor,
                                   const OpenCLBufferType buffer_type,
                                   const int wino_blk_size);

  static bool CheckTensorDataFloat(const Tensor *tensor, const float value);
};

#endif  // TEST_CODL_RUN_UTILS_TENSOR_BUFFER_UTIL_H_
