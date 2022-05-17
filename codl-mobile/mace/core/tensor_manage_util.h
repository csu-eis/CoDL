
#ifndef MACE_CORE_TENSOR_MANAGE_UTIL_H_
#define MACE_CORE_TENSOR_MANAGE_UTIL_H_

#include <stdbool.h>
#include <string.h>
#include <memory>
#include <vector>

#include "mace/core/tensor.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_util.h"
#endif

namespace mace {

enum AllocateType {
  ALLOC_TYPE_GROW,
  ALLOC_TYPE_REUSE
};

enum CompareType {
  CT_BUFFER_SIZE,
  CT_BUFFER_SHAPE,
  CT_IMAGE_SHAPE
};

class TensorManageUtil {
 public:
  TensorManageUtil(Allocator *cpu_allocator)
    : in_out_size_(0),
      weight_size_(0),
      max_in_out_size_(0),
      buf_offset_(0),
      gpu_image_tensor_idx_(-1),
      cpu_allocator_(cpu_allocator),
      gpu_allocator_(nullptr) {
   cpu_buffers_ = std::vector<Buffer*>(3, nullptr);
   gpu_buffers_ = std::vector<Buffer*>(3, nullptr);
  }
  
  ~TensorManageUtil() {
    if (!tensors_.empty()) {
      DeleteTensors();
    }
    for (size_t i = 0; i < cpu_buffers_.size(); i ++) {
      if (cpu_buffers_[i]) {
        delete cpu_buffers_[i];
      }
    }
    for (size_t i = 0; i < gpu_buffers_.size(); i ++) {
      if (gpu_buffers_[i]) {
        delete gpu_buffers_[i];
      }
    }
  }

  void set_gpu_allocator(Allocator *allocator) {
    gpu_allocator_ = allocator;
  }
  
  Tensor *CreateOrReshapeTensor(
      const std::vector<index_t> &shape,
      DataType dt,
      bool is_weight,
      const DataFormat data_format,
      const DeviceType device_type,
      const std::string name,
      const bool map_gpu_buffer_hint = false,
      const AllocatorMapType map_type = AllocatorMapType::AMT_READ_WRITE,
      const index_t buffer_idx = 0,
      const AllocateType alloc_type = AllocateType::ALLOC_TYPE_GROW,
      index_t *tensor_index_ptr = nullptr);

  Tensor *CreateOrReshapePartTensor(
      const Tensor *src,
      const PartitionDim partition_dim,
      const index_t start,
      const bool reshape_hint,
      index_t *tensor_index_ptr = nullptr);
  
  Tensor *CreateOrReshapePartTensor(
      const Tensor *src,
      const PartitionDim partition_dim,
      const index_t start,
      const index_t length,
      const bool reshape_hint,
      index_t *tensor_index_ptr = nullptr);

  Tensor *CreateOrReshapeCpuConv2dWeightsTensor(
      const Tensor *src,
      const PartitionDim partition_dim,
      const index_t length,
      index_t *tensor_index_ptr = nullptr);

#ifdef MACE_ENABLE_OPENCL
  Tensor *CreateOrReshapeOpenCLImageTensor(
      const std::vector<index_t> &shape,
      DataType dt,
      bool is_weight,
      const DataFormat data_format,
      const OpenCLBufferType buffer_type,
      const std::string name,
      const AllocateType alloc_type = AllocateType::ALLOC_TYPE_GROW,
      index_t *tensor_index_ptr = nullptr);

  Tensor *CreateOrReshapePartOpenCLImageTensor(
      const Tensor *src,
      const std::vector<index_t> &part_shape,
      const bool init_reshape_hint,
      index_t *tensor_index_ptr = nullptr);

  Tensor *CreateOrReshapePartOpenCLBufferTensor(
      const Tensor *src,
      const std::vector<index_t> &part_shape,
      index_t *tensor_index_ptr);
#endif  // MACE_ENABLE_OPENCL
  
  inline Tensor *GetTensor(const index_t i) const {
    MACE_CHECK(static_cast<size_t>(i) < tensors_.size(),
               "tensor ", i, " does not in ", tensors_.size(), " tensors");
    return tensors_[i];
  }

  inline void Manage(Tensor *tensor) {
    MACE_CHECK(tensor != nullptr, "Can not manage null tensor");
    tensors_.push_back(tensor);
  }

  int DeleteTensors();

  std::string BuildStatsSizeInfo() const;
  
 private:
  void StatsSize(index_t size, bool is_weight);

  Tensor *CreateTensorV1(
      const std::vector<index_t> &shape,
      DataType dt,
      bool is_weight,
      const DataFormat data_format,
      const DeviceType device_type,
      const std::string name,
      const bool map_gpu_buffer_hint,
      const AllocatorMapType map_type,
      const index_t buffer_idx,
      const AllocateType alloc_type,
      index_t *tensor_index_ptr);

  Tensor *CreateTensorV2(
      const std::vector<index_t> &shape,
      DataType dt,
      bool is_weight,
      const DataFormat data_format,
      const DeviceType device_type,
      const std::string name,
      const bool map_gpu_buffer_hint,
      const AllocatorMapType map_type,
      const index_t buffer_idx,
      const AllocateType alloc_type,
      index_t *tensor_index_ptr);

  Tensor *ReshapeOrRecreateOpenCLImageTensor(
      Tensor *tensor,
      const std::vector<index_t> &shape,
      DataType dt,
      bool is_weight,
      const DataFormat data_format,
      const OpenCLBufferType buffer_type,
      const std::string name,
      const CompareType compare_type);

  index_t in_out_size_;
  index_t weight_size_;
  index_t max_in_out_size_;
  index_t buf_offset_;
  index_t gpu_image_tensor_idx_;
  Allocator *cpu_allocator_;
  Allocator *gpu_allocator_;
  std::vector<Buffer *> cpu_buffers_;
  std::vector<Buffer *> gpu_buffers_;
  std::vector<Tensor *> tensors_;
};

}  // namespace mace

#endif  // MACE_CORE_TENSOR_MANAGE_UTIL_H_
