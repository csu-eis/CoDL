
#include "mace/core/memory_optimizer.h"
#include "mace/core/tensor_manage_util.h"

#define DEFAULT_MEMORY_ALIGN_SIZE 8388608 // 8 MB

//#define CODL_ENABLE_DEBUG_INFO

namespace mace {

template <typename T>
inline T RoundUp(T size, T round_up_size) {
  if (size % round_up_size != 0) {
    return (size / round_up_size + 1) * round_up_size;
  } else {
    return size;
  }
}

class TensorShapeUtils {
public:
  static void CalcImage2DShape(
      const std::vector<index_t> &shape,
      const OpenCLBufferType buffer_type,
      std::vector<size_t> &image_shape) {
    std::vector<index_t> new_shape;
    switch (buffer_type) {
      case OpenCLBufferType::IN_OUT_CHANNEL:
        if (shape.size() == 2) {
          new_shape = {shape[0], 1, 1, shape[1]};
        } else {
          MACE_CHECK(shape.size() == 4, "GPU only support 2D/4D input");
          new_shape = shape;
        }
        break;
      default:
        break;
    }
    OpenCLUtil::CalImage2DShape(new_shape, buffer_type, &image_shape);
  }

  static int CompareSize(const std::vector<index_t> &s1,
                         const std::vector<index_t> &s2) {
    index_t size1 = std::accumulate(s1.begin(), s1.end(), 1,
                                    std::multiplies<index_t>());
    index_t size2 = std::accumulate(s2.begin(), s2.end(), 1,
                                    std::multiplies<index_t>());
    if (size1 == size2) {
      return 0;
    } else if (size1 < size2) {
      return -1;
    } else {
      return 1;
    }
  }

  static int Compare(const std::vector<index_t> &s1,
                     const std::vector<index_t> &s2) {
    if (s1.size() == 4 && s2.size() == 4) {
      if (s1[0] == s2[0] && s1[1] == s2[1] &&
          s1[2] == s2[2] && s1[3] == s2[3]) {
        return 0;
      } else if (s1[0] <= s2[0] && s1[1] <= s2[1] &&
                 s1[2] <= s2[2] && s1[3] <= s2[3]) {
        return -1;
      } else if (s1[0] > s2[0] || s1[1] > s2[1] ||
                 s1[2] > s2[2] || s1[3] > s2[3]) {
        return 1;
      }
    } else if (s1.size() == 2 && s2.size() == 2) {
      if (s1[0] == s2[0] && s1[1] == s2[1]) {
        return 0;
      } else if (s1[0] <= s2[0] && s1[1] <= s2[1]) {
        return -1;
      } else if (s1[0] > s2[0] || s1[1] > s2[1]) {
        return 1;
      }
    } else if (s1.size() == 1 && s2.size() == 1) {
      if (s1[0] == s2[0]) {
        return 0;
      } else if (s1[0] < s2[0]) {
        return -1;
      } else if (s1[0] > s2[0]) {
        return 1;
      }
    }

    LOG(ERROR) << "Unknown compare result:"
               << " s1 " << VectorToString<index_t>(s1)
               << ", s2 " << VectorToString<index_t>(s2);
    MACE_NOT_IMPLEMENTED;
    return 2;
  }

  static int CompareImage2D(const std::vector<index_t> &s1,
                            const std::vector<size_t> &s2) {
    MACE_CHECK(s1.size() == 2 || s1.size() == 4);
    MACE_CHECK(s2.size() == 2 || s2.size() == 4);

#if CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "CompareImage2D:"
              << " s1 " << VectorToString<index_t>(s1)
              << ", s2 " << VectorToString<size_t>(s2);
#endif

    std::vector<size_t> img_s1;
    CalcImage2DShape(s1, OpenCLBufferType::IN_OUT_CHANNEL, img_s1);

    if (img_s1[0] == s2[0] && img_s1[1] == s2[1]) {
      return 0;
    } else if (img_s1[0] <= s2[0] && img_s1[1] <= s2[1]) {
      return -1;
    } else if (img_s1[0] > s2[0] || img_s1[1] > s2[1]) {
      return 1;
    }

    LOG(ERROR) << "Unknown compare result:"
               << " s1 " << VectorToString<index_t>(s1)
               << ", s2 " << VectorToString<size_t>(s2);
    MACE_NOT_IMPLEMENTED;
    return 2;
  }

  // Return a shape with max value of each dimension in two input shapes.
  static std::vector<index_t> BuildMaxShape(
      const std::vector<index_t> &s1,
      const std::vector<index_t> &s2) {
    MACE_CHECK(s1.size() == 4 && s2.size() == 4);
    std::vector<index_t> out_shape(4);
#define SET_OUT_SHAPE_DIM(i) \
    out_shape[i] = (s1[i] > s2[i]) ? s1[i] : s2[i];

    SET_OUT_SHAPE_DIM(0)
    SET_OUT_SHAPE_DIM(1)
    SET_OUT_SHAPE_DIM(2)
    SET_OUT_SHAPE_DIM(3)

#undef SET_OUT_SHAPE_DIM

    return out_shape;
  }

  static std::vector<size_t> BuildMaxImageShape(
      const std::vector<index_t> &s1,
      const std::vector<index_t> &s2) {
    MACE_CHECK(s1.size() == 2 || s1.size() == 4);
    MACE_CHECK(s2.size() == 2 || s2.size() == 4);

    std::vector<size_t> img_s1;
    std::vector<size_t> img_s2;
    CalcImage2DShape(s1, OpenCLBufferType::IN_OUT_CHANNEL, img_s1);
    CalcImage2DShape(s2, OpenCLBufferType::IN_OUT_CHANNEL, img_s2);

    return {static_cast<size_t>(fmax(img_s1[0], img_s2[0])),
            static_cast<size_t>(fmax(img_s1[1], img_s2[1]))};
  }

  static bool IsLess(
      const std::vector<index_t> &s1,
      const std::vector<index_t> &s2) {
      MACE_CHECK(s1.size() == 4);
      MACE_CHECK(s2.size() == 4);
    return (s1[0] <= s2[0] && s1[1] <= s2[1] &&
            s1[2] <= s2[2] && s1[3] <= s2[3]);
  }
};

void TensorSizeDebugPrint(const Tensor *tensor) {
  LOG(INFO) << "TensorSizeDebugPrint:"
            << " is_weight " << tensor->is_weight()
            << ", shape " << VectorToString<index_t>(tensor->shape())
            << ", size " << tensor->size()
            << ", raw_size " << tensor->raw_size();
}

Tensor *TensorManageUtil::CreateTensorV1(
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
    index_t *tensor_index_ptr) {
  MACE_UNUSED(buffer_idx);
  MACE_UNUSED(alloc_type);
  if (device_type == DeviceType::CPU) {
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "CreateCpuTensor: shape " << VectorToString<index_t>(shape);
#endif
    const index_t tensor_size = std::accumulate(shape.begin(),
                                                shape.end(),
                                                1,
                                                std::multiplies<index_t>());
    const index_t data_size = tensor_size * GetEnumTypeSize(dt);
    index_t require_buf_size = buf_offset_ + data_size;
    require_buf_size = RoundUp<index_t>(require_buf_size,
                                        DEFAULT_MEMORY_ALIGN_SIZE);
    Buffer *cpu_buffer = cpu_buffers_[0];
    if (cpu_buffer != nullptr) {
      if (require_buf_size > cpu_buffer->size()) {
        // Save current data.
        std::shared_ptr<Buffer> tmp_buf(new Buffer(cpu_allocator_));
        tmp_buf->Allocate(cpu_buffer->size());
        tmp_buf->Copy(cpu_buffer->raw_mutable_data(), 0, cpu_buffer->size());
        
        // Resize buffer.
        const index_t buf_size = require_buf_size + MACE_EXTRA_BUFFER_PAD_SIZE;
        cpu_buffer->Resize(buf_size);
#ifdef CODL_ENABLE_DEBUG_INFO
        LOG(INFO) << "CPU buffer resize to " << buf_size;
        LOG(INFO) << "Buffer on host " << cpu_buffer->OnHost();
#endif
        // Restore data.
        cpu_buffer->Copy(tmp_buf->raw_mutable_data(), 0, tmp_buf->size());
      }
    } else {
      const index_t buf_size = require_buf_size + MACE_EXTRA_BUFFER_PAD_SIZE;
      cpu_buffer = new Buffer(cpu_allocator_);
      cpu_buffer->Allocate(buf_size);
#ifdef CODL_ENABLE_DEBUG_INFO
      LOG(INFO) << "Create CPU buffer size " << buf_size;
      LOG(INFO) << "Buffer on host " << cpu_buffer->OnHost();
#endif
    }
          
    Tensor *t = new Tensor(
        BufferSlice(cpu_buffer, buf_offset_, data_size),
        dt, is_weight, name);
    
    t->Reshape(shape);
    t->set_data_format(data_format);
    
    if (map_gpu_buffer_hint) {
      t->MapBuffer(map_type, BlockFlag::BF_FALSE);
    }
    
    tensors_.push_back(t);
    if (tensor_index_ptr != nullptr) {
      *tensor_index_ptr = tensors_.size() - 1;
#if 1
      LOG(INFO) << "Create tensor index " << *tensor_index_ptr;
#endif
    }
    
    buf_offset_ += data_size;

#ifdef CODL_ENABLE_DEBUG_INFO
    TensorSizeDebugPrint(t);
#endif
    StatsSize(t->raw_size(), t->is_weight());
    
    return t;
  } else if (device_type == DeviceType::GPU) {
    MACE_CHECK(gpu_allocator_ != nullptr);
    MACE_CHECK(dt != DataType::DT_UINT8, "GPU buffer does not support uint8");

    Tensor *t = new Tensor(gpu_allocator_, dt, is_weight, name);

    t->set_data_format(data_format);
    // Resize and allocate buffer.
    t->Resize(shape);
    // Map GPU buffer if set the hint.
    if (map_gpu_buffer_hint) {
      t->MapBuffer(map_type, BlockFlag::BF_FALSE);
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "Map created tensor name " << name;
#endif
    }

    tensors_.push_back(t);
    if (tensor_index_ptr != nullptr) {
      *tensor_index_ptr = tensors_.size() - 1;
#if 1
      LOG(INFO) << "Create tensor index " << *tensor_index_ptr;
#endif
    }
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "Create GPU buffer with size " << t->raw_size();
#endif

#ifdef CODL_ENABLE_DEBUG_INFO
    TensorSizeDebugPrint(t);
#endif
    StatsSize(t->raw_size(), t->is_weight());

    return t;
  } else {
    LOG(ERROR) << "Not supported tensor device type " << device_type;
    MACE_NOT_IMPLEMENTED;
    return nullptr;
  }
}

Tensor *TensorManageUtil::CreateTensorV2(
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
    index_t *tensor_index_ptr) {
  Buffer **buffer = nullptr;
  Allocator *allocator = nullptr;
  if (device_type == DeviceType::CPU) {
    buffer = &cpu_buffers_[buffer_idx];
    allocator = cpu_allocator_;
  } else if (device_type == DeviceType::GPU) {
    buffer = &gpu_buffers_[buffer_idx];
    allocator = gpu_allocator_;
  } else {
    LOG(ERROR) << "Not supported tensor device type " << device_type;
    MACE_NOT_IMPLEMENTED;
    return nullptr;
  }

  const index_t tensor_size = std::accumulate(shape.begin(),
                                              shape.end(),
                                              1,
                                              std::multiplies<index_t>());
  const index_t data_size = tensor_size * GetEnumTypeSize(dt);
  index_t buffer_offset = 0;
  if (alloc_type == AllocateType::ALLOC_TYPE_GROW && (*buffer) != nullptr) {
    buffer_offset = (*buffer)->size();
  }
  
  index_t require_buf_size = buffer_offset + data_size;
  require_buf_size = RoundUp<index_t>(require_buf_size,
                                      DEFAULT_MEMORY_ALIGN_SIZE);
  if ((*buffer) != nullptr) {
    if (require_buf_size > (*buffer)->size()) {
      // Save current data.
      std::shared_ptr<Buffer> tmp_buf(new Buffer(cpu_allocator_));
      tmp_buf->Allocate((*buffer)->size());
      
      if (!(*buffer)->OnHost()) {
        (*buffer)->Map(nullptr);
      }
      
      tmp_buf->Copy((*buffer)->raw_mutable_data(), 0, (*buffer)->size());
      
      if (!(*buffer)->OnHost()) {
        (*buffer)->Unmap();
      }
      
      // Resize buffer.
      const index_t buf_size = require_buf_size + MACE_EXTRA_BUFFER_PAD_SIZE;
      (*buffer)->Resize(buf_size);
#ifdef CODL_ENABLE_DEBUG_INFO
      LOG(INFO) << "Buffer resize to " << buf_size;
      LOG(INFO) << "Buffer on host " << (*buffer)->OnHost();
#endif
      // Restore data.
      if (!(*buffer)->OnHost()) {
        (*buffer)->Map(nullptr);
      }
      
      (*buffer)->Copy(tmp_buf->raw_mutable_data(), 0, tmp_buf->size());
      
      if (!(*buffer)->OnHost()) {
        (*buffer)->Unmap();
      }
    }
  } else {
    const index_t buf_size = require_buf_size + MACE_EXTRA_BUFFER_PAD_SIZE;
    *buffer = new Buffer(allocator);
    (*buffer)->Allocate(buf_size);
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "Create buffer size " << buf_size;
    LOG(INFO) << "Buffer on host " << (*buffer)->OnHost();
#endif
  }
  
  Tensor *t = new Tensor(BufferSlice(*buffer, buffer_offset, data_size),
                         dt, is_weight, name);
  t->Reshape(shape);
  t->set_data_format(data_format);
  if (map_gpu_buffer_hint) {
    t->MapBuffer(map_type, BlockFlag::BF_FALSE);
  }
  
  tensors_.push_back(t);
  if (tensor_index_ptr != nullptr) {
    *tensor_index_ptr = tensors_.size() - 1;
#if 1
    LOG(INFO) << "Create tensor index " << *tensor_index_ptr;
#endif
  }

#ifdef CODL_ENABLE_DEBUG_INFO
  TensorSizeDebugPrint(t);
#endif
  StatsSize(t->raw_size(), t->is_weight());
    
  return t;
}

Tensor *TensorManageUtil::CreateOrReshapeTensor(
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
    index_t *tensor_index_ptr) {
#ifdef CODL_ENABLE_DEBUG_INFO
  LOG(INFO) << "Create or reshape a tensor of which the shape size is "
            << VectorToString<index_t>(shape);
#endif

  index_t tensor_index = -1;
  if (tensor_index_ptr != nullptr) {
    tensor_index = *tensor_index_ptr;
    //LOG(INFO) << "Get tensor index " << *tensor_index_ptr;
  }
  
  if (tensor_index >= 0) {
    Tensor *tensor = tensors_[tensor_index];
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "CompareShape:"
              << " old " << VectorToString<index_t>(tensor->shape())
              << ", new " << VectorToString<index_t>(shape);
#endif
    if (tensor->shape() != shape) {
#ifdef CODL_ENABLE_DEBUG_INFO
      LOG(INFO) << "ReshapeTensor:"
                << " old " << VectorToString<index_t>(tensor->shape())
                << ", new " << VectorToString<index_t>(shape);
#endif
      tensor->Reshape(shape);
    }

    return tensor;
  } else {
    return CreateTensorV1(shape, dt, is_weight, data_format, device_type,
                          name, map_gpu_buffer_hint, map_type, buffer_idx,
                          alloc_type, tensor_index_ptr);
  }
}

Tensor *TensorManageUtil::CreateOrReshapePartTensor(
    const Tensor *src,
    const PartitionDim partition_dim,
    const index_t start,
    const index_t length,
    const bool reshape_hint,
    index_t *tensor_index_ptr) {
  if (!src->is_weight()) {
    MACE_CHECK(src->data_format() == DataFormat::NHWC);
  } else {
    MACE_CHECK(src->data_format() == DataFormat::OIHW ||
               src->data_format() == DataFormat::NONE);
  }
  
  if (src->has_opencl_image()) {
    MACE_CHECK(start == 0);
  }
  
  index_t tensor_index = -1;
  if (tensor_index_ptr != nullptr) {
    tensor_index = *tensor_index_ptr;
    //LOG(INFO) << "Get tensor index " << tensor_index;
  }
  
  if (tensor_index > 0) {
    Tensor *tensor = tensors_[tensor_index];
    if (reshape_hint) {
      bool is_shape_changed = false;
      std::vector<index_t> part_shape;
      if (partition_dim == DIM_INPUT_HEIGHT) {
        if (!src->is_weight()) {
          is_shape_changed = tensor->dim(H_NHWC) != length;
          part_shape.push_back(src->dim(N_NHWC));
          part_shape.push_back(length);
          part_shape.push_back(src->dim(W_NHWC));
          part_shape.push_back(src->dim(C_NHWC));
        } else {
          is_shape_changed = false;
          part_shape = src->shape();
        }
      } else if (partition_dim == DIM_OUTPUT_CHANNEL) {
        if (!src->is_weight()) {
          if (src->shape().size() == 4) {
            is_shape_changed = tensor->dim(C_NHWC) != length;
            part_shape.push_back(src->dim(N_NHWC));
            part_shape.push_back(src->dim(H_NHWC));
            part_shape.push_back(src->dim(W_NHWC));
            part_shape.push_back(length);
          } else if (src->shape().size() == 2) {
            is_shape_changed = tensor->dim(H_HW) != length;
            part_shape.push_back(src->dim(length));
            part_shape.push_back(src->dim(W_HW));
          }
        } else {
          if (src->shape().size() == 4) {
            is_shape_changed = tensor->dim(O_OIHW) != length;
            part_shape.push_back(length);
            part_shape.push_back(src->dim(I_OIHW));
            part_shape.push_back(src->dim(H_OIHW));
            part_shape.push_back(src->dim(W_OIHW));
          } else if (src->shape().size() == 2) {
            is_shape_changed = tensor->dim(H_HW) != length;
            part_shape.push_back(length);
            part_shape.push_back(src->dim(W_HW));
          } else if (src->shape().size() == 1) {
            is_shape_changed = tensor->dim(0) != length;
            part_shape.push_back(length);
          }
        }
      } else {
        LOG(ERROR) << "Unsupported partition dimension";
        return nullptr;
      }

      if (is_shape_changed) {
#ifdef CODL_ENABLE_DEBUG_INFO
        LOG(INFO) << "ReshapePartialTensor:"
                  << " old " << VectorToString<index_t>(tensor->shape())
                  << ", new " << VectorToString<index_t>(part_shape);
#endif
        tensor->Reshape(part_shape);
      }
    }
    return tensor;
  } else {
    std::vector<index_t> part_shape;
    index_t inner_size = 0;
    if (partition_dim == DIM_INPUT_HEIGHT) {
      if (!src->is_weight()) {
        part_shape.push_back(src->dim(N_NHWC));
        part_shape.push_back(length);
        part_shape.push_back(src->dim(W_NHWC));
        part_shape.push_back(src->dim(C_NHWC));
        inner_size = src->dim(W_NHWC) * src->dim(C_NHWC)
            * GetEnumTypeSize(src->dtype());
      } else {
        if (src->shape().size() == 4) {
          part_shape.push_back(src->dim(O_OIHW));
          part_shape.push_back(src->dim(I_OIHW));
          part_shape.push_back(src->dim(H_OIHW));
          part_shape.push_back(src->dim(W_OIHW));
          inner_size = src->dim(I_OIHW) * src->dim(H_OIHW) * src->dim(W_OIHW)
              * GetEnumTypeSize(src->dtype());
        } else if (src->shape().size() == 1) {
          part_shape.push_back(src->dim(0));
          inner_size = 1 * GetEnumTypeSize(src->dtype());
        }
      }
    } else if (partition_dim == DIM_OUTPUT_CHANNEL) {
      if (!src->is_weight()) {
        if (src->shape().size() == 4) {
          part_shape.push_back(src->dim(N_NHWC));
          part_shape.push_back(src->dim(H_NHWC));
          part_shape.push_back(src->dim(W_NHWC));
          part_shape.push_back(length);
          inner_size = src->dim(H_NHWC) * src->dim(W_NHWC)
              * GetEnumTypeSize(src->dtype());
        } else if (src->shape().size() == 2) {
          part_shape.push_back(length);
          part_shape.push_back(src->dim(W_HW));
          inner_size = src->dim(W_HW) * GetEnumTypeSize(src->dtype());
        } else {
          MACE_NOT_IMPLEMENTED;
        }
      } else {
        if (src->shape().size() == 4) {
          part_shape.push_back(length);
          part_shape.push_back(src->dim(I_OIHW));
          part_shape.push_back(src->dim(H_OIHW));
          part_shape.push_back(src->dim(W_OIHW));
          inner_size = src->dim(I_OIHW) * src->dim(H_OIHW) * src->dim(W_OIHW)
              * GetEnumTypeSize(src->dtype());
        } else if (src->shape().size() == 2) {
          part_shape.push_back(src->dim(length));
          part_shape.push_back(src->dim(W_HW));
          inner_size = src->dim(W_HW) * GetEnumTypeSize(src->dtype());
        } else {
          part_shape.push_back(length);
          inner_size = GetEnumTypeSize(src->dtype());
        }
      }
    } else {
      LOG(ERROR) << "Unsupported partition dimension";
      return nullptr;
    }

#if 0
    LOG(INFO) << "CreatePartialTensor:"
              << " src " << VectorToString<index_t>(src->shape())
              << ", dst " << VectorToString<index_t>(part_shape)
              << ", dtype " << src->dtype()
              << ", dtype_size " << GetEnumTypeSize(src->dtype());
#endif

    int result = TensorShapeUtils::Compare(part_shape, src->shape());
    if (result > 0) {
      if (src->has_opencl_image()) {
        return CreateOrReshapePartOpenCLImageTensor(
            src, part_shape, reshape_hint, tensor_index_ptr);
      } else if (src->has_opencl_buffer()) {
        //MACE_NOT_IMPLEMENTED;
        return CreateOrReshapePartOpenCLBufferTensor(
            src, part_shape, tensor_index_ptr);
      }
    }

    index_t offset = start * inner_size;
    index_t size = length * inner_size;
    
    Allocator *allocator = nullptr;
    Tensor *dst = new Tensor(allocator, src->dtype(), src->is_weight(),
                             std::string("part_tensor"));
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "PartitionParameter:"
              << " start " << start
              << ", length " << length
              << ", offset " << offset
              << ", size " << size;
#endif
    dst->ReuseTensorBuffer(*src, offset, size);
    if (reshape_hint) {
      int result = TensorShapeUtils::Compare(part_shape, src->shape());
      MACE_CHECK(result <= 0, "Part shape ", VectorToString<index_t>(part_shape),
          " should be less than full shape ", VectorToString<index_t>(src->shape()));
#ifdef CODL_ENABLE_DEBUG_INFO
      LOG(INFO) << "ReshapePartialTensor:"
                << " old " << VectorToString<index_t>(dst->shape())
                << ", new " << VectorToString<index_t>(part_shape);
#endif

      dst->Reshape(part_shape);
    }
    dst->set_data_format(src->data_format());
    
    tensors_.push_back(dst);
    if (tensor_index_ptr != nullptr) {
      *tensor_index_ptr = tensors_.size() - 1;
#if 1
    LOG(INFO) << "Create tensor index " << *tensor_index_ptr;
#endif
    }
    
    return dst;
  }
}

Tensor *TensorManageUtil::CreateOrReshapePartTensor(
    const Tensor *src,
    const PartitionDim partition_dim,
    const index_t start,
    const bool reshape_hint,
    index_t *tensor_index_ptr) {
  index_t length;
  if (!src->is_weight()) {
    MACE_CHECK(src->data_format() == DataFormat::NHWC);
    switch (partition_dim) {
      case DIM_INPUT_HEIGHT:
        length = src->dim(H_NHWC) - start;
        break;
      case DIM_OUTPUT_CHANNEL:
        if (src->dim_size() == 2) {
          length = src->dim(N_NHWC) - start;
        } else {
          length = src->dim(C_NHWC) - start;
        }
        break;
      default:
        LOG(ERROR) << "Unsupported partition dimension";
        return nullptr;
    }
  } else {
    MACE_CHECK(src->data_format() == DataFormat::OIHW ||
               src->data_format() == DataFormat::NONE);
    length = src->dim(O_OIHW) - start;
  }
  
  return CreateOrReshapePartTensor(src,
                                   partition_dim,
                                   start,
                                   length,
                                   reshape_hint,
                                   tensor_index_ptr);
}

#ifdef MACE_ENABLE_OPENCL
static MemoryBlock CreateImageMemoryBlock(
    const std::vector<int64_t> &shape,
    const OpenCLBufferType buffer_type) {
  std::vector<size_t> image_shape;
  TensorShapeUtils::CalcImage2DShape(shape, buffer_type, image_shape);
  
  MemoryBlock block;
  block.set_x(image_shape[0]);
  block.set_y(image_shape[1]);
  return block;
}

static Tensor *CreateOpenCLImageTensor(
    const std::vector<int64_t> &shape,
    const OpenCLBufferType buffer_type,
    Allocator *allocator,
    DataType dt,
    const DataFormat data_format,
    bool is_weight,
    const std::string name) {
  MemoryBlock mem_block = CreateImageMemoryBlock(shape, buffer_type);

  BufferBase *image_buf = new Image(allocator);
  MaceStatus status = image_buf->Allocate({static_cast<size_t>(mem_block.x()),
                                           static_cast<size_t>(mem_block.y())},
                                          dt);
  if (status != MaceStatus::MACE_SUCCESS) {
      LOG(ERROR) << "Allocate image buffer failed";
      return nullptr;
  }

#ifdef CODL_ENABLE_DEBUG_INFO
  LOG(INFO) << "CreateOpenCLImageTensor:"
            << " name " << name
            << ", shape " << VectorToString<index_t>(shape)
            << ", mem_block " << VectorToString<size_t>({
                  static_cast<size_t>(mem_block.x()),
                  static_cast<size_t>(mem_block.y())});
#endif

  Tensor *tensor = new Tensor(image_buf, dt, is_weight, name);
  tensor->Reshape(shape);
  tensor->set_data_format(data_format);
  tensor->set_is_buffer_owner(true);
  
  return tensor;
}

Tensor *TensorManageUtil::ReshapeOrRecreateOpenCLImageTensor(
    Tensor *tensor,
    const std::vector<index_t> &shape,
    DataType dt,
    bool is_weight,
    const DataFormat data_format,
    const OpenCLBufferType buffer_type,
    const std::string name,
    const CompareType compare_type) {
  int ret = 0;
  if (compare_type == CompareType::CT_BUFFER_SIZE) {
    ret = TensorShapeUtils::CompareSize(shape, tensor->shape());
  } else if (compare_type == CompareType::CT_BUFFER_SHAPE) {
    ret = TensorShapeUtils::Compare(shape, tensor->shape());
  } else if (compare_type == CompareType::CT_IMAGE_SHAPE) {
    ret = TensorShapeUtils::CompareImage2D(shape, tensor->image_shape());
  } else {
    LOG(ERROR) << "Unsupported compare type: " << compare_type;
    MACE_CHECK(false);
  }

  if (ret > 0) {
    LOG(INFO) << "Larger shape " << VectorToString<index_t>(shape)
              << " vs. " << VectorToString<index_t>(tensor->shape())
              <<", delete and create tensor with larger shape automatically";
    std::vector<index_t> max_shape =
        TensorShapeUtils::BuildMaxShape(shape, tensor->shape());
    delete tensor;
    return CreateOpenCLImageTensor(max_shape,
                                   buffer_type,
                                   gpu_allocator_,
                                   dt,
                                   data_format,
                                   is_weight,
                                   name);
    return tensor;
  }

  tensor->Reshape(shape);
  return tensor;
}

Tensor *TensorManageUtil::CreateOrReshapeOpenCLImageTensor(
    const std::vector<index_t> &shape,
    DataType dt,
    bool is_weight,
    const DataFormat data_format,
    const OpenCLBufferType buffer_type,
    const std::string name,
    const AllocateType alloc_type,
    index_t *tensor_index_ptr) {
  index_t tensor_index = -1;
  if (tensor_index_ptr != nullptr) {
    tensor_index = *tensor_index_ptr;
    //LOG(INFO) << "Get tensor index " << *tensor_index_ptr;
  }
    
  if (tensor_index >= 0) {
    Tensor *tensor = tensors_[tensor_index];
    if (tensor->shape() != shape) {
#ifdef CODL_ENABLE_DEBUG_INFO
      LOG(INFO) << "ReshapeOpenCLTensor:"
                << " name " << name
                << ", old " << VectorToString<index_t>(tensor->shape())
                << ", new " << VectorToString<index_t>(shape);
#endif
      tensor = ReshapeOrRecreateOpenCLImageTensor(tensor,
                                                  shape,
                                                  dt,
                                                  is_weight,
                                                  data_format,
                                                  buffer_type,
                                                  name,
                                                  CompareType::CT_BUFFER_SHAPE);
      tensors_[tensor_index] = tensor;
      return tensor;
    } else {
      return tensor;
    }
  } else {
    Tensor *tensor = nullptr;
    if (alloc_type == AllocateType::ALLOC_TYPE_REUSE &&
        gpu_image_tensor_idx_ >= 0) {
      tensor = tensors_[gpu_image_tensor_idx_];
      tensor = ReshapeOrRecreateOpenCLImageTensor(tensor,
                                                  shape,
                                                  dt,
                                                  is_weight,
                                                  data_format,
                                                  buffer_type,
                                                  name,
                                                  CompareType::CT_IMAGE_SHAPE);
      tensors_[gpu_image_tensor_idx_] = tensor;
      if (tensor_index_ptr != nullptr) {
        *tensor_index_ptr = gpu_image_tensor_idx_;
      }
    } else {
#ifdef CODL_ENABLE_DEBUG_INFO
      LOG(INFO) << "CreateOpenCLTensor:"
                << " name " << name
                << ", shape " << VectorToString<index_t>(shape);
#endif
      tensor = CreateOpenCLImageTensor(shape,
                                       buffer_type,
                                       gpu_allocator_,
                                       dt,
                                       data_format,
                                       is_weight,
                                       name);

      tensors_.push_back(tensor);
      index_t tensor_idx = tensors_.size() - 1;
      if (tensor_index_ptr != nullptr) {
        *tensor_index_ptr = tensor_idx;
#if 1
        LOG(INFO) << "Create tensor index " << *tensor_index_ptr;
#endif
      }

      if (alloc_type == AllocateType::ALLOC_TYPE_REUSE) {
        gpu_image_tensor_idx_ = tensor_idx;
      }
    }

    return tensor;
  }
}

Tensor *TensorManageUtil::CreateOrReshapePartOpenCLImageTensor(
    const Tensor *src,
    const std::vector<index_t> &part_shape,
    const bool init_reshape_hint,
    index_t *tensor_index_ptr) {
  MACE_CHECK(src->data_format() == DataFormat::NHWC ||
             src->data_format() == DataFormat::OIHW,
             "we got data format ", static_cast<int>(src->data_format()));
  MACE_CHECK(src->has_opencl_image());
  
  index_t tensor_index = -1;
  if (tensor_index_ptr != nullptr) {
    tensor_index = *tensor_index_ptr;
  }
  
  if (tensor_index > 0) {
    Tensor *tensor = tensors_[tensor_index];
    if (init_reshape_hint) {
#ifdef CODL_ENABLE_DEBUG_INFO
      LOG(INFO) << "ReshapePartOpenCLTensor:"
                << " old " << VectorToString<index_t>(tensor->shape())
                << ", new " << VectorToString<index_t>(part_shape)
                << ", src_shape " << VectorToString<index_t>(src->shape());
#endif
      if (TensorShapeUtils::Compare(part_shape, src->shape()) > 0) {
        // Use max shape to create memory block.
        std::vector<index_t> max_shape
            = TensorShapeUtils::BuildMaxShape(part_shape, src->shape());
        MemoryBlock mem_block = CreateImageMemoryBlock(
            max_shape, OpenCLBufferType::IN_OUT_CHANNEL);
        const std::vector<size_t> image_shape = {
            static_cast<size_t>(mem_block.x()),
            static_cast<size_t>(mem_block.y())};
        Tensor *src_tmp = const_cast<Tensor *>(src);
        src_tmp->ResizeImageV2(image_shape, gpu_allocator_);
#ifdef CODL_ENABLE_DEBUG_INFO
        LOG(INFO) << "Resize source tensor image shape to "
                  << VectorToString<size_t>(image_shape);
#endif
      }

      if (!tensor->is_buffer_equal(*src)) {
        tensor->ReuseTensorBuffer(*src, 0, 0);
      }

      tensor->Reshape(part_shape);
    }
    
    return tensor;
  } else {
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "CreatePartOpenCLTensor:"
              << " src " << VectorToString<index_t>(src->shape())
              << ", dst " << VectorToString<index_t>(part_shape)
              << ", dtype " << src->dtype()
              << ", dtype size " << GetEnumTypeSize(src->dtype());
#endif
    Tensor *dst = new Tensor((Allocator*) nullptr, src->dtype(),
                             src->is_weight(),
                             std::string("part_tensor"));
    dst->ReuseTensorBuffer(*src, 0, 0);
    if (init_reshape_hint) {
      if (TensorShapeUtils::Compare(part_shape, src->shape()) > 0) {
        // Use max shape to create memory block.
        std::vector<index_t> max_shape
            = TensorShapeUtils::BuildMaxShape(part_shape, src->shape());
#ifdef CODL_ENABLE_DEBUG_INFO
        LOG(INFO) << "Max shape " << VectorToString<index_t>(max_shape);
#endif
        MemoryBlock mem_block = CreateImageMemoryBlock(
            max_shape, OpenCLBufferType::IN_OUT_CHANNEL);
        const std::vector<size_t> image_shape = {
            static_cast<size_t>(mem_block.x()),
            static_cast<size_t>(mem_block.y())};
        Tensor *src_tmp = const_cast<Tensor *>(src);
        src_tmp->ResizeImageV2(image_shape, gpu_allocator_);
#ifdef CODL_ENABLE_DEBUG_INFO
        LOG(INFO) << "Resize source tensor image shape to "
                  << VectorToString<size_t>(image_shape);
#endif

        dst->ReuseTensorBuffer(*src, 0, 0);
      }
#ifdef CODL_ENABLE_DEBUG_INFO
      LOG(INFO) << "ReshapePartOpenCLTensor:"
                << " old " << VectorToString<index_t>(dst->shape())
                << ", new " << VectorToString<index_t>(part_shape);
#endif

      dst->Reshape(part_shape);
    }
    
    dst->set_data_format(src->data_format());
    
    tensors_.push_back(dst);
    if (tensor_index_ptr != nullptr) {
      *tensor_index_ptr = tensors_.size() - 1;
#if 1
      LOG(INFO) << "Create tensor index " << *tensor_index_ptr;
#endif
    }
    
    return dst;
  }
}

Tensor *TensorManageUtil::CreateOrReshapePartOpenCLBufferTensor(
    const Tensor *src,
    const std::vector<index_t> &part_shape,
    index_t *tensor_index_ptr) {
  //LOG(INFO) << "ReshapePartOpenCLBufferTensor: start";
  MACE_CHECK(src != nullptr);
  //LOG(INFO) << "ReshapePartOpenCLBufferTensor: check data format";
  MACE_CHECK(src->data_format() == DataFormat::NHWC ||
             src->data_format() == DataFormat::OIHW,
             "we got data format ", static_cast<int>(src->data_format()));
  //LOG(INFO) << "ReshapePartOpenCLBufferTensor: check has opencl buffer";
  MACE_CHECK(src->has_opencl_buffer());

#ifdef CODL_ENABLE_DEBUG_INFO
  LOG(INFO) << "ReshapePartOpenCLBufferTensor:"
            << " old " << VectorToString<index_t>(src->shape())
            << ", new " << VectorToString<index_t>(part_shape)
            << ", src_shape " << VectorToString<index_t>(src->shape());
#endif
  if (TensorShapeUtils::Compare(part_shape, src->shape()) > 0) {
    // Use max shape to create buffer.
    std::vector<index_t> max_shape
        = TensorShapeUtils::BuildMaxShape(part_shape, src->shape());
    Tensor *src_tmp = const_cast<Tensor *>(src);
    src_tmp->Resize(max_shape);
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "Resize source tensor buffer shape to "
              << VectorToString<index_t>(max_shape);
#endif
  }

  //LOG(INFO) << "Create or get tensor";
  Tensor *tensor = nullptr;
  if (tensor_index_ptr != nullptr) {
    tensor = tensors_[*tensor_index_ptr];
  } else {
    tensor = new Tensor((Allocator*) nullptr, src->dtype(),
                         src->is_weight(),
                         std::string("part_tensor"));
    tensors_.push_back(tensor);
  }

  //LOG(INFO) << "Check if buffer is equal";
  if (!tensor->is_buffer_equal(*src)) {
    //LOG(INFO) << "Reuse source tensor buffer";
    tensor->ReuseTensorBuffer(*src, 0, 0);
  }

  //LOG(INFO) << "Reshape tensor";
  tensor->Reshape(part_shape);

  return tensor;
}
#endif  // MACE_ENABLE_OPENCL

Tensor *TensorManageUtil::CreateOrReshapeCpuConv2dWeightsTensor(
    const Tensor *src,
    const PartitionDim partition_dim,
    const index_t length,
    index_t *tensor_index_ptr) {
  MACE_CHECK(src != nullptr);

  if (partition_dim == PartitionDim::DIM_OUTPUT_CHANNEL) {
    MACE_CHECK(length <= src->dim(O_OIHW),
               "partial output channels should be less than full output channels.");
    const index_t si = src->dim(O_OIHW) - length;
    return CreateOrReshapePartTensor(src,
                                     partition_dim,
                                     si,
                                     length,
                                     true,
                                     tensor_index_ptr);
  } else if (partition_dim == PartitionDim::DIM_INPUT_HEIGHT) {
    return const_cast<Tensor *>(src);
  } else {
    LOG(ERROR) << "Unsupported partition dimension.";
    MACE_NOT_IMPLEMENTED;
  }

  return nullptr;
}

int TensorManageUtil::DeleteTensors() {
  while (!tensors_.empty()) {
    Tensor *t = tensors_.back();
    tensors_.pop_back();
    delete t;
  }
  
  buf_offset_ = 0;
  
  return 0;
}

std::string TensorManageUtil::BuildStatsSizeInfo() const {
  std::stringstream stream;
  stream << "in_out_size " << in_out_size_
         << ", weight_size " << weight_size_
         << ", max_in_out_size " << max_in_out_size_;
  return stream.str();
}

void TensorManageUtil::StatsSize(index_t size, bool is_weight) {
  if (is_weight) {
    weight_size_ += size;
  } else {
    in_out_size_ += size;
    if (size > max_in_out_size_) {
      max_in_out_size_ = size;
    }
  }
}

}  // namespace mace

#ifdef CODL_ENABLE_DEBUG_INFO
#undef CODL_ENABLE_DEBUG_INFO
#endif
