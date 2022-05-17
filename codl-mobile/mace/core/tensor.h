// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_CORE_TENSOR_H_
#define MACE_CORE_TENSOR_H_

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "mace/core/buffer.h"
#include "mace/core/preallocated_pooled_allocator.h"
#include "mace/core/types.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif
#include "mace/utils/logging.h"

#ifdef MACE_ENABLE_NEON
// Avoid over-bound accessing memory
#define MACE_EXTRA_BUFFER_PAD_SIZE 64
#else
#define MACE_EXTRA_BUFFER_PAD_SIZE 0
#endif

namespace mace {

#define MACE_SINGLE_ARG(...) __VA_ARGS__
#define MACE_CASE(TYPE, STATEMENTS)   \
  case DataTypeToEnum<TYPE>::value: { \
    typedef TYPE T;                   \
    STATEMENTS;                       \
    break;                            \
  }

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
#define MACE_TYPE_ENUM_SWITCH_CASE_NEON(STATEMENTS)             \
  MACE_CASE(float16_t, MACE_SINGLE_ARG(STATEMENTS))
#else
#define MACE_TYPE_ENUM_SWITCH_CASE_NEON(STATEMENTS)
#endif

#if MACE_ENABLE_OPENCL
#define MACE_TYPE_ENUM_SWITCH_CASE_OPENCL(STATEMENTS)           \
  MACE_CASE(half, MACE_SINGLE_ARG(STATEMENTS))
#else
#define MACE_TYPE_ENUM_SWITCH_CASE_OPENCL(STATEMENTS)
#endif

#define MACE_TYPE_ENUM_SWITCH(                                     \
    TYPE_ENUM, STATEMENTS, INVALID_STATEMENTS, DEFAULT_STATEMENTS) \
  switch (TYPE_ENUM) {                                             \
    MACE_CASE(float, MACE_SINGLE_ARG(STATEMENTS))                  \
    MACE_CASE(uint8_t, MACE_SINGLE_ARG(STATEMENTS))                \
    MACE_CASE(int32_t, MACE_SINGLE_ARG(STATEMENTS))                \
    MACE_TYPE_ENUM_SWITCH_CASE_NEON(STATEMENTS)                    \
    MACE_TYPE_ENUM_SWITCH_CASE_OPENCL(STATEMENTS)                  \
    case DT_INVALID:                                               \
      INVALID_STATEMENTS;                                          \
      break;                                                       \
    default:                                                       \
      DEFAULT_STATEMENTS;                                          \
      break;                                                       \
  }

// `TYPE_ENUM` will be converted to template `T` in `STATEMENTS`
#define MACE_RUN_WITH_TYPE_ENUM(TYPE_ENUM, STATEMENTS)                       \
  MACE_TYPE_ENUM_SWITCH(TYPE_ENUM, STATEMENTS, LOG(FATAL) << "Invalid type"; \
         , LOG(FATAL) << "Unknown type: " << TYPE_ENUM;)

namespace numerical_chars {
inline std::ostream &operator<<(std::ostream &os, char c) {
  return std::is_signed<char>::value ? os << static_cast<int>(c)
                                     : os << static_cast<unsigned int>(c);
}

inline std::ostream &operator<<(std::ostream &os, signed char c) {
  return os << static_cast<int>(c);
}

inline std::ostream &operator<<(std::ostream &os, unsigned char c) {
  return os << static_cast<unsigned int>(c);
}
}  // namespace numerical_chars

enum DataFormatDimIndex {
  N_NHWC = 0, H_NHWC = 1, W_NHWC = 2, C_NHWC = 3,  // NHWC
  N_NCHW = 0, C_NCHW = 1, H_NCHW = 2, W_NCHW = 3,  // NCHW
  O_OIHW = 0, I_OIHW = 1, H_OIHW = 2, W_OIHW = 3,  // OIHW
  H_HW = 0, W_HW = 1,                              // HW
};

class Tensor {
 public:
  Tensor(Allocator *alloc, DataType type,
         bool is_weight = false,
         const std::string name = "")
      : allocator_(alloc),
        dtype_(type),
        buffer_(nullptr),
        is_buffer_owner_(true),
        unused_(false),
        name_(name),
        is_weight_(is_weight),
        scale_(0.f),
        zero_point_(0),
        minval_(0.f),
        maxval_(0.f),
        data_format_(DataFormat::NONE) {}

  Tensor(BufferBase *buffer, DataType dtype,
         bool is_weight = false,
         const std::string name = "")
    : allocator_(nullptr),
      dtype_(dtype),
      buffer_(buffer),
      is_buffer_owner_(false),
      unused_(false),
      name_(name),
      is_weight_(is_weight),
      scale_(0.f),
      zero_point_(0),
      minval_(0.f),
      maxval_(0.f),
      data_format_(DataFormat::NONE) {}

  Tensor(const BufferSlice &buffer_slice,
         DataType dtype,
         bool is_weight = false,
         const std::string name = "")
      : allocator_(nullptr),
        dtype_(dtype),
        buffer_slice_(buffer_slice),
        is_buffer_owner_(false),
        unused_(false),
        name_(name),
        is_weight_(is_weight),
        scale_(0.f),
        zero_point_(0),
        minval_(0.f),
        maxval_(0.f),
        data_format_(DataFormat::NONE) {
    buffer_ = &buffer_slice_;
  }

  explicit Tensor(bool is_weight = false)
      : Tensor(GetCPUAllocator(), DT_FLOAT, is_weight) {}

  ~Tensor() {
    if (is_buffer_owner_ && buffer_ != nullptr) {
      delete buffer_;
    }
  }

  inline void set_allocator(Allocator *alloc) {
    MACE_CHECK(allocator_ == nullptr, ": tensor already has an allocator");
    allocator_ = alloc;
  }

  inline void set_is_buffer_owner(const bool is_buffer_owner) {
    is_buffer_owner_ = is_buffer_owner;
  }

  inline std::string name() const { return name_; }

  inline DataType dtype() const { return dtype_; }

  inline void SetDtype(DataType dtype) { dtype_ = dtype; }

  inline bool unused() const { return unused_; }

  inline const std::vector<index_t> &shape() const { return shape_; }

  inline const std::vector<size_t> &image_shape() const { return image_shape_; }

  inline std::vector<index_t> max_shape() const {
    if (shape_configured_.empty()) {
      return shape();
    } else {
      auto &_shape = shape();
      std::vector<index_t> max_shape(_shape.size());
      MACE_CHECK(_shape.size() == shape_configured_.size());
      for (size_t i = 0; i < shape_configured_.size(); ++i) {
        max_shape[i] = std::max(_shape[i], shape_configured_[i]);
      }
      return max_shape;
    }
  }

  inline index_t max_size() const {
    auto _max_shape = max_shape();
    return std::accumulate(_max_shape.begin(),
                           _max_shape.end(),
                           1,
                           std::multiplies<index_t>());
  }

  inline index_t raw_max_size() const { return max_size() * SizeOfType(); }

  inline void SetShapeConfigured(const std::vector<index_t> &shape_configured) {
    shape_configured_ = shape_configured;
  }

  inline const std::vector<index_t> &buffer_shape() const {
    return buffer_shape_;
  }

  inline index_t dim_size() const { return shape_.size(); }

  inline index_t dim(unsigned int index) const {
    MACE_CHECK(index < shape_.size(),
               name_, ": Dim out of range: ", index, " >= ", shape_.size());
    return shape_[index];
  }

  inline index_t size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<int64_t>());
  }

  inline index_t raw_size() const { return size() * SizeOfType(); }

  inline bool has_opencl_image() const {
    return buffer_ != nullptr && !buffer_->OnHost() &&
           buffer_->buffer_type() == core::BufferType::BT_IMAGE;
  }

  inline bool has_opencl_buffer() const {
    return buffer_ != nullptr && !buffer_->OnHost() && !has_opencl_image();
  }

  inline MemoryType memory_type() const {
    MACE_CHECK(buffer_ != nullptr, "Tensor ", name_, " is empty");
    if (buffer_->OnHost()) {
      return MemoryType::CPU_BUFFER;
    } else if (buffer_->buffer_type() == core::BufferType::BT_IMAGE) {
      return MemoryType::GPU_IMAGE;
    } else {
      return MemoryType::GPU_BUFFER;
    }
  }

  inline void set_data_format(DataFormat data_format) {
    data_format_ = data_format;
  }

  inline DataFormat data_format() const {
    return data_format_;
  }

#ifdef MACE_ENABLE_OPENCL
  inline cl::Image *opencl_image() const {
    MACE_CHECK(has_opencl_image(), name_, " do not have image");
    return static_cast<cl::Image *>(buffer_->buffer());
  }

  inline cl::Buffer *opencl_buffer() const {
    MACE_CHECK(has_opencl_buffer(), name_, " do not have opencl buffer");
    return static_cast<cl::Buffer *>(buffer_->buffer());
  }
#endif

  inline index_t buffer_offset() const { return buffer_->offset(); }

  inline const void *raw_data() const {
    MACE_CHECK(buffer_ != nullptr, "buffer is null");
    return buffer_->raw_data();
  }

  template <typename T>
  inline const T *data() const {
    MACE_CHECK_NOTNULL(buffer_);
    return buffer_->data<T>();
  }

  inline void *raw_mutable_data() {
    MACE_CHECK_NOTNULL(buffer_);
    VLOG(2) << "name " << name_;
    return buffer_->raw_mutable_data();
  }

  template <typename T>
  inline T *mutable_data() {
    MACE_CHECK_NOTNULL(buffer_);
    VLOG(2) << "name " << name_;
    return static_cast<T *>(buffer_->raw_mutable_data());
  }

  inline void MarkUnused() {
    unused_ = true;
  }

  inline void Clear() {
    MACE_CHECK_NOTNULL(buffer_);
    buffer_->Clear(raw_size());
  }

  inline void Reshape(const std::vector<index_t> &shape) {
    const std::vector<index_t> old_shape = shape_;
    shape_ = shape;
    if (has_opencl_image()) {
      if (!(raw_size() <= 4 * buffer_->size())) {
        LOG(INFO) << "Reshape:"
                  << " name " << name_
                  << " old " << VectorToString<index_t>(old_shape)
                  << " new " << VectorToString<index_t>(shape);
      }
      MACE_CHECK(raw_size() <= 4 * buffer_->size(),
                 "Must satisfy: ", raw_size(), " <= ", 4 * buffer_->size());
    } else {
      const bool size_satisfied = raw_size() <= buffer_->size();
      if (!size_satisfied) {
        LOG(INFO) << "Reshape:"
                  << " name " << name_
                  << " old " << VectorToString<index_t>(old_shape)
                  << " new " << VectorToString<index_t>(shape);
      }
      MACE_CHECK(raw_size() <= buffer_->size(),
                 "Must satisfy: ", raw_size(), " <= ", buffer_->size());
      // NOTE(fucheng): Update buffer shape.
      //buffer_shape_ = shape;
    }
  }

  inline MaceStatus Resize(const std::vector<index_t> &shape) {
    const std::vector<index_t> old_shape = shape_;
    shape_ = shape;
    buffer_shape_ = shape;
    image_shape_.clear();
    if (buffer_ != nullptr) {
      MACE_CHECK(!has_opencl_image(),
                 name_, ": Cannot resize image, use ResizeImage.");
      const index_t apply_size = raw_size()
          + ((buffer_ != &buffer_slice_) ? MACE_EXTRA_BUFFER_PAD_SIZE : 0);
      if (apply_size > buffer_->size()) {
        LOG(WARNING) << name_ << ": Resize buffer from size " << buffer_->size()
                     << " to " << apply_size
                     << " old shape " << VectorToString<index_t>(old_shape)
                     << " new shape " << VectorToString<index_t>(shape);
        MACE_CHECK(buffer_ != &buffer_slice_,
                   ": Cannot resize tensor with buffer slice");
        MaceStatus status = buffer_->Resize(apply_size);
        return status;
      }
      return MaceStatus::MACE_SUCCESS;
    } else {
      MACE_CHECK(is_buffer_owner_);
      buffer_ = new Buffer(allocator_);
      return buffer_->Allocate(raw_size() + MACE_EXTRA_BUFFER_PAD_SIZE);
    }
  }

  inline bool is_buffer_equal(const Tensor &other) {
    return buffer_ == other.buffer_;
  }

  // Make this tensor reuse other tensor's buffer.
  // This tensor has the same dtype, shape and image_shape.
  // It could be reshaped later (with image shape unchanged).
  inline void ReuseTensorBuffer(const Tensor &other) {
    if (is_buffer_owner_ && buffer_ != nullptr) {
      delete buffer_;
    }
    is_buffer_owner_ = false;
    buffer_ = other.buffer_;
    allocator_ = other.allocator_;
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    buffer_shape_ = other.buffer_shape_;
    image_shape_ = other.image_shape_;
  }

  // NOTE(fucheng): Support offset and size.
  inline void ReuseTensorBuffer(const Tensor &other,
                                const index_t data_offset,
                                const index_t data_size) {
    if (other.has_opencl_buffer() || other.has_opencl_image()) {
      return ReuseTensorBuffer(other);
    }

    if (is_buffer_owner_ && buffer_ != nullptr) {
      delete buffer_;
    }
    is_buffer_owner_ = false;
    if (other.buffer_ != &other.buffer_slice_) {
      buffer_slice_ = BufferSlice(other.buffer_,
                                  data_offset,
                                  data_size);
    } else {
      buffer_slice_ = BufferSlice(other.buffer_slice_,
                                  other.buffer_slice_.offset() + data_offset,
                                  data_size);
    }
    buffer_       = &buffer_slice_;
    allocator_    = other.allocator_;
    dtype_        = other.dtype_;
    shape_        = other.shape_;
    buffer_shape_ = other.buffer_shape_;
  }

  MaceStatus ResizeImage(const std::vector<index_t> &shape,
                         const std::vector<size_t> &image_shape);

  MaceStatus ResizeImageV2(const std::vector<size_t> &image_shape,
                           Allocator *allocator = nullptr);

  inline MaceStatus ResizeLike(const Tensor &other) {
    return ResizeLike(&other);
  }

  inline MaceStatus ResizeLike(const Tensor *other) {
    if (other->has_opencl_image()) {
      if (is_buffer_owner_ && buffer_ != nullptr && !has_opencl_image()) {
        delete buffer_;
        buffer_ = nullptr;
      }
      return ResizeImage(other->shape(), other->image_shape_);
    } else {
      if (is_buffer_owner_ && buffer_ != nullptr && has_opencl_image()) {
        delete buffer_;
        buffer_ = nullptr;
      }
      return Resize(other->shape());
    }
  }

  template <typename T>
  inline void SetValue(T val) {
    if (buffer_->IsMapped()) {
      //memset(buffer_->mutable_data<T>(), val, size());
      for (index_t i = 0; i < size(); i ++) {
        *(buffer_->mutable_data<T>() + i) = val;
      }
    } else {
      MappingGuard guard(this, AllocatorMapType::AMT_WRITE_ONLY);
      //memset(buffer_->mutable_data<T>(), val, size());
      for (index_t i = 0; i < size(); i ++) {
        *(buffer_->mutable_data<T>() + i) = val;
      }
    }
  }

  inline void CopyBytes(const void *src, size_t size) {
    if (buffer_->IsMapped()) {
      memcpy(buffer_->raw_mutable_data(), src, size);
    } else {
      MappingGuard guard(this, AllocatorMapType::AMT_WRITE_ONLY);
      memcpy(buffer_->raw_mutable_data(), src, size);
    }
  }

  template <typename T>
  inline void Copy(const T *src, index_t length) {
    MACE_CHECK(length == size(), "copy src and dst with different size.");
    CopyBytes(static_cast<const void *>(src), sizeof(T) * length);
  }

  inline void Copy(const Tensor &other, BlockFlag block_flag = BlockFlag::BF_TRUE) {
    dtype_ = other.dtype_;
    ResizeLike(other);
    MappingGuard map_other(&other, AllocatorMapType::AMT_READ_ONLY, block_flag);
    CopyBytes(other.raw_data(), other.size() * SizeOfType());
  }

  inline void ReadBytes(void *dst, size_t size) const {
    if (buffer_->IsMapped()) {
      memcpy(dst, buffer_->raw_data(), size);
    } else {
      MappingGuard guard(this, AllocatorMapType::AMT_READ_ONLY);
      memcpy(dst, buffer_->raw_data(), size);
    }
  }

  template <typename T>
  inline void Read(T *dst, index_t length) const {
    MACE_CHECK(length == size(), "read src and dst with different size.");
    ReadBytes(static_cast<void *>(dst), sizeof(T) * length);
  }

  inline size_t SizeOfType() const {
    size_t type_size = 0;
    MACE_RUN_WITH_TYPE_ENUM(dtype_, type_size = sizeof(T));
    return type_size;
  }

  inline BufferBase *UnderlyingBuffer() const { return buffer_; }

  void DebugPrint(bool multi_lines=true) const;

  void DebugWriteData() const;

  class MappingGuard {
   public:
    explicit MappingGuard(const Tensor *tensor,
                          AllocatorMapType map_type = AllocatorMapType::AMT_READ_WRITE,
                          BlockFlag block_flag = BlockFlag::BF_TRUE) : tensor_(tensor) {
      if (tensor_ != nullptr) {
        MACE_CHECK_NOTNULL(tensor_->buffer_);
        VLOG(1) << "MapBuffer: tensor_name " << tensor->name();
        tensor_->buffer_->Map(&mapped_image_pitch_, map_type, block_flag);
      }
    }

    MappingGuard(MappingGuard &&other) {
      tensor_ = other.tensor_;
      other.tensor_ = nullptr;
    }

    ~MappingGuard() {
      if (tensor_ != nullptr) tensor_->buffer_->Unmap();
    }

    inline const std::vector<size_t> &mapped_image_pitch() const {
      return mapped_image_pitch_;
    }

   private:
    const Tensor *tensor_;
    std::vector<size_t> mapped_image_pitch_;

    MACE_DISABLE_COPY_AND_ASSIGN(MappingGuard);
  };

  inline void MapBuffer(
      AllocatorMapType map_type = AllocatorMapType::AMT_READ_WRITE,
      BlockFlag block_flag = BlockFlag::BF_TRUE,
      StatsFuture *future = nullptr,
      utils::SimpleCountDownLatch *count_down_latch = nullptr) {
    MACE_CHECK_NOTNULL(buffer_);
    map_type_ = map_type;
    map_block_flag_ = block_flag;
    buffer_->Map(&mapped_image_pitch_, map_type_, map_block_flag_,
                 future, count_down_latch);

    VLOG(1) << "MapBuffer: tensor_name " << name_;
    MACE_CHECK(buffer_->IsMapped(), name_, ": Map failture.");
  }

  inline void UnmapBuffer(StatsFuture *future = nullptr) {
    VLOG(1) << "UnmapBuffer: tensor_name " << name_;
    buffer_->Unmap(future);
  }

  inline bool IsBufferMapped() const {
    return buffer_->IsMapped();
  }

  inline bool is_weight() const {
    return is_weight_;
  }

  inline float scale() const {
    return scale_;
  }

  inline int32_t zero_point() const {
    return zero_point_;
  }

  // hexagon now uses min/max instead of scale and zero
  inline float minval() const {
    return minval_;
  }

  inline float maxval() const {
    return maxval_;
  }

  inline void SetScale(float scale) {
    scale_ = scale;
  }

  inline void SetZeroPoint(int32_t zero_point) {
    zero_point_ = zero_point;
  }

  inline void SetIsWeight(bool is_weight) {
    is_weight_ = is_weight;
  }

  inline void SetMinVal(float minval) {
    minval_ = minval;
  }

  inline void SetMaxVal(float maxval) {
    maxval_ = maxval;
  }

 private:
  MaceStatus AllocateImage(const std::vector<size_t> &image_shape,
                           Allocator *allocator = nullptr);

 private:
  Allocator *allocator_;
  DataType dtype_;
  // the shape of buffer(logical)
  std::vector<index_t> shape_;
  std::vector<index_t> shape_configured_;
  std::vector<size_t> image_shape_;
  // the shape of buffer(physical storage)
  std::vector<index_t> buffer_shape_;
  BufferBase *buffer_;
  BufferSlice buffer_slice_;
  bool is_buffer_owner_;
  bool unused_;
  std::string name_;
  bool is_weight_;
  float scale_;
  int32_t zero_point_;
  float minval_;
  float maxval_;
  DataFormat data_format_;  // used for 4D input/output tensor

  AllocatorMapType map_type_;
  BlockFlag map_block_flag_;
  std::vector<size_t> mapped_image_pitch_;

  MACE_DISABLE_COPY_AND_ASSIGN(Tensor);
};

}  // namespace mace

#endif  // MACE_CORE_TENSOR_H_
