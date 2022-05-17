
#include "mace/core/tensor.h"

namespace mace {

MaceStatus Tensor::ResizeImage(const std::vector<index_t> &shape,
                               const std::vector<size_t> &image_shape) {
  const std::vector<index_t> old_shape = shape_;
  shape_ = shape;
  buffer_shape_ = shape;
  image_shape_ = image_shape;
  if (buffer_ == nullptr) {
    MACE_CHECK(is_buffer_owner_);
    buffer_ = new Image(allocator_);
    return buffer_->Allocate(image_shape, dtype_);
  } else {
    MACE_CHECK(has_opencl_image(),
               name_, ": Cannot ResizeImage buffer, use Resize.");
    if (image_shape[0] > buffer_->shape()[0] ||
        image_shape[1] > buffer_->shape()[1]) {
      LOG(INFO) << "Tensor: source op " << name_
                << ", Shape"
                << " old " << VectorToString<index_t>(old_shape)
                << " new " << VectorToString<index_t>(shape)
                << ", Buffer Shape"
                << " old " << VectorToString<size_t>(buffer_->shape())
                << " new " << VectorToString<size_t>(image_shape);
      return ResizeImageV2(image_shape, allocator_);
    }
    MACE_CHECK(image_shape[0] <= buffer_->shape()[0] &&
                   image_shape[1] <= buffer_->shape()[1],
               "tensor (source op ", name_,
               "): current logical image shape:",
               image_shape[0], ", ", image_shape[1],
               " > physical image shape: ",
               buffer_->shape()[0], ", ", buffer_->shape()[1]);
    return MaceStatus::MACE_SUCCESS;
  }
}

MaceStatus Tensor::AllocateImage(
    const std::vector<size_t> &image_shape,
    Allocator *allocator) {
  is_buffer_owner_ = true;
  if (allocator != nullptr) {
    buffer_ = new Image(allocator);
  } else {
    MACE_CHECK(allocator_ != nullptr);
    buffer_ = new Image(allocator_);
  }
  return buffer_->Allocate(image_shape, dtype_);
}

MaceStatus Tensor::ResizeImageV2(
    const std::vector<size_t> &image_shape,
    Allocator *allocator) {
  image_shape_ = image_shape;
  if (buffer_ == nullptr) {
    return AllocateImage(image_shape, allocator);
  } else {
    MACE_CHECK(has_opencl_image(),
               name_, ": Cannot ResizeImage buffer, use Resize.");
    if (image_shape[0] > buffer_->shape()[0] ||
        image_shape[1] > buffer_->shape()[1]) {
      if (is_buffer_owner_) {
        delete buffer_;
      }
      return AllocateImage(image_shape, allocator);
    } else {
      return MaceStatus::MACE_SUCCESS;
    }
  }
}

void Tensor::DebugPrint(bool multi_lines) const {
  MappingGuard guard(this, AllocatorMapType::AMT_READ_ONLY);

  using namespace numerical_chars;  // NOLINT(build/namespaces)
  std::stringstream os;
  os << "Tensor " << name_ << ", dtype " << dtype_ << ", size [";
  for (index_t i : shape_) {
    os << i << ", ";
  }
  os << "], content:";

  LOG(INFO) << os.str();
  os.str("");

  for (int i = 0; i < size(); ++i) {
    if (multi_lines) {
      if (i != 0 && i % shape_.back() == 0) {
        //os << "\n";
        LOG(INFO) << os.str();
        os.str("");
      }
    }
    MACE_RUN_WITH_TYPE_ENUM(dtype_, (os << (this->data<T>()[i]) << ", "));
  }
  
  if (!multi_lines) {
    //os << "\n";
    LOG(INFO) << os.str();
    os.str("");
  }
  
  LOG(INFO) << os.str();
}

void Tensor::DebugWriteData() const {
  std::string output_name = "mace_output_" + name_;
  std::ofstream out_file(output_name, std::ios::binary);
  MappingGuard guard(this, AllocatorMapType::AMT_READ_ONLY);
  out_file.write(data<char>(), raw_size());
  out_file.flush();
  out_file.close();
  LOG(INFO) << "Write data file: " << output_name;
}

}  // namespace mace
