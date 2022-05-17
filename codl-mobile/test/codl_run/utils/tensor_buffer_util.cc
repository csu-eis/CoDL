
#include <assert.h>
#include <numeric>
#include <stdio.h>
#include "mace/ops/common/transpose.h"
#include "mace/ops/opencl/buffer_transformer.h"
#include "test/codl_run/utils/tensor_buffer_util.h"

constexpr int kMaxPrintDataCount = 8192;

const std::string ShapeToString(const std::vector<index_t> &shape) {
  if (shape.empty()) {
    return "";
  }

  std::stringstream stream;
  stream << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    stream << shape[i];
    if (i != shape.size() - 1) {
      stream << ",";
    }
  }
  stream << "]";

  return stream.str();
}

void PrintDataTemplateFloat(const std::vector<float> &data_template) {
  printf("===== Data Template =====\n");
  printf("[");
  for (unsigned int i = 0; i < data_template.size(); i ++) {
    printf("%.1f,", data_template.at(i));
  }
  printf("]\n");
}

void PrintTempDataFloat(const float *data, index_t size) {
  printf("===== Temp Data =====\n");
  printf("[");
  for (int i = 0; i < size; i ++) {
    printf("%.1f,", data[i]);
  }
  printf("]\n");
}

bool IsGPUTensor(const Tensor *tensor) {
  return (tensor->has_opencl_image() || tensor->has_opencl_buffer());
}

Tensor* TensorUtils::CreateBufferTensor(
    TestDeviceContext *device_context,
    const std::vector<index_t> &shape,
    const DataType dt,
    const bool is_weight,
    const DataFormat data_format,
    const DeviceType device_type,
    const std::string name,
    const bool alloc_buffer_hint,
    const bool map_buffer_hint) {
  Tensor *tensor = nullptr;
  
  if (device_type == DeviceType::CPU) {
    tensor = new Tensor(device_context->GetCpuDevice()->allocator(),
                        dt, is_weight, name);
  }
#ifdef MACE_ENABLE_OPENCL
  else if (device_type == DeviceType::GPU) {
    tensor = new Tensor(device_context->GetGpuDevice()->allocator(),
                        dt, is_weight, name);
  }
#endif
  else {
    fprintf(stderr, "Error: Not support device type\n");
    return nullptr;
  }
    
  tensor->set_data_format(data_format);
  tensor->set_is_buffer_owner(true);
  
  // NOTE(fucheng): Resize to allocate memory for tensor.
  if (alloc_buffer_hint) {
    tensor->Resize(shape);
  }

  if (device_type == DeviceType::GPU && map_buffer_hint) {
    tensor->MapBuffer();
  }
  
  return tensor;
}

MemoryBlock CreateImageMemoryBlock(
    const std::vector<index_t> &shape,
    OpenCLBufferType buffer_type) {
  MemoryBlock block;
  std::vector<index_t> tmp_shape = shape;
  std::vector<size_t> image_shape;
  
  switch (buffer_type) {
    case OpenCLBufferType::IN_OUT_CHANNEL:
      if (shape.size() == 2) {
        tmp_shape = {shape[0], 1, 1, shape[1]};
      } else {
        MACE_CHECK(shape.size() == 4) << "GPU only support 2D/4D input";
      }
      break;
    default:
      break;
  }
  
  OpenCLUtil::CalImage2DShape(tmp_shape, buffer_type, &image_shape, 0);
  block.set_x(image_shape[0]);
  block.set_y(image_shape[1]);
  return block;
}

#ifdef MACE_ENABLE_OPENCL
Tensor* TensorUtils::CreateGPUImageTensor(
    TestDeviceContext *device_context,
    const std::vector<index_t> &shape,
    const DataType dt,
    const bool is_weight,
    const DataFormat data_format,
    const OpenCLBufferType buffer_type,
    const std::string name) {
  Device *device = device_context->GetGpuDevice();
  
  MemoryBlock mem_block = CreateImageMemoryBlock(shape, buffer_type);
  
  BufferBase *image_buf = new Image(device->allocator());
  MaceStatus status = image_buf->Allocate(
      {static_cast<size_t>(mem_block.x()),
       static_cast<size_t>(mem_block.y())}, dt);
  if (status != MaceStatus::MACE_SUCCESS) {
    LOG(ERROR) << "Image buffer allocate failed,"
               << " tensor_name " << name
               << " alloc_shape " << VectorToString<index_t>(shape);
    return nullptr;
  }
  
  Tensor* tensor = new Tensor(image_buf, dt, is_weight, name);
  tensor->Reshape(shape);
  tensor->set_data_format(data_format);
  tensor->set_is_buffer_owner(true);
  
#if 0
  LOG(INFO) << "Create tensor using GPU image shape ["
            << tensor->UnderlyingBuffer()->shape()[0] << ","
            << tensor->UnderlyingBuffer()->shape()[1] << "]";
#endif
  
  return tensor;
}
#endif  // MACE_ENABLE_OPENCL

Tensor* TensorUtils::CreatePartTensorV2(
    TestDeviceContext *device_context,
    Tensor *src,
    const index_t h_dim_start,
    const index_t h_dim_length) {
  MACE_CHECK(src->data_format() == DataFormat::NHWC);
  //MACE_CHECK(src->dtype() == DataType::DT_FLOAT);
  
  std::vector<index_t> part_shape;
  part_shape.push_back(src->dim(0));
  part_shape.push_back(h_dim_length);
  part_shape.push_back(src->dim(2));
  part_shape.push_back(src->dim(3));
  
  index_t reuse_data_offset = h_dim_start  * src->dim(2)
                                * src->dim(3) * GetEnumTypeSize(src->dtype());
  index_t reuse_data_size   = h_dim_length * src->dim(2)
                                * src->dim(3) * GetEnumTypeSize(src->dtype());
  DeviceType device_type = IsGPUTensor(src) ? DeviceType::GPU : DeviceType::CPU;
  
  Tensor *dst = CreateBufferTensor(
      device_context, part_shape, src->dtype(),
      src->is_weight(), src->data_format(),
      device_type, std::string("part_tensor"), false);
                             
  dst->ReuseTensorBuffer(*src, reuse_data_offset, reuse_data_size);
  dst->Reshape(part_shape);
  
  return dst;
}

Tensor* TensorUtils::CreatePartTensorV1(
    TestDeviceContext *device_context,
    Tensor *src,
    const index_t h_dim_start) {
  MACE_CHECK(src->data_format() == DataFormat::NHWC);
  //MACE_CHECK(src->dtype() == DataType::DT_FLOAT);
  return CreatePartTensorV2(device_context,
                            src,
                            h_dim_start,
                            src->dim(1) - h_dim_start);
}

template<>
Tensor *TensorUtils::CreateTensor<MemoryType::CPU_BUFFER>(
    TestDeviceContext *dev_context,
    const std::vector<index_t> &shape,
    const DataType dt,
    const bool is_weight,
    const DataFormat data_format,
    const DeviceType dev_type,
    const OpenCLBufferType buffer_type,
    const std::string name,
    const bool alloc_buffer_hint,
    const bool map_buffer_hint) {
  MACE_CHECK(dev_type == DeviceType::CPU);
  MACE_UNUSED(buffer_type);
  return CreateBufferTensor(
      dev_context, shape, dt, is_weight, data_format, dev_type,
      name, alloc_buffer_hint, map_buffer_hint);
}

template<>
Tensor *TensorUtils::CreateTensor<MemoryType::GPU_BUFFER>(
    TestDeviceContext *dev_context,
    const std::vector<index_t> &shape,
    const DataType dt,
    const bool is_weight,
    const DataFormat data_format,
    const DeviceType dev_type,
    const OpenCLBufferType buffer_type,
    const std::string name,
    const bool alloc_buffer_hint,
    const bool map_buffer_hint) {
  MACE_CHECK(dev_type == DeviceType::GPU);
  MACE_UNUSED(buffer_type);
  return CreateBufferTensor(
      dev_context, shape, dt, is_weight, data_format, dev_type,
      name, alloc_buffer_hint, map_buffer_hint);
}

template<>
Tensor *TensorUtils::CreateTensor<MemoryType::GPU_IMAGE>(
    TestDeviceContext *dev_context,
    const std::vector<index_t> &shape,
    const DataType dt,
    const bool is_weight,
    const DataFormat data_format,
    const DeviceType dev_type,
    const OpenCLBufferType buffer_type,
    const std::string name,
    const bool alloc_buffer_hint,
    const bool map_buffer_hint) {
  MACE_CHECK(dev_type == DeviceType::GPU);
  MACE_UNUSED(alloc_buffer_hint);
  MACE_UNUSED(map_buffer_hint);
  return CreateGPUImageTensor(
      dev_context, shape, dt, is_weight, data_format, buffer_type, name);
}

const std::string DataFormatToString(const DataFormat df) {
  if (df == DataFormat::NHWC) {
    return "NHWC";
  } else if (df == DataFormat::NCHW) {
    return "NCHW";
  } else {
    return "NONE";
  }
}

template<> float* TensorBufferUtils::CreateSampleBuffer(
    const std::vector<index_t> &shape) {
  MACE_CHECK(shape.size() == 4);
  
  index_t data_size = std::accumulate(shape.begin(),
                                      shape.end(),
                                      1, std::multiplies<int64_t>());
                                      
  float *sam_buf = new float[data_size];
  index_t strides[3];
  strides[0] = shape[1] * shape[2] * shape[3];
  strides[1] = shape[2] * shape[3];
  strides[2] = shape[3];
  
  for (int n = 0; n < shape[0]; n ++) {
    for (int h = 0; h < shape[1]; h ++) {
      for (int w = 0; w < shape[2]; w ++) {
        for (int c = 0; c < shape[3]; c ++) {
          sam_buf[n * strides[0] +
                  h * strides[1] +
                  w * strides[2] + c] = (float) c;
        }
      }
    }
  }
  
  return sam_buf;
}

void FillTensorDataFloatFormatNHWC(
    Tensor *tensor,
    const std::vector<float> &template_data) {
  std::shared_ptr<float> tensor_temp_data(reinterpret_cast<float *>(
      malloc(tensor->raw_size() + EXTRA_BUFFER_PAD_SIZE)));
  // Fill temporary buffer with template data.
  index_t strides[3];
  strides[0] = tensor->dim(1) * tensor->dim(2) * tensor->dim(3);
  strides[1] = tensor->dim(2) * tensor->dim(3);
  strides[2] = tensor->dim(3);
  
  std::vector<float> next_template_data = template_data;
  
  for (int n = 0; n < tensor->dim(0); n ++) {
    for (int h = 0; h < tensor->dim(1); h ++) {
      for (int w = 0; w < tensor->dim(2); w ++) {
        for (int c = 0; c < tensor->dim(3); c ++) {
            float *data_ptr = tensor_temp_data.get();
          data_ptr[n * strides[0] +
                   h * strides[1] +
                   w * strides[2] + c]
              = next_template_data[c % next_template_data.size()];
        }
      }
    }
  }
  
  // Tensor copy data from temporary buffer.
  tensor->Copy<float>(tensor_temp_data.get(), tensor->size());
}

void FillTensorDataFloatFormatNCHW(
    Tensor *tensor,
    const std::vector<float> &template_data) {
  std::shared_ptr<float> tensor_temp_data(reinterpret_cast<float *>(
      malloc(tensor->raw_size() + EXTRA_BUFFER_PAD_SIZE)));
  // Fill temporary buffer with template data.
  index_t strides[3];
  strides[0] = tensor->dim(1) * tensor->dim(2) * tensor->dim(3);
  strides[1] = tensor->dim(2) * tensor->dim(3);
  strides[2] = tensor->dim(3);
  
  std::vector<float> next_template_data = template_data;
  
  for (int n = 0; n < tensor->dim(0); n ++) {
    for (int c = 0; c < tensor->dim(1); c ++) {
      for (int h = 0; h < tensor->dim(2); h ++) {
        for (int w = 0; w < tensor->dim(3); w ++) {
          float *data_ptr = tensor_temp_data.get();
          data_ptr[n * strides[0] +
                   c * strides[1] +
                   h * strides[2] + w]
              = next_template_data[c % next_template_data.size()];

        }
      }
    }
  }
  
  // Tensor copy data from temporary buffer.
  tensor->Copy<float>(tensor_temp_data.get(), tensor->size());
}

void FillTensorDataFloatFormatNONE(
    Tensor *tensor,
    const std::vector<float> &template_data) {
  std::shared_ptr<float> tensor_temp_data(reinterpret_cast<float *>(
      malloc(tensor->raw_size() + EXTRA_BUFFER_PAD_SIZE)));
  
  for (int i = 0; i < tensor->size(); i ++) {
    float *data_ptr = tensor_temp_data.get();
    data_ptr[i] = template_data[i % template_data.size()];
  }
  
  tensor->Copy<float>(tensor_temp_data.get(), tensor->size());
}

void FillTensorDataFloat(
    Tensor *tensor,
    const std::vector<float> &template_data) {
  if (tensor->data_format() == DataFormat::NHWC && tensor->dim_size() == 4) {
    FillTensorDataFloatFormatNHWC(tensor, template_data);
  } else if ((tensor->data_format() == DataFormat::NCHW &&
              tensor->dim_size() == 4) ||
             (tensor->data_format() == DataFormat::OIHW &&
              tensor->dim_size() == 4)) {
    FillTensorDataFloatFormatNCHW(tensor, template_data);
  } else {
    FillTensorDataFloatFormatNONE(tensor, template_data);
  }
}

void TensorDataFiller::Fill(Tensor *tensor,
                            const std::vector<float> &template_data) {
  MACE_CHECK(!tensor->has_opencl_image(), "tensor data must be buffer");
  FillTensorDataFloat(tensor, template_data);
}

void TensorDataFiller::FillImage(TestDeviceContext *dev_context,
                                 OpContext *op_context,
                                 Tensor *tensor,
                                 const OpenCLBufferType buffer_type,
                                 const int wino_blk_size,
                                 const std::vector<float> &template_data) {
  MACE_CHECK(tensor->has_opencl_image(), "tensor data must be opencl image");
  // 1. Create temporary buffer tensor.
  std::shared_ptr<Tensor> temp_tensor(
      TensorUtils::CreateTensor<MemoryType::GPU_BUFFER>(
          dev_context, tensor->shape(), DataType::DT_FLOAT, tensor->is_weight(),
          tensor->data_format(), DeviceType::GPU, buffer_type, "temp_tensor",
          true, false));
  // 2. Fill data for temporary tensor.
  Fill(temp_tensor.get(), template_data);
  if (template_data.size() == 1) {
    MACE_CHECK(TensorDataChecker().EqualBuffer(temp_tensor.get(),
                                               template_data[0]),
               "fill image data tensor name ", tensor->name());
  }
  // 3. Fill image data using data transform approach.
  StatsFuture future;
  StatsFuture *old_future = op_context->future();
  op_context->set_future(&future);
  TensorDataTransformer(MemoryType::GPU_BUFFER,
                        MemoryType::GPU_IMAGE).Transform(
      op_context, temp_tensor.get(), buffer_type, wino_blk_size, true, tensor);
  future.wait_fn(nullptr);
  op_context->set_future(old_future);
}

void TensorDebugUtils::PrintTensorInfo(const Tensor *tensor) {
  char string_buf[1024];
  snprintf(string_buf, 1024,
      "Tensor name: %s\n"
      "Tensor dims: [%ld,%ld,%ld,%ld]\n"
      "Tensor size: %ld\n"
      "Tensor raw size: %ld\n"
      "Tensor data type: %s\n"
      "Tensor data format: %s\n",
       tensor->name().c_str(),
       tensor->dim(0), tensor->dim(1), tensor->dim(2), tensor->dim(3),
       tensor->size(), tensor->raw_size(),
       DataTypeToString(tensor->dtype()).c_str(),
       DataFormatToString(tensor->data_format()).c_str());
  printf("%s", string_buf);
}

void TensorDebugUtils::PrintTensorBufferData(const Tensor *tensor) {
  MACE_CHECK(tensor != nullptr);
  Tensor::MappingGuard guard(tensor, AllocatorMapType::AMT_READ_ONLY);
  MACE_CHECK(tensor->raw_data() != nullptr);
  
  LOG(INFO) << "===== Print Tensor Data =====";
  LOG(INFO) << "Tensor name: " << tensor->name();
  LOG(INFO) << "Tensor shape: " << ShapeToString(tensor->shape());
  LOG(INFO) << "Tensor size: " << tensor->size();
  LOG(INFO) << "Tensor data format: " << DataFormatToString(tensor->data_format());
  LOG(INFO) << "Tensor data:";
  std::stringstream stream;
  stream << "[";
  for (int i = 0; i < tensor->size() && i < kMaxPrintDataCount; i ++) {
    if (tensor->dtype() == DataType::DT_FLOAT){
      stream << tensor->data<float>()[i];
    } else if (tensor->dtype() == DataType::DT_HALF) {
      stream << static_cast<float>(tensor->data<half>()[i]);
    }
    
    stream << ",";
  }
  stream << "]";
  LOG(INFO) << stream.str();
}

void TensorDebugUtils::PrintTensorImageData(
    TestDeviceContext *dev_context,
    OpContext *op_context,
    const Tensor *tensor,
    const OpenCLBufferType buffer_type,
    const int wino_blk_size) {
  // 1. Create temporary buffer tensor.
  std::shared_ptr<Tensor> temp_tensor(
      TensorUtils::CreateTensor<MemoryType::GPU_BUFFER>(
          dev_context, tensor->shape(), DataType::DT_FLOAT,
          tensor->is_weight(), tensor->data_format(), DeviceType::GPU,
          buffer_type, "temp_tensor", true, false));
  // 2. Tranform data.
  StatsFuture future;
  StatsFuture *old_future = op_context->future();
  op_context->set_future(&future);
  TensorDataTransformer(MemoryType::GPU_IMAGE,
                        MemoryType::GPU_BUFFER).Transform(
      op_context, tensor, buffer_type, wino_blk_size, true, temp_tensor.get());
  future.wait_fn(nullptr);
  op_context->set_future(old_future);
  // 3. Print data.
  return PrintTensorBufferData(temp_tensor.get());
}

void TensorDebugUtils::PrintTensorData(
    TestDeviceContext *dev_context,
    OpContext *op_context,
    const Tensor *tensor,
    const OpenCLBufferType buffer_type,
    const int wino_blk_size) {
  if (tensor->has_opencl_image()) {
    TensorDebugUtils::PrintTensorImageData(dev_context,
                                           op_context,
                                           tensor,
                                           buffer_type,
                                           wino_blk_size);
  } else {
    TensorDebugUtils::PrintTensorBufferData(tensor);
  }
}

bool TensorDebugUtils::CheckTensorDataFloat(
    const Tensor *tensor,
    const float value) {
  MACE_CHECK(tensor != nullptr);
  
  Tensor::MappingGuard guard(tensor, AllocatorMapType::AMT_READ_ONLY);
  
  MACE_CHECK(tensor->data<float>() != nullptr);
  
  bool ret = true;
  const int max_count = 100;
  
  for (int i = 0, count = 0; i < tensor->size() && count < max_count; i ++) {
    if (tensor->data<float>()[i] != value) {
      LOG(INFO) << "Tensor data[" << i << "] != value ("
                << tensor->data<float>()[i] << " != "
                << value << ")";
      ret = false;
      
      count ++;
    }
  }
  
  return ret;
}

int TensorDataClearer::ClearTensor(Tensor *tensor) {
  Tensor::MappingGuard guard(tensor);
  tensor->Clear();
  return 0;
}

MaceStatus TensorDataTransformer::Transform(
    OpContext *context,
    const Tensor *input,
    const OpenCLBufferType buffer_type,
    const int wino_block_size,
    const bool use_internal_tensor,
    Tensor *output) {
  ops::OpenCLBufferTransformer transformer(in_mem_type_, out_mem_type_);
  if (use_internal_tensor) {
    return transformer.Transform(context, input, buffer_type,
                                 out_mem_type_, wino_block_size, output);
  } else {
    return transformer.TransformNoInternalTensor(
        context, input, buffer_type,
        out_mem_type_, wino_block_size, output);
  }
}

bool TensorDataChecker::Equal(const Tensor *a, const Tensor *b) {
  MACE_CHECK(a->shape() == b->shape());
  MACE_CHECK(a->dtype() == b->dtype());
  std::shared_ptr<Tensor::MappingGuard> guard_a_ptr;
  if (!a->IsBufferMapped()) {
    guard_a_ptr.reset(new Tensor::MappingGuard(a));
  }
  Tensor::MappingGuard guard_b(b);

  const DataType dt = a->dtype();
  switch (dt) {
    case DataType::DT_FLOAT:
      return EqualInternal<float>(a, b);
    default:
      LOG(ERROR) << "Unsupported data type " << DataTypeToString(dt);
  }

  return true;
}

template<class T>
bool TensorDataChecker::EqualInternal(const Tensor *a, const Tensor *b) {
  const T *a_ptr = a->data<T>();
  const T *b_ptr = b->data<T>();
  int log_count = 0;
  const int kMaxLogCount = 10;
  size_t diff_value_count = 0;
  bool result = true;
  for (int i = 0; i < a->size(); ++i) {
    T v1 = a_ptr[i];
    T v2 = b_ptr[i];
    if (v1 != v2) {
      if (log_count < kMaxLogCount) {
        LOG(WARNING) << "Different data found,"
                     << " index " << i
                     << ", v1 " << v1
                     << ", v2 " << v2;
        log_count ++;
      }

      result = false;
      diff_value_count ++;
    }
  }

  LOG(INFO) << "Total value count " << a->size()
            << ", different value count " << diff_value_count;

  return result;
}

bool TensorDataChecker::EqualBuffer(const Tensor *tensor,
                                    const float value) {
  MACE_CHECK(!tensor->has_opencl_image(), "tensor data must be buffer");
  std::shared_ptr<Tensor::MappingGuard> guard;
  if (!tensor->IsBufferMapped()) {
    guard.reset(new Tensor::MappingGuard(tensor));
  }
  const int kMaxLogCount = 10;
  int log_count = 0;
  int num_not_equal = 0;
  const float *data = tensor->data<float>();
  for (int i = 0; i < tensor->size(); i ++) {
    if (data[i] != value) {
      if (log_count < kMaxLogCount) {
        LOG(INFO) << "Not qual value: id " << i
                  << ", data " << data[i] << ", value " << value;
        log_count ++;
      }
      num_not_equal ++;
    }
  }

  if (num_not_equal) {
    LOG(INFO) << "Not euqal " << num_not_equal
              << ", total " << tensor->size();
  }

  return num_not_equal == 0;
}

bool TensorDataChecker::EqualImage(TestDeviceContext *dev_context,
                                   OpContext *op_context,
                                   const Tensor *tensor,
                                   const OpenCLBufferType buffer_type,
                                   const int wino_blk_size,
                                   const float value) {
  if (tensor->has_opencl_image()) {
    // 1. Create temporary buffer tensor.
    std::shared_ptr<Tensor> temp_tensor(
        TensorUtils::CreateTensor<MemoryType::GPU_BUFFER>(
            dev_context, tensor->shape(), DataType::DT_FLOAT,
            tensor->is_weight(), tensor->data_format(), DeviceType::GPU,
            buffer_type, "temp_tensor", true, false));
    // 2. Tranform data.
    StatsFuture future;
    StatsFuture *old_future = op_context->future();
    op_context->set_future(&future);
    TensorDataTransformer(MemoryType::GPU_IMAGE,
                          MemoryType::GPU_BUFFER).Transform(
        op_context, tensor, buffer_type, wino_blk_size, true, temp_tensor.get());
    future.wait_fn(nullptr);
    op_context->set_future(old_future);
    // 3. Compare data.
    return EqualBuffer(temp_tensor.get(), value);
  } else {
    return EqualBuffer(tensor, value);
  }
}
