
#include "mace/core/allocator.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/gemm.h"
#include "mace/ops/opencl/buffer/gemm.h"
#endif

#include "test/fucheng/gemm_test_task.h"
#include "test/fucheng/gemm_test_param.h"

namespace mace {

static std::unique_ptr<TestDeviceContext> context;

TestDeviceContext *GetDeviceContext() {
  return context.get();
}

int CodlGemmTestTask::TensorRead(Tensor *tensor,
                                 const float *buf,
                                 const index_t size) {
  if (buf != nullptr) {
    tensor->Copy<float>(buf, size);
  }
  return 0;
}

int CodlGemmTestTask::TensorWrite(const Tensor *tensor,
                                  float *buf,
                                  const index_t size) {
  if (buf != nullptr) {
    tensor->Read<float>(buf, size);
  }
  return 0;
}

int CodlGemmCpuTestTask::Prepare(TestParam *test_param) {
  // Initilize parameters.
  GemmTestParam *gemm_test_param =
      reinterpret_cast<GemmTestParam *>(test_param);
  rows_ = gemm_test_param->rows;
  depth_ = gemm_test_param->depth;
  cols_ = gemm_test_param->cols;
  const int num_threads = gemm_test_param->num_threads;
  const CPUAffinityPolicy policy
      = static_cast<CPUAffinityPolicy>(gemm_test_param->cpu_affinity_policy);

  // Initialize device context.
  TestDeviceContext *dev_context = GetDeviceContext();
  if (dev_context == nullptr) {
    context.reset(new TestDeviceContext(num_threads, policy));
    dev_context = GetDeviceContext();
  }
  if (!dev_context->is_initialized()) {
    dev_context->InitCpuDevice();
    dev_context->set_is_initialized(true);
  }
  
  // Initialize context.
  context_.reset(new OpContext(GetWorkspace(), dev_context->GetCpuDevice()));
  context_->set_cpu_device(dev_context->GetCpuDevice());

  tensor_manage_util_.reset(new TensorManageUtil(
      dev_context->GetCpuDevice()->allocator()));

  const std::vector<index_t> lhs_shape = {rows_, depth_, 1, 1};
  const std::vector<index_t> rhs_shape = {1, 1, depth_, cols_};
  const std::vector<index_t> output_shape = {1, 1, rows_, cols_};
  lhs_ = tensor_manage_util_->CreateOrReshapeTensor(
        lhs_shape, DT_FLOAT, false,
        DataFormat::OIHW, DeviceType::CPU,
        std::string("lhs"), false);
  rhs_ = tensor_manage_util_->CreateOrReshapeTensor(
        rhs_shape, DT_FLOAT, false,
        DataFormat::NCHW, DeviceType::CPU,
        std::string("rhs"), false);
  output_ = tensor_manage_util_->CreateOrReshapeTensor(
        output_shape, DT_FLOAT, false,
        DataFormat::NCHW, DeviceType::CPU,
        std::string("output"), false);

  TensorRead(lhs_, gemm_test_param->lhs, rows_ * depth_);
  TensorRead(rhs_, gemm_test_param->rhs, depth_ * cols_);
  if (gemm_test_param->output != nullptr) {
    output_buffer_ = gemm_test_param->output;
  }

  return 0;
}

int CodlGemmCpuTestTask::Run(mace::DurationCollector<double> *dura_collector) {
  MACE_CHECK_NOTNULL(lhs_);
  MACE_CHECK_NOTNULL(rhs_);
  MACE_CHECK_NOTNULL(output_);
  MACE_UNUSED(dura_collector);

  const index_t lhs_rows = lhs_->dim(0);
  const index_t lhs_cols = lhs_->dim(1);
  const index_t rhs_rows = rhs_->dim(2);
  const index_t rhs_cols = rhs_->dim(3);

  const index_t out_rows = output_->dim(2);
  const index_t out_cols = output_->dim(3);

  auto scratch_buffer = context_->cpu_device()->scratch_buffer();
  const index_t pack_lhs_size =
      PadAlignSize(sizeof(float) * lhs_rows * lhs_cols);
  const index_t pack_rhs_size =
      PadAlignSize(sizeof(float) * rhs_rows * rhs_cols);
  const index_t pack_output_size =
      PadAlignSize(sizeof(float) * out_rows * out_cols);

  const index_t gemm_pack_size =
      pack_lhs_size + pack_rhs_size + pack_output_size;

  scratch_buffer->Rewind();
  scratch_buffer->GrowSize(gemm_pack_size);

  gemm_.Compute(context_.get(),
                lhs_,
                rhs_,
                1,
                rows_,
                depth_,
                depth_,
                cols_,
                false,
                false,
                false,
                false,
                true,
                output_);

  return 0;
}

int CodlGemmCpuTestTask::Destroy() {
  TensorWrite(output_, output_buffer_, rows_ * cols_);

  if (tensor_manage_util_ != nullptr) {
    tensor_manage_util_->DeleteTensors();
  }
  
  context.reset(nullptr);

  return 1;
}

#ifdef MACE_ENABLE_OPENCL

ops::OpenCLGemmKernel *CodlGemmGpuTestTask::CreateOpenCLGemmKernel(
    MemoryType memory_type) {
  if (memory_type == MemoryType::GPU_IMAGE) {
    return new ops::opencl::image::GemmKernel();
  } else {
    return new ops::opencl::buffer::GemmKernel();
  }
}

int CodlGemmGpuTestTask::TensorTransformRead(
    const float *input,
    const index_t size,
    const OpenCLBufferType type,
    Tensor *output) {
  StatsFuture future;
  context_->set_future(&future);
  Buffer internal_buffer(context_->cpu_device()->allocator(),
                         const_cast<float *>(input),
                         size * sizeof(float));
  Tensor internal_tensor(&internal_buffer, DataType::DT_FLOAT);
  internal_tensor.Reshape(output->shape());
  internal_tensor.set_data_format(output->data_format());
  ops::OpenCLBufferTransformer transformer(MemoryType::CPU_BUFFER,
                                           MemoryType::GPU_IMAGE);
  transformer.Transform(context_.get(),
                        &internal_tensor,
                        type,
                        MemoryType::GPU_IMAGE,
                        0,
                        output);
  future.wait_fn(nullptr);

  return 0;
}

int CodlGemmGpuTestTask::TensorTransformWrite(
    const Tensor *input,
    const index_t size,
    const OpenCLBufferType type,
    float *output) {
  StatsFuture future;
  context_->set_future(&future);
  Buffer internal_buffer(context_->cpu_device()->allocator(),
                         const_cast<float *>(output),
                         size * sizeof(float));
  Tensor internal_tensor(&internal_buffer, DataType::DT_FLOAT);
  internal_tensor.Reshape(input->shape());
  internal_tensor.set_data_format(input->data_format());
  ops::OpenCLBufferTransformer transformer(MemoryType::GPU_IMAGE,
                                           MemoryType::CPU_BUFFER);
  transformer.Transform(context_.get(),
                        input,
                        type,
                        MemoryType::CPU_BUFFER,
                        0,
                        &internal_tensor);
  future.wait_fn(nullptr);

  return 0;
}

int CodlGemmGpuTestTask::Prepare(TestParam *test_param) {
  // Initilize parameters.
  GemmTestParam *gemm_test_param = reinterpret_cast<GemmTestParam *>(test_param);
  rows_ = gemm_test_param->rows;
  depth_ = gemm_test_param->depth;
  cols_ = gemm_test_param->cols;
  const int num_threads = gemm_test_param->num_threads;
  const CPUAffinityPolicy policy
      = static_cast<CPUAffinityPolicy>(gemm_test_param->cpu_affinity_policy);
  const int memory_type_idx = gemm_test_param->memory_type_idx;
  memory_type_ = memory_type_idx == 0 ? MemoryType::GPU_BUFFER :
                                        MemoryType::GPU_IMAGE;

  // Initialize device context.
  TestDeviceContext *dev_context = GetDeviceContext();
  if (dev_context == nullptr) {
    context.reset(new TestDeviceContext(num_threads, policy));
    dev_context = GetDeviceContext();
  }
  if (!dev_context->is_initialized()) {
    dev_context->InitCpuDevice();
    dev_context->InitGpuDevice();
    dev_context->set_is_initialized(true);
  }
  
  // Initialize context.
  context_.reset(new OpContext(GetWorkspace(), dev_context->GetGpuDevice()));
  context_->set_cpu_device(dev_context->GetCpuDevice());

  tensor_manage_util_.reset(new TensorManageUtil(
      dev_context->GetCpuDevice()->allocator()));

  std::vector<index_t> lhs_shape;
  std::vector<index_t> rhs_shape;
  std::vector<index_t> output_shape;
  if (memory_type_ == MemoryType::GPU_IMAGE) {
    lhs_shape = {rows_, depth_, 1, 1};
    rhs_shape = {1, 1, cols_, depth_};
    output_shape = {1, 1, cols_, rows_};
    lhs_ = TensorUtils::CreateGPUImageTensor(
        dev_context, lhs_shape, DT_FLOAT, false,
        DataFormat::OIHW, OpenCLBufferType::WEIGHT_WIDTH,
        std::string("lhs"));
    rhs_ = TensorUtils::CreateGPUImageTensor(
        dev_context, rhs_shape, DT_FLOAT, false,
        DataFormat::NHWC, OpenCLBufferType::GEMM_IN_OUT,
        std::string("rhs"));
    output_ = TensorUtils::CreateGPUImageTensor(
        dev_context, output_shape, DT_FLOAT, false,
        DataFormat::NHWC, OpenCLBufferType::GEMM_IN_OUT,
        std::string("output"));
  } else {
    lhs_shape = {rows_, depth_, 1, 1};
    rhs_shape = {1, depth_, 1, cols_};
    output_shape = {1, rows_, 1, cols_};
    lhs_ = TensorUtils::CreateBufferTensor(
        dev_context, lhs_shape, DT_FLOAT, false,
        DataFormat::OIHW, DeviceType::GPU,
        std::string("lhs"), true);
    rhs_ = TensorUtils::CreateBufferTensor(
        dev_context, rhs_shape, DT_FLOAT, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("rhs"), true);
    output_ = TensorUtils::CreateBufferTensor(
        dev_context, output_shape, DT_FLOAT, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("output"), true);
  }

  gemm_.reset(CreateOpenCLGemmKernel(memory_type_));

  if (memory_type_ == MemoryType::GPU_IMAGE) {
    // TODO(fucheng): Transform memory type from buffer to image.
    TensorTransformRead(gemm_test_param->lhs,
                        rows_ * depth_,
                        OpenCLBufferType::WEIGHT_WIDTH,
                        lhs_);
    TensorTransformRead(gemm_test_param->rhs,
                        depth_ * cols_,
                        OpenCLBufferType::GEMM_IN_OUT,
                        rhs_);
  } else {
    TensorRead(lhs_, gemm_test_param->lhs, rows_ * depth_);
    TensorRead(rhs_, gemm_test_param->rhs, depth_ * cols_);
  }

  if (gemm_test_param->output != nullptr) {
    output_buffer_ = gemm_test_param->output;
  }

  return 0;
}

int CodlGemmGpuTestTask::Run(mace::DurationCollector<double> *dura_collector) {
  MACE_CHECK_NOTNULL(lhs_);
  MACE_CHECK_NOTNULL(rhs_);
  MACE_CHECK_NOTNULL(output_);
  MACE_CHECK_NOTNULL(gemm_);
  MACE_UNUSED(dura_collector);

  StatsFuture future;
  context_->set_future(&future);
  gemm_->Compute(context_.get(), lhs_, rhs_, output_);
  future.wait_fn(nullptr);

  return 0;
}

int CodlGemmGpuTestTask::Destroy() {
  OpenCLRuntime *opencl_runtime =
      GetDeviceContext()->GetGpuDevice()->gpu_runtime()->opencl_runtime();
  OpenCLEventManager *event_manager = opencl_runtime->event_manager();
  opencl_runtime->command_queue().finish();
  event_manager->Clear();

  if (memory_type_ == MemoryType::GPU_IMAGE) {
    // TODO(fucheng): Transform memory type from image to buffer.
    TensorTransformWrite(output_,
                         rows_ * cols_,
                         OpenCLBufferType::GEMM_IN_OUT,
                         output_buffer_);
  } else {
    //output_->DebugPrint();
    TensorWrite(output_, output_buffer_, rows_ * cols_);
  }

  if (tensor_manage_util_ != nullptr) {
    tensor_manage_util_->DeleteTensors();
  }
  context.reset(nullptr);

  return 1;
}

#endif  // MACE_ENABLE_OPENCL

} // namespace mace
