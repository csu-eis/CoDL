
#include "mace/utils/thread_pool.h"
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/matmul_part_plan.h"
#include "test/codl_run/matmul_test_param.h"
#include "test/codl_run/matmul_test_task.h"

namespace mace {

int CodlMatMulCpuGpuTestTask::Prepare(TestParam *test_param) {
  MatMulTestParam *matmul_test_param =
      reinterpret_cast<MatMulTestParam *>(test_param);

  // Load parameters for testing.
  const int num_threads = matmul_test_param->num_threads;
  const CPUAffinityPolicy policy
      = static_cast<CPUAffinityPolicy>(matmul_test_param->cpu_affinity_policy);
  is_debug_on_ = matmul_test_param->is_debug_on;
  do_data_transform_ = matmul_test_param->do_data_transform;
  do_compute_ = matmul_test_param->do_compute;
  const MemoryType gpu_memory_type = matmul_test_param->gpu_memory_type;
  const PartitionDim part_dim = static_cast<PartitionDim>(matmul_test_param->part_dim);
  const float part_ratio = matmul_test_param->part_ratio;
  compute_unit_hint_     = matmul_test_param->compute_unit_hint;
  const DataType cpu_dtype = matmul_test_param->cpu_dtype;
  const DataType gpu_dtype = matmul_test_param->gpu_dtype;

  transpose_a_ = matmul_test_param->transpose_a;
  transpose_b_ = matmul_test_param->transpose_b;
  
  const std::vector<index_t> input_shape = matmul_test_param->input_shape;
  const std::vector<index_t> rhs_shape = matmul_test_param->rhs_shape;
  std::vector<index_t> output_shape;
  ops::MatMulPartPlanUtils::CalcOutputShape(
      input_shape, rhs_shape, transpose_a_, transpose_b_, output_shape);
  const size_t rank = input_shape.size();

  // Create partition plan.
  ShowText("Create matmul partition plan");
  ops::MatMulPartPlan *matmul_part_plan = new ops::MatMulPartPlan(
      part_dim, part_ratio, DataFormat::NCHW);
  ops::PartitionResult partition_result;
  do {
    partition_result = matmul_part_plan->Make(input_shape,
                                              rhs_shape,
                                              transpose_a_,
                                              transpose_b_);
  } while (partition_result == ops::PartitionResult::PARTITION_REDO);

  if (partition_result == ops::PartitionResult::PARTITION_FAILED) {
    LOG(ERROR) << "Make matmul part plan failed";
    return -1;
  }

#if 0
  matmul_part_plan->Show();
#endif

  part_plan_.reset(matmul_part_plan);

  // Initialize device context.
  ShowText("Initialize device context");
  TestDeviceContext *dev_context = GetDeviceContext();
  if (dev_context == nullptr) {
    SetDeviceContext(new TestDeviceContext(num_threads, policy));
    dev_context = GetDeviceContext();
  }
  if (!dev_context->is_initialized()) {
    dev_context->InitCpuDevice();
    dev_context->InitGpuDevice();
    dev_context->set_is_initialized(true);
  }
  // Initialize cpu and gpu context.
  gpu_context_.reset(new OpContext(GetWorkspace(), dev_context->GetGpuDevice()));
  cpu_context_.reset(new OpContext(GetWorkspace(), dev_context->GetCpuDevice()));
  cpu_context_->set_cpu_device(dev_context->GetCpuDevice());

  const bool use_opencl_image = (gpu_memory_type == MemoryType::GPU_IMAGE);
  const DataType in_out_dt = gpu_dtype;
  const DataType weights_dt = gpu_dtype;
  MACE_CHECK(gpu_dtype == DataType::DT_FLOAT || gpu_dtype == DataType::DT_HALF);

  if (!use_opencl_image) {
    // No data transforming for gpu buffer.
    do_data_transform_ = false;
  }

  ShowText("Create gpu tensor");
  if (use_opencl_image) {
    // Create GPU (image) full tensors.
    input_ = TensorUtils::CreateGPUImageTensor(
        dev_context, input_shape, in_out_dt, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("input_image"));
    rhs_ = TensorUtils::CreateGPUImageTensor(
        dev_context, rhs_shape, weights_dt, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("rhs_image"));
    output_ = TensorUtils::CreateGPUImageTensor(
        dev_context, output_shape, in_out_dt, false,
        DataFormat::NHWC, OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("output_image"));
  } else {
    // Create GPU (buffer) tensors.
    input_ = TensorUtils::CreateBufferTensor(
        dev_context, input_shape, in_out_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("input_buffer"), true);
    rhs_ = TensorUtils::CreateBufferTensor(
        dev_context, rhs_shape, weights_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("rhs_buffer"), true);
    output_ = TensorUtils::CreateBufferTensor(
        dev_context, output_shape, in_out_dt, false,
        DataFormat::NHWC, DeviceType::GPU,
        std::string("output_buffer"), true);
  }

  const std::vector<index_t>
      part_shape_input_gpu = part_plan_->gpu_input_part_shape();
  const std::vector<index_t>
      part_shape_rhs_gpu = matmul_part_plan->gpu_rhs_part_shape();
  const std::vector<index_t>
      part_shape_output_gpu = part_plan_->gpu_output_part_shape();
  const std::vector<index_t>
      part_shape_input_cpu  = part_plan_->cpu_input_part_shape();
  const std::vector<index_t>
      part_shape_rhs_cpu = matmul_part_plan->cpu_rhs_part_shape();
  const std::vector<index_t>
      part_shape_output_cpu = part_plan_->cpu_output_part_shape();

  tensor_manage_util_.reset(new TensorManageUtil(
      dev_context->GetCpuDevice()->allocator()));
  tensor_manage_util_->set_gpu_allocator(
      dev_context->GetGpuDevice()->allocator());

  // Create gpu partial tensors and cpu tensors.
  if (!part_plan_->IsGpuOnly()) {
    if (!part_plan_->IsCpuOnly()) {
      ShowText("Create gpu partial tensors");
      index_t in_out_dim_idx = H_NHWC;
      index_t part_length;
      if (part_dim == DIM_INPUT_HEIGHT) {
        in_out_dim_idx = H_NHWC;
      } else if (part_dim == DIM_OUTPUT_CHANNEL) {
        if (rank == 2) {
          in_out_dim_idx = N_NHWC;
        } else if (rank == 4) {
          in_out_dim_idx = C_NHWC;
        } else {
          MACE_NOT_IMPLEMENTED;
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
      part_length = part_shape_input_gpu[in_out_dim_idx];
      input_gpu_  = tensor_manage_util_->CreateOrReshapePartTensor(
                      input_, part_dim, 0, part_length, true);
      part_length = part_shape_rhs_gpu[in_out_dim_idx];
      rhs_gpu_ = tensor_manage_util_->CreateOrReshapePartTensor(
                      rhs_, part_dim, 0, part_length, true);
      part_length = part_shape_output_gpu[in_out_dim_idx];
      output_gpu_ = tensor_manage_util_->CreateOrReshapePartTensor(
                      output_, part_dim, 0, part_length, true);
    }

    ShowText("Create cpu tensors");
    const DeviceType in_out_device_type = DeviceType::GPU;
    const DeviceType rhs_device_type = DeviceType::GPU;
    bool in_out_mapping_hint = false;
    const bool weights_mapping_hint = true;

    input_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_input_cpu, DT_FLOAT, false,
        DataFormat::NCHW, in_out_device_type,
        std::string("input_cpu"), in_out_mapping_hint);
    rhs_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_rhs_cpu, DT_FLOAT, rhs_->is_weight(),
        DataFormat::NCHW, rhs_device_type,
        std::string("rhs_cpu"), weights_mapping_hint);
    output_cpu_ = tensor_manage_util_->CreateOrReshapeTensor(
        part_shape_output_cpu, DT_FLOAT, false,
        DataFormat::NCHW, in_out_device_type,
        std::string("output_cpu"), in_out_mapping_hint);
    bias_cpu_ = nullptr;
  }

  // Create computing kernel.
  ShowText("Create cpu computing kernel");
  if (!part_plan_->IsGpuOnly()) {
    if (cpu_dtype == DataType::DT_FLOAT) {
      cpu_kernel_.reset(new MatMulCpuFloatKernel());
    } else if (cpu_dtype == DataType::DT_UINT8) {
      cpu_kernel_.reset(new MatMulCpuUint8Kernel());
      input_cpu_->SetScale(0.5);
      rhs_cpu_->SetScale(0.5);
      output_cpu_->SetScale(0.5);
    } else {
      LOG(ERROR) << "Unsupported cpu data type " << static_cast<int>(cpu_dtype);
      MACE_NOT_IMPLEMENTED;
    }
  }

  ShowText("Create gpu computing kernel");
  if (!part_plan_->IsCpuOnly()) {
    const MemoryType opencl_mem_type
        = use_opencl_image ? MemoryType::GPU_IMAGE : MemoryType::GPU_BUFFER;
    opencl_kernel_ = CreateOpenCLMatMulKernel(opencl_mem_type);
  }

#if 0
  // Fill data to tensors.
  const std::vector<float> lhs_data = {1};
  const std::vector<float> rhs_data = {1};
  TensorDataFiller tensor_data_filler;
  if (use_opencl_image) {
    const OpenCLBufferType buffer_type = OpenCLBufferType::IN_OUT_CHANNEL;
    tensor_data_filler.FillImage(dev_context, gpu_context_.get(), input_.get(),
                                 buffer_type, 0, lhs_data);
    tensor_data_filler.FillImage(dev_context, gpu_context_.get(), rhs_.get(),
                                 buffer_type, 0, rhs_data);
  } else {
    tensor_data_filler.Fill(input_.get(), lhs_data);
    tensor_data_filler.Fill(rhs_.get(), rhs_data);
  }

  TensorDebugUtils::PrintTensorData(dev_context,
                                    gpu_context_.get(),
                                    input_.get(),
                                    OpenCLBufferType::IN_OUT_CHANNEL,
                                    0);
  TensorDebugUtils::PrintTensorData(dev_context,
                                    gpu_context_.get(),
                                    rhs_.get(),
                                    OpenCLBufferType::IN_OUT_CHANNEL,
                                    0);

  if (part_plan_->IsCpuOnly()) {
    tensor_data_filler.Fill(input_cpu_, lhs_data);
    tensor_data_filler.Fill(rhs_cpu_, rhs_data);
    TensorDebugUtils::PrintTensorData(dev_context,
                                      cpu_context_.get(),
                                      input_cpu_,
                                      OpenCLBufferType::IN_OUT_CHANNEL,
                                      0);
    TensorDebugUtils::PrintTensorData(dev_context,
                                      cpu_context_.get(),
                                      rhs_cpu_,
                                      OpenCLBufferType::IN_OUT_CHANNEL,
                                      0);
  }
#endif

  // TODO(fucheng): Warming up.

  return 0;
}

int CodlMatMulCpuGpuTestTask::EnqueueInputDataTransformKernel(
    StatsFuture *future) {
  //ShowText("Enqueue input data transform kernel");
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }

  if (!do_data_transform_) {
    return 0;
  }

  StatsFuture *old_future = gpu_context_->future();
  gpu_context_->set_future(future);

  const ops::MatMulPartPlan *matmul_part_plan =
      reinterpret_cast<ops::MatMulPartPlan *>(part_plan_.get());
  const OdimRanges *lhs_odim_ranges = matmul_part_plan->input_odim_ranges();
  const OdimRanges *rhs_odim_ranges = matmul_part_plan->rhs_odim_ranges();
  ops::OpenCLPartBufferTransformer lhs_transformer(MemoryType::GPU_IMAGE,
                                                   MemoryType::GPU_BUFFER);
  ops::OpenCLPartBufferTransformer rhs_transformer(MemoryType::GPU_IMAGE,
                                                   MemoryType::GPU_BUFFER);
  lhs_transformer.Transform(gpu_context_.get(),
                            input_,
                            OpenCLBufferType::IN_OUT_CHANNEL,
                            MemoryType::GPU_BUFFER,
                            0,
                            *lhs_odim_ranges,
                            input_cpu_);
  rhs_transformer.Transform(gpu_context_.get(),
                            rhs_,
                            OpenCLBufferType::IN_OUT_CHANNEL,
                            MemoryType::GPU_BUFFER,
                            0,
                            *rhs_odim_ranges,
                            rhs_cpu_);
  
  gpu_context_->set_future(old_future);

  return 0;
}

int CodlMatMulCpuGpuTestTask::EnqueueMapKernel(
    StatsFuture *future,
    StatsFuture *map_in_future) {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }

  StatsFuture *old_future = gpu_context_->future();
  gpu_context_->set_future(future);

  input_cpu_->MapBuffer(AllocatorMapType::AMT_READ_ONLY,
                        BlockFlag::BF_FALSE,
                        map_in_future);
  output_cpu_->MapBuffer(AllocatorMapType::AMT_WRITE_ONLY,
                         BlockFlag::BF_FALSE,
                         future);
  
  gpu_context_->set_future(old_future);

  return 0;
}

int CodlMatMulCpuGpuTestTask::EnqueueGpuComputeKerenl(
    StatsFuture *future) {
  if (part_plan_->IsCpuOnly()) {
    return 0;
  }

  StatsFuture *old_future = gpu_context_->future();
  gpu_context_->set_future(future);

  if (!part_plan_->IsGpuOnly()) {
    MatMulUtils::Validate(input_gpu_, rhs_gpu_, transpose_a_, transpose_b_);
    opencl_kernel_->Compute(
        gpu_context_.get(),
        input_gpu_, rhs_gpu_, output_gpu_,
        transpose_a_, transpose_b_);
  } else {
    MatMulUtils::Validate(input_, rhs_, transpose_a_, transpose_b_);
    opencl_kernel_->Compute(
        gpu_context_.get(),
        input_, rhs_, output_,
        transpose_a_, transpose_b_);
  }

  gpu_context_->set_future(old_future);

  return 0;
}

int CodlMatMulCpuGpuTestTask::EnqueueUnmapKernel(
    cl::UserEvent **event,
    StatsFuture *unmap_in_future,
    StatsFuture *unmap_out_future) {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }

  OpenCLRuntime *opencl_runtime =
      GetDeviceContext()->GetGpuDevice()->gpu_runtime()->opencl_runtime();
  OpenCLEventManager *event_manager = opencl_runtime->event_manager();
  *event = event_manager->CreateSingleUserEvent(
      opencl_runtime->context(),
      EventActionType::WAIT,
      EventOpType::TRANSFORM_OUTPUT);
  input_cpu_->UnmapBuffer(unmap_in_future);
  event_manager->InsertNullEvent(EventActionType::WAIT);
  output_cpu_->UnmapBuffer(unmap_out_future);
  return 0;
}

int CodlMatMulCpuGpuTestTask::EnqueueOutputDataTransformKernel(
    StatsFuture *future) {
  //ShowText("Enqueue output data transform kernel");
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }

  if (!do_data_transform_) {
    return 0;
  }

  StatsFuture *old_future = gpu_context_->future();
  gpu_context_->set_future(future);

  const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
  ops::OpenCLPartBufferTransformer transformer(MemoryType::GPU_BUFFER,
                                               MemoryType::GPU_IMAGE);
  transformer.Transform(gpu_context_.get(),
                        output_cpu_,
                        OpenCLBufferType::IN_OUT_CHANNEL,
                        MemoryType::GPU_IMAGE,
                        0,
                        *output_odim_ranges,
                        output_);
  
  gpu_context_->set_future(old_future);

  return 0;
}

int CodlMatMulCpuGpuTestTask::RunCpuComputeKernel() {
  if (part_plan_->IsGpuOnly()) {
    return 0;
  }

  cpu_kernel_->Compute(cpu_context_.get(),
                       input_cpu_,
                       rhs_cpu_,
                       bias_cpu_,
                       transpose_a_,
                       transpose_b_,
                       output_cpu_);

  return 0;
}

int CodlMatMulCpuGpuTestTask::PostProcess() {
#if 0
  TestDeviceContext *dev_context = GetDeviceContext();
  if (!part_plan_->IsGpuOnly()) {
    TensorDebugUtils::PrintTensorData(dev_context,
                                      cpu_context_.get(),
                                      input_cpu_,
                                      OpenCLBufferType::IN_OUT_CHANNEL,
                                      0);
    TensorDebugUtils::PrintTensorData(dev_context,
                                      cpu_context_.get(),
                                      rhs_cpu_,
                                      OpenCLBufferType::IN_OUT_CHANNEL,
                                      0);
    TensorDebugUtils::PrintTensorData(dev_context,
                                      cpu_context_.get(),
                                      output_cpu_,
                                      OpenCLBufferType::IN_OUT_CHANNEL,
                                      0);
  }

  if (!part_plan_->IsCpuOnly()) {
    TensorDebugUtils::PrintTensorData(dev_context,
                                      gpu_context_.get(),
                                      output_.get(),
                                      OpenCLBufferType::IN_OUT_CHANNEL,
                                      0);
  }
#endif

  return 0;
}

}  // namespace mace
