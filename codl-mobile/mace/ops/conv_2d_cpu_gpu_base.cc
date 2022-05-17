
#include <fstream>
#include <regex>

#include "mace/ops/conv_2d.h"

namespace mace {
namespace ops {

#ifndef CODL_ENABLE_MACE_CONV2D_GPU
#ifdef MACE_ENABLE_OPENCL

class PaddingUtils {
public:
  static void ComputePaddedShape(const std::vector<index_t> &shape,
                                 const int *paddings,
                                 std::vector<index_t> *output_shape) {
#ifdef CODL_ENABLE_DEBUG_INFO
    MACE_CHECK(shape.size() == 4);
    LOG(INFO) << "Compute padded shape...";
    LOG(INFO) << "input_shape " << VectorToString<index_t>(shape)
              << ", paddings " << VectorToString<int>({paddings[0], paddings[1]});
#endif

    output_shape->push_back(shape[0]);
    output_shape->push_back(shape[1] + paddings[0]);
    output_shape->push_back(shape[2] + paddings[1]);
    output_shape->push_back(shape[3]);

#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "padded_shape " << VectorToString<index_t>(*output_shape);
#endif
  }
};

void Conv2dGpuFloatOp::InitCpu(OpConstructContext *context) {
  MACE_UNUSED(context);
  // Initialize temporary tensor indices.
  MACE_CHECK(kNumTempTensorsConv2D == LAST_TENSOR_INDEX);
  for (size_t i = 0; i < kNumTempTensorsConv2D; i ++) {
    temp_tensor_indices_[i] = -1;
  }
}

void Conv2dGpuFloatOp::InitGpu(OpConstructContext *context) {
  if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
    mem_type_ = MemoryType::GPU_IMAGE;
    kernel_ = make_unique<opencl::image::Conv2dKernel>();
  } else {
    mem_type_ = MemoryType::GPU_BUFFER;
    kernel_ = make_unique<opencl::buffer::Conv2dKernel>();
  }
  
  // Transform filter tensor to target format.
  if ((wino_block_size_ == 2 || wino_block_size_ == 4) &&
      (kernel_->CheckUseWinograd(
        context->device()->gpu_runtime()->opencl_runtime(),
        context->workspace()->GetTensor(
            operator_def_->input(1))->shape(),
        std::vector<index_t>(operator_def_->output_shape(0).dims().begin(),
                             operator_def_->output_shape(0).dims().end()),
        strides_.data(),
        dilations_.data(),
        &wino_block_size_))) {
    MACE_CHECK(TransformFilter(
        context, operator_def_.get(), 1,
        OpenCLBufferType::WINOGRAD_FILTER, mem_type_, wino_block_size_)
                   == MaceStatus::MACE_SUCCESS);
  } else {
    wino_block_size_ = 0;
    MACE_CHECK(TransformFilter(
        context, operator_def_.get(), 1,
        OpenCLBufferType::CONV2D_FILTER, mem_type_)
                   == MaceStatus::MACE_SUCCESS);
  }
  if (operator_def_->input_size() > 2) {
    MACE_CHECK(TransformFilter(
        context, operator_def_.get(), 2, OpenCLBufferType::ARGUMENT, mem_type_)
                   == MaceStatus::MACE_SUCCESS);
  }

  if (wino_block_size_ == 2 || wino_block_size_ == 4) {
    LOG(INFO) << "Conv2d OpenCL Image Winograd Block Size: "
              << wino_block_size_;
  }

  // Initialize pad kernel.
  if (!paddings_.empty()) {
    MACE_CHECK(paddings_.size() == 2);
    const int padding_top    = paddings_[0] / 2;
    const int padding_left   = paddings_[1] / 2;
    const int padding_bottom = paddings_[0] - padding_top;
    const int padding_right  = paddings_[1] - padding_left;
    const std::vector<int> paddings_v2 = {0, 0, padding_top, padding_bottom,
                                          padding_left, padding_right, 0, 0};
    const std::vector<int> *paddings = new std::vector<int>(paddings_v2);
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      pad_kernel_ = make_unique<opencl::image::PadKernel>(
          PadType::CONSTANT, *paddings, 0.0f);
    } else {
      pad_kernel_ = make_unique<opencl::buffer::PadKernel>(
          PadType::CONSTANT, *paddings, 0.0f);
    }
  } else if (padding_type_ == Padding::SAME) {
    const std::vector<int> paddings = {0, 0, 0, 0, 0, 0, 0, 0};
    pad_kernel_ = make_unique<opencl::image::PadKernel>(
          PadType::CONSTANT, paddings, 0.0f);
  } else if (padding_type_ == Padding::FULL) {
    MACE_CHECK(false, "Unsupported padding type FULL");
  } else {
    pad_kernel_ = nullptr;
  }
}

#ifdef MACE_ENABLE_NEON
arm::fp32::Conv2dBase* Conv2dGpuFloatOp::CreateNEONConv2dDelegator(
    const Tensor *input,
    const Tensor *filter,
    const std::vector<int> strides,
    const std::vector<int> paddings,
    const Padding padding_type,
    const std::vector<int> dilations) {

  arm::fp32::Conv2dBase *conv2d_delegator = nullptr;
  
  // the following params are used to decide which conv delegator to use
  const index_t stride_h = strides[0];
  const index_t stride_w = strides[1];
  const index_t dilation_h = dilations[0];
  const index_t dilation_w = dilations[1];
  const index_t filter_h = filter->dim(2);
  const index_t filter_w = filter->dim(3);
  const index_t input_channels = input->dim(1);
  const index_t channels = filter->dim(0);

  // NOTE: delegator is fixed after first round of running,
  // although winograd depends on input params.
  // We do not support changeable filter for now.
  if (filter_h   == 1 && filter_w   == 1 &&
      stride_h   == 1 && stride_w   == 1 &&
      dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK1x1(
        paddings, padding_type);
  } else if (filter_h   == 1 && filter_w   == 1 &&
             stride_h   == 2 && stride_w   == 2 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK1x1S2(
        paddings, padding_type);
  } else if (filter_h   == 3 && filter_w   == 3 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    // ic >= 8 && oc >= 8
    if (input_channels >= 8 && channels >= 8) {
      conv2d_delegator = new arm::fp32::Conv2dK3x3Winograd(
          paddings, padding_type);
    } else {
      conv2d_delegator = new arm::fp32::Conv2dK3x3S1(
          paddings, padding_type);
    }
  } else if (filter_h   == 3 && filter_w   == 3 &&
             stride_h   == 2 && stride_w   == 2 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK3x3S2(
        paddings, padding_type);
  } else if (filter_h   == 5 && filter_w   == 5 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK5x5S1(
        paddings, padding_type);
  } else if (filter_h   == 7 && filter_w   == 7 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK7x7S1(
        paddings, padding_type);
  } else if (filter_h   == 7 && filter_w   == 7 &&
             stride_h   == 2 && stride_w   == 2 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK7x7S2(
        paddings, padding_type);
  } else if (filter_h   == 7 && filter_w   == 7 &&
             stride_h   == 3 && stride_w   == 3 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK7x7S3(
        paddings, padding_type);
  } else if (filter_h   == 1 && filter_w   == 7 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK1x7S1(
        paddings, padding_type);
  } else if (filter_h   == 7 && filter_w   == 1 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK7x1S1(
        paddings, padding_type);
  } else if (filter_h   == 9 && filter_w   == 9 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK9x9S1(
        paddings, padding_type);
  } else if (filter_h   == 1 && filter_w   == 15 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK1x15S1(
        paddings, padding_type);
  } else if (filter_h   == 15 && filter_w   == 1 &&
             stride_h   == 1  && stride_w   == 1 &&
             dilation_h == 1  && dilation_w == 1) {
    conv2d_delegator = new arm::fp32::Conv2dK15x1S1(
        paddings, padding_type);
  } else {
    LOG(WARNING) << "Use general kernel"
                 << ": input shape "
                 << VectorToString<index_t>(input->shape())
                 << ", filter shape "
                 << VectorToString<index_t>(filter->shape())
                 << ", strides " << VectorToString<int>(strides)
                 << ", paddings " << VectorToString<int>(paddings)
                 << ", padding type " << PaddingTypeToString(padding_type)
                 << ", dilations " << VectorToString<int>(dilations);
    conv2d_delegator = new arm::fp32::Conv2dGeneral(
        strides,
        dilations,
        paddings,
        padding_type);
  }
  
  return conv2d_delegator;
}
#endif  // MACE_ENABLE_NEON

const Tensor *Conv2dGpuFloatOp::RunPadInput(OpContext *context) {
  const Tensor *input = this->Input(INPUT);
  const Tensor *filter = this->Input(FILTER);
  Tensor *padded_input = nullptr;
  TensorManageUtil *tensor_manage_util
      = context->workspace()->tensor_manage_util();
  if (!paddings_.empty()) {
    std::vector<index_t> padded_shape;
    DataType input_dt = input->dtype();
    if (input_dt == DataType::DT_HALF)
      input_dt = DataType::DT_FLOAT;
    PaddingUtils::ComputePaddedShape(
        input->shape(), paddings_.data(), &padded_shape);
    padded_input = tensor_manage_util->CreateOrReshapeOpenCLImageTensor(
        padded_shape, input_dt, input->is_weight(),
        input->data_format(), OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("padded_input"),
        AllocateType::ALLOC_TYPE_GROW,
        &temp_tensor_indices_[GPU_PADDED_INPUT]);
    pad_kernel_->Compute(context, input, padded_input);
  } else if (padding_type_ == Padding::SAME &&
             filter->dim(2) > 1 && filter->dim(3) > 1) {
    const std::vector<int> paddings = part_plan_->paddings();
    std::vector<index_t> padded_shape;
    DataType input_dt = input->dtype();
    if (input_dt == DataType::DT_HALF)
      input_dt = DataType::DT_FLOAT;
    PaddingUtils::ComputePaddedShape(
        input->shape(), paddings.data(), &padded_shape);
    padded_input = tensor_manage_util->CreateOrReshapeOpenCLImageTensor(
        padded_shape, input_dt, input->is_weight(),
        input->data_format(), OpenCLBufferType::IN_OUT_CHANNEL,
        std::string("padded_input"),
        AllocateType::ALLOC_TYPE_GROW,
        &temp_tensor_indices_[GPU_PADDED_INPUT]);

    const int padding_top    = paddings[0] / 2;
    const int padding_left   = paddings[1] / 2;
    const int padding_bottom = paddings[0] - padding_top;
    const int padding_right  = paddings[1] - padding_left;
    const std::vector<int> paddings_v2
        = {0, 0, padding_top, padding_bottom,
          padding_left, padding_right, 0, 0};
    pad_kernel_->set_paddings(&paddings_v2);
    pad_kernel_->Compute(context, input, padded_input);
  }

  return (padded_input != nullptr) ? padded_input : input;
}

MaceStatus Conv2dGpuFloatOp::TransformWeightGpuToCpu(
    OpContext *context,
    const Tensor *src,
    Tensor *dst) {
  OpenCLBufferTransformer(MemoryType::GPU_IMAGE,
                          MemoryType::CPU_BUFFER)
      .Transform(context, src, OpenCLBufferType::CONV2D_FILTER,
                 MemoryType::CPU_BUFFER, wino_block_size_, dst);
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dGpuFloatOp::TransformBiasGpuToCpu(
    OpContext *context,
    const Tensor *src,
    Tensor *dst) {
  OpenCLBufferTransformer(MemoryType::GPU_IMAGE,
                          MemoryType::CPU_BUFFER)
      .Transform(context, src, OpenCLBufferType::ARGUMENT,
                 MemoryType::CPU_BUFFER, wino_block_size_, dst);
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dGpuFloatOp::MakePartitionPlan(OpContext *context) {
  const Tensor *input  = this->Input(INPUT);
  const Tensor *filter = this->Input(FILTER);

  PartitionDim partition_dim = context->part_run_config()->dim_type();
  float partition_ratio = context->part_run_config()->ratio();
  if (partition_ratio == kRatioFromConfigFile ||
      partition_ratio == kRatioFromPredictor) {
    partition_dim = static_cast<PartitionDim>(context->partition_configer()->dim());
    partition_ratio = context->partition_configer()->ratio();
  }

  if (part_plan_) {
    MACE_CHECK(context->op_runtime_mode() == OpRuntimeMode::RUNNING);
    if (part_plan_->CheckPlanChanged(partition_dim, partition_ratio)) {
      const DataFormat df = DataFormat::NCHW;
      part_plan_.reset(new ConvPool2dPartPlan(partition_dim, partition_ratio, df));
      PartitionResult partition_result;
      do {
        partition_result = part_plan_->Make(input->shape(),
                                            filter->shape(),
                                            strides_,
                                            dilations_,
                                            padding_type_,
                                            paddings_);
      } while (partition_result == PartitionResult::PARTITION_REDO);
      part_plan_->Show();
    }
  } else {
    const DataFormat df = DataFormat::NCHW;
    part_plan_.reset(new ConvPool2dPartPlan(partition_dim, partition_ratio, df));
    PartitionResult partition_result;
    do {
      partition_result = part_plan_->Make(input->shape(),
                                          filter->shape(),
                                          strides_,
                                          dilations_,
                                          padding_type_,
                                          paddings_);
    } while (partition_result == PartitionResult::PARTITION_REDO);
    part_plan_->Show();
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dGpuFloatOp::PrepareTemporaryTensors(OpContext *context) {
  const Tensor *input = this->Input(INPUT);
  const Tensor *filter = this->Input(FILTER);
  const Tensor *bias = this->Input(BIAS);
  Tensor *output = this->Output(OUTPUT);

  kernel_->ResizeOutputTensor(input, filter, strides_.data(),
                              padding_type_, paddings_,
                              dilations_.data(), output);

  if (part_plan_->IsGpuOnly()) {
    return MaceStatus::MACE_SUCCESS;
  }

  RunPadInput(context);
  
  TensorManageUtil *tensor_manager = context->workspace()->tensor_manage_util();
  const Tensor *padded_input
      = tensor_manager->GetTensor(temp_tensor_indices_[GPU_PADDED_INPUT]);

  const PartitionDim partition_dim = part_plan_->dimension();

  if (!part_plan_->IsCpuOnly()) {
    LOG(INFO) << "Prepare gpu tensors";
    index_t in_out_dim_idx = (partition_dim == DIM_OUTPUT_CHANNEL) ? C_NHWC : H_NHWC;
    tensor_manager->CreateOrReshapePartTensor(
        padded_input, partition_dim,
        0, part_plan_->gpu_input_part_shape()[in_out_dim_idx],
        /* reshape */ true, &temp_tensor_indices_[GPU_INPUT]);
    tensor_manager->CreateOrReshapePartTensor(
        filter, partition_dim,
        0, part_plan_->gpu_filter_part_shape()[O_OIHW],
        /* reshape */ true, &temp_tensor_indices_[GPU_FILTER]);
    tensor_manager->CreateOrReshapePartTensor(
        output, partition_dim,
        0, part_plan_->gpu_output_part_shape()[in_out_dim_idx],
        /* reshape */ false, &temp_tensor_indices_[GPU_OUTPUT]);
  }

  LOG(INFO) << "Prepare cpu tensors";
  const DataFormat in_out_data_format = DataFormat::NCHW;
  const DeviceType in_out_device_type = DeviceType::GPU;
  const DeviceType weights_device_type = DeviceType::CPU;
  bool in_out_map_buffer_hint = false;
  bool weights_map_gpu_buffer_hint = false;
  const AllocatorMapType in_map_type = AllocatorMapType::AMT_READ_ONLY;
  const AllocatorMapType out_map_type = AllocatorMapType::AMT_WRITE_ONLY;
  Tensor *cpu_filter_v1 = nullptr, *cpu_bias_v1 = nullptr;
  tensor_manager->CreateOrReshapeTensor(
      part_plan_->cpu_input_part_shape(),
      DT_FLOAT, false, in_out_data_format,
      in_out_device_type, std::string("cpu_input"),
      in_out_map_buffer_hint, in_map_type,
      CpuBufferIdx::BUF_IDX_IN, AllocateType::ALLOC_TYPE_REUSE,
      &temp_tensor_indices_[CPU_INPUT]);
  cpu_filter_v1 = tensor_manager->CreateOrReshapeTensor(
      filter->shape(),
      DT_FLOAT, filter->is_weight(), DataFormat::OIHW,
      weights_device_type, std::string("cpu_filter"),
      weights_map_gpu_buffer_hint, in_map_type,
      CpuBufferIdx::BUF_IDX_WEIGHTS, AllocateType::ALLOC_TYPE_GROW,
      &temp_tensor_indices_[CPU_FILTER_V1]);
  if (bias != nullptr) {
    cpu_bias_v1 = tensor_manager->CreateOrReshapeTensor(
        bias->shape(),
        DT_FLOAT, bias->is_weight(), DataFormat::NONE,
        weights_device_type, std::string("cpu_bias"),
        weights_map_gpu_buffer_hint, in_map_type,
        CpuBufferIdx::BUF_IDX_WEIGHTS, AllocateType::ALLOC_TYPE_GROW,
        &temp_tensor_indices_[CPU_BIAS_V1]);
  }
  tensor_manager->CreateOrReshapeTensor(
      part_plan_->cpu_output_part_shape(),
      DT_FLOAT, false, in_out_data_format,
      in_out_device_type, std::string("cpu_output"),
      in_out_map_buffer_hint, out_map_type,
      CpuBufferIdx::BUF_IDX_OUT, AllocateType::ALLOC_TYPE_REUSE,
      &temp_tensor_indices_[CPU_OUTPUT]);

  // Transform filter and bias tensor if it is the first run.
  if (is_first_time_run_) {
    TransformWeightGpuToCpu(context, filter, cpu_filter_v1);
    if (bias != nullptr) {
      TransformBiasGpuToCpu(context, bias, cpu_bias_v1);
    }
    is_first_time_run_ = false;
  }

  tensor_manager->CreateOrReshapeCpuConv2dWeightsTensor(
      cpu_filter_v1, partition_dim,
      part_plan_->cpu_filter_part_shape()[O_OIHW],
      &temp_tensor_indices_[CPU_FILTER_V2]);
  if (bias != nullptr) {
    tensor_manager->CreateOrReshapeCpuConv2dWeightsTensor(
        cpu_bias_v1, partition_dim,
        part_plan_->cpu_filter_part_shape()[O_OIHW],
        &temp_tensor_indices_[CPU_BIAS_V2]);
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dGpuFloatOp::EnqueueInputDataTransform(
    OpContext *context,
    StatsFuture *future) {
  MACE_UNUSED(future);

  if (part_plan_->IsGpuOnly()) {
    return MaceStatus::MACE_SUCCESS;
  }

  TensorManageUtil *tensor_manager = context->workspace()->tensor_manage_util();
  const Tensor *input
      = tensor_manager->GetTensor(temp_tensor_indices_[GPU_PADDED_INPUT]);
  Tensor *cpu_input
      = tensor_manager->GetTensor(temp_tensor_indices_[CPU_INPUT]);
  const OdimRanges *input_odim_ranges = part_plan_->input_odim_ranges();
  OpenCLPartBufferTransformer transformer(MemoryType::GPU_IMAGE,
                                          MemoryType::GPU_BUFFER);

  MACE_RETURN_IF_ERROR(transformer.Transform(
      context, input, OpenCLBufferType::IN_OUT_CHANNEL,
      MemoryType::GPU_BUFFER, wino_block_size_, *input_odim_ranges, cpu_input));

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dGpuFloatOp::EnqueueMap(
    OpContext *context,
    StatsFuture *future) {
  if (part_plan_->IsGpuOnly()) {
    return MaceStatus::MACE_SUCCESS;
  }

  TensorManageUtil *tensor_manager = context->workspace()->tensor_manage_util();
  Tensor *cpu_input
      = tensor_manager->GetTensor(temp_tensor_indices_[CPU_INPUT]);
  Tensor *cpu_output
      = tensor_manager->GetTensor(temp_tensor_indices_[CPU_OUTPUT]);

  cpu_input->MapBuffer(AMT_READ_ONLY, BF_FALSE);
  cpu_output->MapBuffer(AMT_WRITE_ONLY, BF_FALSE, future);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dGpuFloatOp::EnqueueGpuCompute(
    OpContext *context,
    StatsFuture *future) {
  MACE_UNUSED(future);

  if (part_plan_->IsGpuOnly()) {
    return RunGpu(context);
  }

  if (part_plan_->IsCpuOnly()) {
    return MaceStatus::MACE_SUCCESS;
  }

  if (!part_plan_->IsCpuOnly()) {
    TensorManageUtil *tensor_manager = context->workspace()->tensor_manage_util();
    const Tensor *input
        = tensor_manager->GetTensor(temp_tensor_indices_[GPU_INPUT]);
    const Tensor *filter
        = tensor_manager->GetTensor(temp_tensor_indices_[GPU_FILTER]);
    const Tensor *bias = this->Input(BIAS);
    Tensor *output
        = tensor_manager->GetTensor(temp_tensor_indices_[GPU_OUTPUT]);

    return RunGpu(context, input, filter, bias, output);
  } else {
    return MaceStatus::MACE_SUCCESS;
  }
}

MaceStatus Conv2dGpuFloatOp::EnqueueUnmap(
    OpContext *context,
    cl::UserEvent **event,
    StatsFuture *future) {
  MACE_UNUSED(future);

  if (part_plan_->IsGpuOnly()) {
    return MaceStatus::MACE_SUCCESS;
  }

  auto opencl_runtime = context->device()->gpu_runtime()->opencl_runtime();
  OpenCLEventManager *event_manager = opencl_runtime->event_manager();

  if (event != nullptr) {
    *event = event_manager->CreateSingleUserEvent(
            opencl_runtime->context(),
            EventActionType::WAIT,
            EventOpType::TRANSFORM_OUTPUT);
  }

  TensorManageUtil *tensor_manager = context->workspace()->tensor_manage_util();
  Tensor *cpu_input
      = tensor_manager->GetTensor(temp_tensor_indices_[CPU_INPUT]);
  Tensor *cpu_output
      = tensor_manager->GetTensor(temp_tensor_indices_[CPU_OUTPUT]);
  
  cpu_input->UnmapBuffer();

  if (event != nullptr) {
    event_manager->InsertNullEvent(EventActionType::WAIT);
  }
  
  cpu_output->UnmapBuffer();

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dGpuFloatOp::EnqueueOutputDataTransform(
    OpContext *context,
    StatsFuture *future) {
  MACE_UNUSED(future);

  if (part_plan_->IsGpuOnly()) {
    return MaceStatus::MACE_SUCCESS;
  }

  TensorManageUtil *tensor_manager = context->workspace()->tensor_manage_util();
  const Tensor *cpu_output
      = tensor_manager->GetTensor(temp_tensor_indices_[CPU_OUTPUT]);
  Tensor *output
      = tensor_manager->GetTensor(temp_tensor_indices_[GPU_OUTPUT]);

  const OdimRanges *output_odim_ranges = part_plan_->output_odim_ranges();
  OpenCLPartBufferTransformer transformer(MemoryType::GPU_BUFFER,
                                          MemoryType::GPU_IMAGE);
  MACE_RETURN_IF_ERROR(transformer.Transform(
      context, cpu_output, OpenCLBufferType::IN_OUT_CHANNEL,
      MemoryType::GPU_IMAGE, wino_block_size_, *output_odim_ranges, output));

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dGpuFloatOp::RunCpuCompute(OpContext *context) {
  if (part_plan_->IsGpuOnly()) {
    return MaceStatus::MACE_SUCCESS;
  }

  TensorManageUtil *tensor_manager = context->workspace()->tensor_manage_util();
  const Tensor *input
      = tensor_manager->GetTensor(temp_tensor_indices_[CPU_INPUT]);
  const Tensor *filter
      = tensor_manager->GetTensor(temp_tensor_indices_[CPU_FILTER_V2]);
  const Tensor *bias
      = tensor_manager->GetTensor(temp_tensor_indices_[CPU_BIAS_V2]);
  Tensor *output
      = tensor_manager->GetTensor(temp_tensor_indices_[CPU_OUTPUT]);

  return RunCpu(context, input, filter, bias, output);
}

#endif  // MACE_ENABLE_OPENCL
#endif  // CODL_ENABLE_MACE_CONV2D_GPU

}  // namespace ops
}  // namespace mace
