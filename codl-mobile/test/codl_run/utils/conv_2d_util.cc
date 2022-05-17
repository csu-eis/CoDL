
#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/common/gemmlowp_util.h"
#include "mace/ops/arm/q8/quantization_util.h"
#endif

#include "test/codl_run/utils/conv_2d_util.h"

#define CODL_ENABLE_WINOGRAD

namespace mace {

#ifdef MACE_ENABLE_NEON
ops::arm::fp32::Conv2dBase* CreateNEONConv2dDelegator(
    const Tensor *input,
    const Tensor *filter,
    const std::vector<int> &strides,
    const std::vector<int> &paddings,
    const Padding padding_type,
    const std::vector<int> &dilations) {

  if (input == nullptr || filter == nullptr) {
    VLOG(2) << "Create neon conv2d delegator failed"
            << ", input or filter tensor is null";
    return nullptr;
  }

  MACE_CHECK(input != nullptr);
  MACE_CHECK(filter != nullptr);
  MACE_CHECK(strides.size() == 2);
  MACE_CHECK(paddings.size() == 0 || paddings.size() == 2);
  MACE_CHECK(dilations.size() == 2);
      
  ops::arm::fp32::Conv2dBase *conv2d_delegator = nullptr;
    
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
    conv2d_delegator = new ops::arm::fp32::Conv2dK1x1(
        paddings, padding_type);
  } else if (filter_h   == 3 && filter_w   == 3 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    if (input_channels >= 8 && channels >= 8) {
#ifdef CODL_ENABLE_WINOGRAD
      conv2d_delegator = new ops::arm::fp32::Conv2dK3x3Winograd(
          paddings, padding_type);
#else
      conv2d_delegator = new ops::arm::fp32::Conv2dK3x3S1(
          paddings, padding_type);
#endif
    } else {
      conv2d_delegator = new ops::arm::fp32::Conv2dK3x3S1(
          paddings, padding_type);
    }
  } else if (filter_h   == 3 && filter_w   == 3 &&
             stride_h   == 2 && stride_w   == 2 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new ops::arm::fp32::Conv2dK3x3S2(
        paddings, padding_type);
  } else if (filter_h   == 5 && filter_w   == 5 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new ops::arm::fp32::Conv2dK5x5S1(
        paddings, padding_type);
  } else if (filter_h   == 7 && filter_w   == 7 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new ops::arm::fp32::Conv2dK7x7S1(
        paddings, padding_type);
  } else if (filter_h   == 7 && filter_w   == 7 &&
             stride_h   == 2 && stride_w   == 2 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new ops::arm::fp32::Conv2dK7x7S2(
        paddings, padding_type);
  } else if (filter_h   == 7 && filter_w   == 7 &&
             stride_h   == 3 && stride_w   == 3 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new ops::arm::fp32::Conv2dK7x7S3(
        paddings, padding_type);
  } else if (filter_h   == 1 && filter_w   == 7 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new ops::arm::fp32::Conv2dK1x7S1(
        paddings, padding_type);
  } else if (filter_h   == 7 && filter_w   == 1 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new ops::arm::fp32::Conv2dK7x1S1(
        paddings, padding_type);
  } else if (filter_h   == 9 && filter_w   == 9 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new ops::arm::fp32::Conv2dK9x9S1(
        paddings, padding_type);
  } else if (filter_h   == 1 && filter_w   == 15 &&
             stride_h   == 1 && stride_w   == 1 &&
             dilation_h == 1 && dilation_w == 1) {
    conv2d_delegator = new ops::arm::fp32::Conv2dK1x15S1(
        paddings, padding_type);
  } else if (filter_h   == 15 && filter_w   == 1 &&
             stride_h   == 1  && stride_w   == 1 &&
             dilation_h == 1  && dilation_w == 1) {
    conv2d_delegator = new ops::arm::fp32::Conv2dK15x1S1(
        paddings, padding_type);
  } else {
    conv2d_delegator = new ops::arm::fp32::Conv2dGeneral(
        strides,
        dilations,
        paddings,
        padding_type);
  }

  if (conv2d_delegator == nullptr) {
    VLOG(2) << "Create neon conv2d delegator failed, unknown reason";
    return nullptr;
  }

#if 0
  LOG(INFO) << "Create neon conv2d delegator successfully"
            << " (filter=[" << filter_h << "," << filter_w << "],"
            << " strides=[" << stride_h << "," << stride_w << "])";
#endif
  
  return conv2d_delegator;
}

#ifdef CODL_ENABLE_WINOGRAD
#undef CODL_ENABLE_WINOGRAD
#endif

MaceStatus Conv2dCpuFloatKernel::Compute(const OpContext *context,
                                         const Tensor *input,
                                         const Tensor *filter,
                                         const Tensor *bias,
                                         Tensor *output) {
  conv2d_delegator_->Compute(context, input, filter, output);
  bias_add_delegator_.Compute(context, output, bias, output);
  activation_delegator_.Compute(context, output, output);
  return MaceStatus::MACE_SUCCESS;
}

#else  // MACE_ENABLE_NEON

MaceStatus Conv2dCpuFloatKernel::Compute(const OpContext *context,
                                         const Tensor *input,
                                         const Tensor *filter,
                                         const Tensor *bias,
                                         Tensor *output) {
  ref_conv2d_delegator_->Compute(context, input, filter, output);
  return MaceStatus::MACE_SUCCESS;
}

#endif // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_QUANTIZE

template<>
void Conv2dCpuUint8Kernel::Im2col<uint8_t>(
    const OpContext *context,
    const uint8_t *in_data, const std::vector<index_t> &in_shape,
    const index_t filter_h, const index_t filter_w, const index_t stride_h,
    const index_t stride_w, const uint8_t zero_point, const int pad_height,
    const int pad_width, const std::vector<index_t> &out_shape,
    const index_t depth, uint8_t *im2col_data) {
    const index_t input_row_size = in_shape[2] * in_shape[3];
    const index_t patch_row_size = filter_w * in_shape[3];

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute3D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1,
                            index_t start2, index_t end2, index_t step2) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t h = start1; h < end1; h += step1) {
        for (index_t w = start2; w < end2; w += step2) {
          // Reshape a patch of input to column, which is corresponding to
          // a column of output(:, column).
          const index_t ih_begin = h * stride_h - (pad_height >> 1);
          const index_t ih_end = ih_begin + filter_h;
          const index_t iw_begin = w * stride_w - (pad_width >> 1);
          const index_t iw_end = iw_begin + filter_w;
          // gate height and width to separate padding
          const index_t ih_begin_gated = std::max<index_t>(0, ih_begin);
          const index_t ih_end_gated = std::min<index_t>(ih_end, in_shape[1]);
          const index_t iw_begin_gated = std::max<index_t>(0, iw_begin);
          const index_t iw_end_gated = std::min<index_t>(iw_end, in_shape[2]);
          const index_t pad_top = std::max<index_t>(0, -ih_begin);
          const index_t pad_bottom = ih_end - ih_end_gated;
          const index_t pad_left = std::max<index_t>(0, -iw_begin);
          const index_t pad_right = iw_end - iw_end_gated;
          index_t im2col_column_offset =
              ((b * out_shape[1] + h) * out_shape[2] + w) * depth;

          // fill in padding top
          if (pad_top > 0) {
            std::fill_n(im2col_data + im2col_column_offset,
                        pad_top * patch_row_size, zero_point);
          }

          const index_t patch_row_size_gated =
              std::min(filter_w - pad_left,
                       in_shape[2] - iw_begin_gated) * in_shape[3];
          MACE_CHECK(patch_row_size_gated ==
              ((filter_w - (pad_left + pad_right)) * in_shape[3]));
          const index_t pad_left_size = pad_left * in_shape[3];
          const index_t pad_right_size = pad_right * in_shape[3];
          index_t im2col_offset = im2col_column_offset +
              (pad_top * filter_w + pad_left) * in_shape[3];
          index_t
              in_offset = ((b * in_shape[1] + ih_begin_gated) * in_shape[2]
              + iw_begin_gated) * in_shape[3];

          // fill in effective rows
          for (index_t ih = ih_begin_gated; ih < ih_end_gated; ++ih) {
            // fill in padding left
            if (pad_left > 0) {
              const index_t left_offset = im2col_offset - pad_left_size;
              std::fill_n(im2col_data + left_offset,
                          pad_left_size,
                          zero_point);
            }
            // copy effective data
            std::copy_n(in_data + in_offset, patch_row_size_gated,
                        im2col_data + im2col_offset);
            // fill in padding right
            if (pad_right > 0) {
              const index_t
                  right_offset = im2col_offset + patch_row_size_gated;
              std::fill_n(im2col_data + right_offset, pad_right_size,
                          zero_point);
            }
            in_offset += input_row_size;
            im2col_offset += patch_row_size;
          }

          // fill in padding bottom
          if (pad_bottom > 0) {
            const index_t pad_bottom_size = pad_bottom * patch_row_size;
            const index_t bottom_offset =
                im2col_column_offset + depth - pad_bottom_size;
            std::fill_n(im2col_data + bottom_offset, pad_bottom_size,
                        zero_point);
          }
        }
      }
    }
  }, 0, out_shape[0], 1, 0, out_shape[1], 1, 0, out_shape[2], 1);
}

MaceStatus Conv2dCpuUint8Kernel::Compute(const OpContext *context,
                                         const Tensor *input,
                                         const Tensor *filter,
                                         const Tensor *bias,
                                         Tensor *output) {
  int64_t t0;
  std::vector<double> durations;
  auto gemm_context = context->device()->cpu_runtime()->GetGemmlowpContext();
  MACE_CHECK_NOTNULL(gemm_context);

  index_t batch = output->dim(0);
  index_t height = output->dim(1);
  index_t width = output->dim(2);
  index_t channels = output->dim(3);
  index_t input_batch = input->dim(0);
  index_t input_channels = input->dim(3);
  index_t filter_h = filter->dim(1);
  index_t filter_w = filter->dim(2);
  index_t stride_h = strides_[0];
  index_t stride_w = strides_[1];
  const index_t depth = input_channels * filter_h * filter_w;
  const index_t columns = batch * height * width;

  VLOG(2) << "input scale/zero: " << input->scale() << ", "
          << input->zero_point();
  VLOG(2) << "filter scale/zero: " << filter->scale() << ", "
          << filter->zero_point();
  if (bias) {
    VLOG(2) << "bias scale/zero: " << bias->scale() << ", "
            << bias->zero_point();
  }
  VLOG(2) << "output scale/zero: " << output->scale() << ", "
          << output->zero_point();

  MACE_CHECK(filter->dim(0) == channels, filter->dim(0), " != ", channels);
  MACE_CHECK(filter->dim(3) == input_channels, filter->dim(3), " != ",
             input_channels);
  MACE_CHECK(batch == input_batch, "Input/Output batch size mismatch");

  auto input_data = input->data<uint8_t>();
  auto filter_data = filter->data<uint8_t>();
  auto output_data = output->mutable_data<uint8_t>();
  auto bias_data = ops::GetBiasData(bias,
                                    input->scale(),
                                    filter->scale(),
                                    channels,
                                    &bias_);

  auto gemm_input_data = input_data;
  std::unique_ptr<Tensor> im2col;
  bool im2col_required =
      filter_h != 1 || filter_w != 1 || stride_h != 1 || stride_w != 1;
  t0 = NowMicros();
  if (im2col_required) {
    index_t im2col_size = depth * columns * sizeof(uint8_t);
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    scratch->GrowSize(im2col_size);
    im2col = make_unique<Tensor>(scratch->Scratch(im2col_size), DT_UINT8);
    uint8_t *im2col_data = im2col->mutable_data<uint8_t>();
    Im2col(context, input_data, input->shape(), filter_h, filter_w, stride_h,
           stride_w, static_cast<uint8_t>(input->zero_point()),
           paddings_[0], paddings_[1], output->shape(), depth, im2col_data);
    gemm_input_data = im2col_data;
  }
  durations.push_back((NowMicros() - t0) / 1000.0);

  t0 = NowMicros();
  const int gemm_filter_rows = static_cast<int>(channels);
  const int gemm_filter_cols = static_cast<int>(depth);
  const int gemm_input_rows = static_cast<int>(depth);
  const int gemm_input_cols = static_cast<int>(columns);
  const int gemm_output_rows = static_cast<int>(channels);
  const int gemm_output_cols = static_cast<int>(columns);
  gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor>
      filter_matrix(filter_data, gemm_filter_rows, gemm_filter_cols);
  gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor>
      input_matrix(gemm_input_data, gemm_input_rows, gemm_input_cols);
  gemmlowp::MatrixMap<uint8_t, gemmlowp::MapOrder::ColMajor>
      output_matrix(output_data, gemm_output_rows, gemm_output_cols);

  const auto &output_pipeline = GemmlowpOutputPipeline::Make(
      bias_data, channels, filter->scale(), input->scale(), output->scale(),
      output->zero_point());

  using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;
  gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t, BitDepthParams>(
      gemm_context, filter_matrix, input_matrix, &output_matrix,
      -filter->zero_point(), -input->zero_point(), output_pipeline);
  durations.push_back((NowMicros() - t0) / 1000.0);

  LOG(INFO) << "duration " << VectorToString<double>(durations) << " ms";

  return MaceStatus::MACE_SUCCESS;
}

#endif  // MACE_ENABLE_QUANTIZE

// Conv2d CPU task function.
void Conv2dCpuTask::set_in_transform_future(StatsFuture *future) {
  in_transform_future_.wait_fn = future->wait_fn;
}

void Conv2dCpuTask::set_out_transform_user_event(cl::UserEvent *event) {
  out_transform_user_event_ = event;
}

MaceStatus Conv2dCpuTask::Run(OpContext *cpu_context,
                              OpContext *gpu_context) {
  int64_t t0, t1;

  t0 = NowMicros();
  in_transform_future_.wait_fn(nullptr);
  t1 = NowMicros();
  LOG(INFO) << "CPU ItoB Wait"
            << " t0 " << t0
            << " t1 " << t1
            << " t " << ((t1 - t0) / 1000.0) << " ms";

  t0 = NowMicros();
  conv2d_kernel_->Compute(cpu_context,
                          input_,
                          filter_,
                          bias_,
                          output_);
  t1 = NowMicros();
  LOG(INFO) << "CPU Compute"
            << " t0 " << t0
            << " t1 " << t1
            << " t " << ((t1 - t0) / 1000.0)  << " ms";

  gpu_context->device()->gpu_runtime()->opencl_runtime()
      ->event_manager()->SetUserEventComplete(out_transform_user_event_);
  return MaceStatus::MACE_SUCCESS;
}

#ifdef MACE_ENABLE_OPENCL
std::unique_ptr<ops::OpenCLConv2dKernel>
    CreateOpenCLConv2dKernel(const MemoryType mtype) {
  switch (mtype) {
    case MemoryType::GPU_IMAGE:
      return std::unique_ptr<ops::OpenCLConv2dKernel>(
          new ops::opencl::image::Conv2dKernel());
    case MemoryType::GPU_BUFFER:
      return std::unique_ptr<ops::OpenCLConv2dKernel>(
          new ops::opencl::buffer::Conv2dKernel());
    default:
      LOG(ERROR) << "Not support memory type";
      return nullptr;
  }
}
#endif  // MACE_ENABLE_OPENCL

}  // namespace mace
