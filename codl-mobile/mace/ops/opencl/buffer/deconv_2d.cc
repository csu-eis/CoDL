
#include "mace/ops/opencl/buffer/deconv_2d.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

MaceStatus Deconv2dKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    const int *strides,
    const int *padding_data,
    const ActivationType activation,
    const float relux_max_limit,
    const float leakyrelu_coefficient,
    const std::vector<index_t> &output_shape,
    Tensor *output) {
  MACE_UNUSED(output_shape);

  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const int stride_h = strides[0];
  const int stride_w = strides[1];
  MACE_CHECK(stride_w > 0 && stride_h > 0, "strides should be > 0.");
  const int width_tile = 5;
  const index_t n_strides = (width + stride_w - 1) / stride_w;
  const index_t width_blocks =
      ((n_strides + width_tile - 1) / width_tile) * stride_w;
  const float stride_h_r = 1.f / static_cast<float>(stride_h);
  const float stride_w_r = 1.f / static_cast<float>(stride_w);
  const int padding_h = (padding_data[0] + 1) >> 1;
  const int padding_w = (padding_data[1] + 1) >> 1;

  const int align_h = stride_h - 1 - padding_h;
  const int align_w = stride_w - 1 - padding_w;
  const int kernel_size = filter->dim(2) * filter->dim(3);

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("deconv_2d");
    built_options.emplace("-Ddeconv_2d=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    switch (activation) {
      case NOOP:
        break;
      case RELU:
        built_options.emplace("-DUSE_RELU");
        break;
      case RELUX:
        built_options.emplace("-DUSE_RELUX");
        break;
      case TANH:
        built_options.emplace("-DUSE_TANH");
        break;
      case SIGMOID:
        built_options.emplace("-DUSE_SIGMOID");
        break;
      case LEAKYRELU:
        built_options.emplace("-DUSE_LEAKYRELU");
        break;
      default:
        LOG(FATAL) << "Unknown activation type: " << activation;
    }

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("deconv_2d_buffer", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_BUFF_OUT_OF_RANGE_SET_ARGS(kernel_, output->size());
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    kernel_.setArg(idx++, *(filter->opencl_buffer()));
    if (bias != nullptr) {
      kernel_.setArg(idx++, *(bias->opencl_buffer()));
    }
    kernel_.setArg(idx++, *(output->opencl_buffer()));
    kernel_.setArg(idx++, relux_max_limit);
    kernel_.setArg(idx++, leakyrelu_coefficient);
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(3)));
    kernel_.setArg(idx++, static_cast<int32_t>(height));
    kernel_.setArg(idx++, static_cast<int32_t>(width));
    kernel_.setArg(idx++, static_cast<int32_t>(channels));
    kernel_.setArg(idx++, static_cast<int32_t>(stride_h));
    kernel_.setArg(idx++, static_cast<int32_t>(stride_w));
    kernel_.setArg(idx++, stride_h_r);
    kernel_.setArg(idx++, stride_w_r);
    kernel_.setArg(idx++, static_cast<int32_t>(align_h));
    kernel_.setArg(idx++, static_cast<int32_t>(align_w));
    kernel_.setArg(idx++, static_cast<int32_t>(padding_h));
    kernel_.setArg(idx++, static_cast<int32_t>(padding_w));
    kernel_.setArg(idx++, static_cast<int32_t>(filter->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(filter->dim(3)));
    kernel_.setArg(idx++, static_cast<int32_t>(kernel_size));
    kernel_.setArg(idx++, static_cast<int32_t>(input_channel_blocks));
    kernel_.setArg(idx++, static_cast<int32_t>(channel_blocks));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("deconv2d_opencl_kernel_", activation, output->dim(0),
             output->dim(1), output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Deconv2dKernel::ResizeOutputTensor(
    const std::vector<index_t> &output_shape,
    Tensor *output) {
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace
