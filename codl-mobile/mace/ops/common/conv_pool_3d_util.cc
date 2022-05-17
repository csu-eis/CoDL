
#include "mace/ops/common/conv_pool_3d_util.h"

namespace mace {
namespace ops {

void Calc3dOutputSize(const index_t *input_shape,
                      const DataFormat input_format,
                      const index_t *filter_shape,
                      const DataFormat filter_format,
                      const int *padding_size,
                      const int *dilations,
                      const int *strides,
                      const RoundType round_type,
                      index_t *output_shape) {
  MACE_CHECK(dilations[0] > 0 && dilations[1] > 0 && dilations[2] > 0,
             "Invalid dilations, must >= 1");
  MACE_CHECK((dilations[0] == 1 || strides[0] == 1) &&
             (dilations[1] == 1 || strides[1] == 1) &&
             (dilations[2] == 1 || strides[2] == 1),
             "If dilations > 1, strides should be 1");
  MACE_CHECK_NOTNULL(output_shape);
  MACE_CHECK_NOTNULL(padding_size);

  index_t input_height = 0, input_width = 0, input_depth = 0;
  index_t kernel_height = 0, kernel_width = 0, kernel_depth = 0;
  if (input_format == DataFormat::NCHW) {
    input_height = input_shape[2];
    input_width = input_shape[3];
    input_depth = input_shape[4];
  } else if (input_format == DataFormat::NHWC) {
    input_height = input_shape[1];
    input_width = input_shape[2];
    input_depth = input_shape[3];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  if (filter_format == DataFormat::OIHW) {
    kernel_height = filter_shape[2];
    kernel_width = filter_shape[3];
    kernel_depth = filter_shape[4];
  } else if (filter_format == DataFormat::OHWI) {
    kernel_height = filter_shape[1];
    kernel_width = filter_shape[2];
    kernel_depth = filter_shape[3];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  VLOG(1) << "input_depth " << input_depth
          << ", kernel_depth " << kernel_depth;
  /*
  * Convlution/pooling arithmetic:
  * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
  * For details, see https://arxiv.org/pdf/1603.07285.pdf or
  * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
  */
  index_t output_height = 0, output_width = 0, output_depth = 0;
  index_t output_channels = filter_shape[0];

  if (round_type == FLOOR) {
    output_height = static_cast<index_t>(
        std::floor(1.0 * (input_height + padding_size[0] - kernel_height -
            (kernel_height - 1) * (dilations[0] - 1)) / strides[0]) + 1);
    output_width = static_cast<index_t>(
        std::floor(1.0 * (input_width + padding_size[1] - kernel_width -
            (kernel_width - 1) * (dilations[1] - 1)) / strides[1]) + 1);
    output_depth = static_cast<index_t>(
        std::floor(1.0 * (input_depth + padding_size[2] - kernel_depth -
            (kernel_depth - 1) * (dilations[2] - 1)) / strides[2]) + 1);
  } else {
    output_height = static_cast<index_t>(
        std::ceil(1.0 * (input_height + padding_size[0] - kernel_height -
            (kernel_height - 1) * (dilations[0] - 1)) / strides[0]) + 1);
    output_width = static_cast<index_t>(
        std::ceil(1.0 * (input_width + padding_size[1] - kernel_width -
            (kernel_width - 1) * (dilations[1] - 1)) / strides[1]) + 1);
    output_depth = static_cast<index_t>(
        std::ceil(1.0 * (input_depth + padding_size[2] - kernel_depth -
            (kernel_depth - 1) * (dilations[2] - 1)) / strides[2]) + 1);
  }

  VLOG(1) << "padding_size[2] " << padding_size[2]
          << ", dilations[2] " << dilations[2]
          << ", strides[2] " << strides[2];

  VLOG(1) << "output_height " << output_height
          << ", output_width " << output_width
          << ", output_depth " << output_depth;

  output_shape[0] = input_shape[0];
  if (input_format == DataFormat::NCHW) {
    output_shape[1] = output_channels;
    output_shape[2] = output_height;
    output_shape[3] = output_width;
    output_shape[4] = output_depth;
  } else if (input_format == DataFormat::NHWC) {
    output_shape[1] = output_height;
    output_shape[2] = output_width;
    output_shape[3] = output_depth;
    output_shape[4] = output_channels;
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

void Calc3dOutputSize(const index_t *input_shape,   // NHWC
                      const index_t *filter_shape,  // OIHW
                      const int *padding_size,
                      const int *dilations,
                      const int *strides,
                      const RoundType round_type,
                      index_t *output_shape) {
  Calc3dOutputSize(input_shape, DataFormat::NHWC, filter_shape,
                   DataFormat::OIHW, padding_size, dilations,
                   strides, round_type, output_shape);
}

void Calc3dNCHWOutputSize(const index_t *input_shape,   // NCHW
                          const index_t *filter_shape,  // OIHW
                          const int *padding_size,
                          const int *dilations,
                          const int *strides,
                          const RoundType round_type,
                          index_t *output_shape) {
  Calc3dOutputSize(input_shape, DataFormat::NCHW, filter_shape,
                   DataFormat::OIHW, padding_size, dilations,
                   strides, round_type, output_shape);
}

}  // namespace ops
}  // namespace mace
