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

#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/utils/math.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mace {

std::string PaddingTypeToString(const Padding padding_type) {
  switch (padding_type) {
    case VALID:
      return "VALID";
    case SAME:
      return "SAME";
    case FULL:
      return "FULL";
    default:
      return "NONE";
  }
}

namespace ops {

Conv2dImplType GetConv2dCpuImplType(const index_t input_channel,
                                    const index_t output_channel,
                                    const index_t kernel_size,
                                    const index_t stride) {
  if (kernel_size == 1) {
    return CONV2D_GEMM;
  } else if (kernel_size == 3 && stride == 1) {
    if (input_channel >= 8 && output_channel >=8) {
      return CONV2D_WINOGRAD;
    } else {
      return CONV2D_DIRECT;
    }
  } else {
    return CONV2D_DIRECT;
  }
}

Conv2dImplType GetConv2dGpuImplType() {
  return CONV2D_DIRECT;
}

void CalcPaddingAndOutputSize(const index_t *input_shape,
                              const DataFormat input_format,
                              const index_t *filter_shape,
                              const DataFormat filter_format,
                              const int *dilations,
                              const int *strides,
                              Padding padding,
                              index_t *output_shape,
                              int *padding_size) {
  MACE_CHECK(dilations[0] > 0 && dilations[1] > 0,
             "Invalid dilations, must >= 1");
  MACE_CHECK((dilations[0] == 1 || strides[0] == 1) &&
      (dilations[1] == 1 || strides[1] == 1),
             "If dilations > 1, strides should be 1");
  MACE_CHECK_NOTNULL(output_shape);
  MACE_CHECK_NOTNULL(padding_size);

  index_t input_height = 0, input_width = 0;
  index_t kernel_height = 0, kernel_width = 0;
  if (input_format == DataFormat::NCHW) {
    input_height = input_shape[2];
    input_width = input_shape[3];
  } else if (input_format == DataFormat::NHWC) {
    input_height = input_shape[1];
    input_width = input_shape[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  if (filter_format == DataFormat::OIHW) {
    kernel_height = filter_shape[2];
    kernel_width = filter_shape[3];
  } else if (filter_format == DataFormat::OHWI) {
    kernel_height = filter_shape[1];
    kernel_width = filter_shape[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Debug(fucheng)
  //LOG(INFO) << "Input h/w " << input_height << "/" << input_width;
  //LOG(INFO) << "Filter h/w " << kernel_height << "/" << kernel_width;

  /*
  * Convlution/pooling arithmetic:
  * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
  * For details, see https://arxiv.org/pdf/1603.07285.pdf or
  * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
  */
  padding_size[0] = 0;
  padding_size[1] = 0;
  index_t output_height = 0, output_width = 0;
  index_t output_channels = filter_shape[0];
  index_t k_extent_height = (kernel_height - 1) * dilations[0] + 1;
  index_t k_extent_width = (kernel_width - 1) * dilations[1] + 1;

  switch (padding) {
    case VALID:
      output_height = (input_height - k_extent_height) / strides[0] + 1;
      output_width = (input_width - k_extent_width) / strides[1] + 1;
      break;
    case SAME:output_height = (input_height - 1) / strides[0] + 1;
      output_width = (input_width - 1) / strides[1] + 1;
      break;
    case FULL:
      output_height = (input_height + k_extent_height - 2) / strides[0] + 1;
      output_width = (input_width + k_extent_width - 2) / strides[1] + 1;
      break;
    default:MACE_CHECK(false, "Unsupported padding type: ", padding);
  }

  // Note: TensorFlow may padded one more on the right/bottom side
  // TODO(liuqi): may be it's better to also truncate the left/top to
  // utilize the more centered features. We need to benchmark
  // based on the model accuracy.

  padding_size[0] = std::max<int>(
      0, (output_height - 1) * strides[0] + k_extent_height - input_height);
  padding_size[1] = std::max<int>(
      0, (output_width - 1) * strides[1] + k_extent_width - input_width);

  output_shape[0] = input_shape[0];
  if (input_format == DataFormat::NCHW) {
    output_shape[1] = output_channels;
    output_shape[2] = output_height;
    output_shape[3] = output_width;
  } else if (input_format == DataFormat::NHWC) {
    output_shape[1] = output_height;
    output_shape[2] = output_width;
    output_shape[3] = output_channels;
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Debug(fucheng)
  //LOG(INFO) << "Output H/W: "
  //          << "[" << output_height << "," << output_width << "]";
}

void CalcNCHWPaddingAndOutputSize(const index_t *input_shape,   // NCHW
                                  const index_t *filter_shape,  // OIHW
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size) {
  CalcPaddingAndOutputSize(input_shape, DataFormat::NCHW, filter_shape,
                           DataFormat::OIHW, dilations,
                           strides, padding, output_shape, padding_size);
}

void CalcNHWCPaddingAndOutputSize(const index_t *input_shape,   // NHWC
                                  const index_t *filter_shape,  // OIHW
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size) {
  CalcPaddingAndOutputSize(input_shape, DataFormat::NHWC, filter_shape,
                           DataFormat::OIHW, dilations,
                           strides, padding, output_shape, padding_size);
}

void CalcOutputSize(const index_t *input_shape,
                    const DataFormat input_format,
                    const index_t *filter_shape,
                    const DataFormat filter_format,
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape) {
  MACE_CHECK(dilations[0] > 0 && dilations[1] > 0,
             "Invalid dilations, must >= 1");
  MACE_CHECK((dilations[0] == 1 || strides[0] == 1) &&
      (dilations[1] == 1 || strides[1] == 1),
             "If dilations > 1, strides should be 1");
  MACE_CHECK_NOTNULL(output_shape);
  MACE_CHECK_NOTNULL(padding_size);

  index_t input_height = 0, input_width = 0;
  index_t kernel_height = 0, kernel_width = 0;
  if (input_format == DataFormat::NCHW) {
    input_height = input_shape[2];
    input_width = input_shape[3];
  } else if (input_format == DataFormat::NHWC) {
    input_height = input_shape[1];
    input_width = input_shape[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  if (filter_format == DataFormat::OIHW) {
    kernel_height = filter_shape[2];
    kernel_width = filter_shape[3];
  } else if (filter_format == DataFormat::OHWI) {
    kernel_height = filter_shape[1];
    kernel_width = filter_shape[2];
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  /*
  * Convlution/pooling arithmetic:
  * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
  * For details, see https://arxiv.org/pdf/1603.07285.pdf or
  * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
  */
  index_t output_height = 0, output_width = 0;
  index_t output_channels = filter_shape[0];

  VLOG(1) << "input_height " << input_height << ", input_width " << input_width;
  VLOG(1) << "kernel_height " << kernel_height << ", kernel_width " << kernel_width;
  VLOG(1) << "stride_height " << strides[0] << ", stride_width " << strides[1];
  VLOG(1) << "padding_height " << padding_size[0] << ", padding_width " << padding_size[1];
  VLOG(1) << "dilation_height " << dilations[0] << ", dilation_width " << dilations[1];

  if (round_type == FLOOR) {
    output_height = static_cast<index_t>(
        std::floor(1.0 * (input_height + padding_size[0] - kernel_height -
            (kernel_height - 1) * (dilations[0] - 1)) / strides[0]) + 1);
    output_width = static_cast<index_t>(
        std::floor(1.0 * (input_width + padding_size[1] - kernel_width -
            (kernel_width - 1) * (dilations[1] - 1)) / strides[1]) + 1);
  } else {
    output_height = static_cast<index_t>(
        std::ceil(1.0 * (input_height + padding_size[0] - kernel_height -
            (kernel_height - 1) * (dilations[0] - 1)) / strides[0]) + 1);
    output_width = static_cast<index_t>(
        std::ceil(1.0 * (input_width + padding_size[1] - kernel_width -
            (kernel_width - 1) * (dilations[1] - 1)) / strides[1]) + 1);
  }

  output_shape[0] = input_shape[0];
  if (input_format == DataFormat::NCHW) {
    output_shape[1] = output_channels;
    output_shape[2] = output_height;
    output_shape[3] = output_width;
  } else if (input_format == DataFormat::NHWC) {
    output_shape[1] = output_height;
    output_shape[2] = output_width;
    output_shape[3] = output_channels;
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

void CalcOutputSize(const index_t *input_shape,   // NHWC
                    const index_t *filter_shape,  // OIHW
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape) {
  CalcOutputSize(input_shape, DataFormat::NHWC, filter_shape,
                 DataFormat::OIHW, padding_size, dilations,
                 strides, round_type, output_shape);
}

void CalcNCHWOutputSize(const index_t *input_shape,   // NCHW
                        const index_t *filter_shape,  // OIHW
                        const int *padding_size,
                        const int *dilations,
                        const int *strides,
                        const RoundType round_type,
                        index_t *output_shape) {
  CalcOutputSize(input_shape, DataFormat::NCHW, filter_shape,
                 DataFormat::OIHW, padding_size, dilations,
                 strides, round_type, output_shape);
}

void CalcDeconvShape_TF(const std::vector<index_t> &input_shape,
                        const std::vector<index_t> &filter_shape,
                        const std::vector<index_t> &output_shape,
                        const std::vector<int> &strides,
                        Padding padding_type,
                        const int group,
                        std::vector<int> *in_pad_size,
                        std::vector<int> *out_pad_size,
                        std::vector<index_t> *padded_out_shape,
                        DataFormat data_format) {
  const index_t
      in_height =
      data_format == DataFormat::NCHW ? input_shape[2] : input_shape[1];
  const index_t
      in_width =
          data_format == DataFormat::NCHW ? input_shape[3] : input_shape[2];

  const index_t
      out_height =
          data_format == DataFormat::NCHW ? output_shape[2] : output_shape[1];
  const index_t
      out_width =
          data_format == DataFormat::NCHW ? output_shape[3] : output_shape[2];

  const index_t extended_in_height = (in_height - 1) * strides[0] + 1;
  const index_t extended_in_width = (in_width - 1) * strides[1] + 1;

#if 0
  LOG(INFO) << "extended_in_height " << extended_in_height
            << ", extended_in_width " << extended_in_width;
#endif

  const index_t kernel_h = filter_shape[2];
  const index_t kernel_w = filter_shape[3];

  index_t expected_input_height = 0, expected_input_width = 0;

  switch (padding_type) {
    case VALID:
      expected_input_height =
          (out_height - kernel_h + strides[0]) / strides[0];
      expected_input_width =
          (out_width - kernel_w + strides[1]) / strides[1];
      break;
    case SAME:
      expected_input_height =
          (out_height + strides[0] - 1) / strides[0];
      expected_input_width =
          (out_width + strides[1] - 1) / strides[1];
      break;
    default:MACE_CHECK(false, "Unsupported padding type: ", padding_type);
  }

#if 0
  LOG(INFO) << "expected_input_height " << expected_input_height
            << ", expected_input_width " << expected_input_width;
#endif

  MACE_CHECK(expected_input_height == in_height,
             expected_input_height, "!=", in_height);
  MACE_CHECK(expected_input_width == in_width,
             expected_input_width, "!=", in_width);

  const index_t padded_out_height =
      (in_height - 1) * strides[0] + kernel_h;
  const index_t padded_out_width =
      (in_width - 1) * strides[1] + kernel_w;

#if 0
  LOG(INFO) << "padded_out_height " << padded_out_height
            << ", padded_out_width " << padded_out_width;
#endif

  if (in_pad_size != nullptr) {
    const int p_h =
        static_cast<int>(out_height + kernel_h - 1 - extended_in_height);
    const int p_w =
        static_cast<int>(out_width + kernel_w - 1 - extended_in_width);
    in_pad_size->resize(2);
    (*in_pad_size)[0] = std::max<int>(0, p_h);
    (*in_pad_size)[1] = std::max<int>(0, p_w);
  }

  if (out_pad_size != nullptr) {
    const int o_p_h = static_cast<int>(padded_out_height - out_height);
    const int o_p_w = static_cast<int>(padded_out_width - out_width);
    out_pad_size->resize(2);
    (*out_pad_size)[0] = std::max<int>(0, o_p_h);
    (*out_pad_size)[1] = std::max<int>(0, o_p_w);
  }

  if (padded_out_shape != nullptr) {
    index_t output_channel = filter_shape[0] * group;
    padded_out_shape->resize(4);
    (*padded_out_shape)[0] = output_shape[0];
    (*padded_out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : padded_out_height;
    (*padded_out_shape)[2] =
        data_format == DataFormat::NCHW ? padded_out_height : padded_out_width;
    (*padded_out_shape)[3] =
        data_format == DataFormat::NCHW ? padded_out_width : output_channel;
  }
}

void CalcDeconvShape_Onnx(const std::vector<index_t> &input_shape,
                          const std::vector<index_t> &filter_shape,
                          const std::vector<int> &strides,
                          const std::vector<int> &out_pad_size,
                          const std::vector<int> &output_paddings,
                          const int group,
                          std::vector<index_t> *out_shape,
                          std::vector<int> *in_pad_size,
                          std::vector<index_t> *padded_out_shape,
                          DataFormat data_format) {
  const index_t
      in_height =
          data_format == DataFormat::NCHW ? input_shape[2] : input_shape[1];
  const index_t
      in_width =
          data_format == DataFormat::NCHW ? input_shape[3] : input_shape[2];

  const index_t output_channel = filter_shape[0] * group;

  const index_t kernel_h = filter_shape[2];
  const index_t kernel_w = filter_shape[3];

  index_t padded_out_height =
      (in_height - 1) * strides[0] + kernel_h;
  index_t padded_out_width =
      (in_width - 1) * strides[1] + kernel_w;

  if (in_pad_size != nullptr) {
    in_pad_size->resize(2);
    (*in_pad_size)[0] = static_cast<int>((kernel_h - 1) * 2 - out_pad_size[0]);
    (*in_pad_size)[1] = static_cast<int>((kernel_w - 1) * 2 - out_pad_size[1]);
    (*in_pad_size)[0] = std::max<int>(0, (*in_pad_size)[0]);
    (*in_pad_size)[1] = std::max<int>(0, (*in_pad_size)[1]);
  }

  if (padded_out_shape != nullptr) {
    padded_out_shape->resize(4);
    (*padded_out_shape)[0] = input_shape[0];
    (*padded_out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : padded_out_height;
    (*padded_out_shape)[2] =
        data_format == DataFormat::NCHW ? padded_out_height : padded_out_width;
    (*padded_out_shape)[3] =
        data_format == DataFormat::NCHW ? padded_out_width : output_channel;
  }

  if (out_shape != nullptr) {
    index_t out_height = padded_out_height - out_pad_size[0] + output_paddings[0];
    index_t out_width = padded_out_width - out_pad_size[1] + output_paddings[1];
    out_shape->resize(4);
    (*out_shape)[0] = input_shape[0];
    (*out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : out_height;
    (*out_shape)[2] = data_format == DataFormat::NCHW ? out_height : out_width;
    (*out_shape)[3] =
        data_format == DataFormat::NCHW ? out_width : output_channel;
  }
}

void CalcDeconvShape_Caffe(const std::vector<index_t> &input_shape,
                           const std::vector<index_t> &filter_shape,
                           const std::vector<int> &strides,
                           const std::vector<int> &out_pad_size,
                           const int group,
                           std::vector<index_t> *out_shape,
                           std::vector<int> *in_pad_size,
                           std::vector<index_t> *padded_out_shape,
                           DataFormat data_format) {
  const index_t
      in_height =
          data_format == DataFormat::NCHW ? input_shape[2] : input_shape[1];
  const index_t
      in_width =
          data_format == DataFormat::NCHW ? input_shape[3] : input_shape[2];

  const index_t output_channel = filter_shape[0] * group;

  const index_t kernel_h = filter_shape[2];
  const index_t kernel_w = filter_shape[3];

  index_t padded_out_height =
      (in_height - 1) * strides[0] + kernel_h;
  index_t padded_out_width =
      (in_width - 1) * strides[1] + kernel_w;

  if (in_pad_size != nullptr) {
    in_pad_size->resize(2);
    (*in_pad_size)[0] = static_cast<int>((kernel_h - 1) * 2 - out_pad_size[0]);
    (*in_pad_size)[1] = static_cast<int>((kernel_w - 1) * 2 - out_pad_size[1]);
    (*in_pad_size)[0] = std::max<int>(0, (*in_pad_size)[0]);
    (*in_pad_size)[1] = std::max<int>(0, (*in_pad_size)[1]);
  }

  if (padded_out_shape != nullptr) {
    padded_out_shape->resize(4);
    (*padded_out_shape)[0] = input_shape[0];
    (*padded_out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : padded_out_height;
    (*padded_out_shape)[2] =
        data_format == DataFormat::NCHW ? padded_out_height : padded_out_width;
    (*padded_out_shape)[3] =
        data_format == DataFormat::NCHW ? padded_out_width : output_channel;
  }

  if (out_shape != nullptr) {
    index_t out_height = padded_out_height - out_pad_size[0];
    index_t out_width = padded_out_width - out_pad_size[1];
    out_shape->resize(4);
    (*out_shape)[0] = input_shape[0];
    (*out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : out_height;
    (*out_shape)[2] = data_format == DataFormat::NCHW ? out_height : out_width;
    (*out_shape)[3] =
        data_format == DataFormat::NCHW ? out_width : output_channel;
  }
}

void CalDeconvOutputShapeAndPadSize(const std::vector<index_t> &input_shape,
                                    const std::vector<index_t> &filter_shape,
                                    const std::vector<int> &strides,
                                    Padding padding_type,
                                    const std::vector<int> &paddings,
                                    const std::vector<int> &output_paddings,
                                    int group,
                                    std::vector<index_t> *output_shape,
                                    std::vector<int> *in_pad_size,
                                    std::vector<int> *out_pad_size,
                                    std::vector<index_t> *padded_out_shape,
                                    FrameworkType framework_type,
                                    DataFormat data_format) {
  VLOG(1) << "framework_type " << static_cast<int>(framework_type);
  if (framework_type == FrameworkType::TENSORFLOW) {
    MACE_CHECK(output_shape->size() == 4,
               "deconv output shape shoud be 4-dims");
    // QUESTION(fucheng): Why do this?
    std::vector<index_t> &out_shape = *output_shape;
    if (data_format == DataFormat::NCHW) {
     const index_t t = out_shape[1];
     out_shape[1] = out_shape[3];
     out_shape[3] = out_shape[2];
     out_shape[2] = t;
    }

    CalcDeconvShape_TF(
        input_shape,
        filter_shape,
        *output_shape,
        strides,
        padding_type,
        group,
        in_pad_size,
        out_pad_size,
        padded_out_shape,
        data_format);
  } else {  // caffe
    if (!paddings.empty()) {
      *out_pad_size = paddings;
    }
    if (framework_type == FrameworkType::ONNX) {
      CalcDeconvShape_Onnx(
          input_shape,
          filter_shape,
          strides,
          *out_pad_size,
          output_paddings,
          group,
          output_shape,
          in_pad_size,
          padded_out_shape,
          data_format);
    } else {
      CalcDeconvShape_Caffe(
          input_shape,
          filter_shape,
          strides,
          *out_pad_size,
          group,
          output_shape,
          in_pad_size,
          padded_out_shape,
          data_format);
    }
  }
}

void CalcWinogradOutTileSizeDefault(const index_t in_height,
                                    const index_t in_width,
                                    index_t *out_tile_size) {
  *out_tile_size = 2;
  // When size of input feature map is bigger than 16x16,
  // set winograd out tile size to 6 to get higher performance.
  if (in_height > 16 && in_width > 16) {
    *out_tile_size = 6;
  }
}

void CalcWinogradOutTileSizeNew(const index_t in_height,
                                const index_t in_width,
                                index_t *out_tile_size) {
  // NOTE(fucheng): I support out tile size 4.
  *out_tile_size = 2;
  if (in_height > 64 && in_width > 64) {
    *out_tile_size = 6;
  } else if (in_height > 16 && in_width > 16) {
    *out_tile_size = 4;
  }
}

void CalcWinogradOutTileSize(const index_t in_height,
                             const index_t in_width,
                             index_t *out_tile_size) {
  CalcWinogradOutTileSizeDefault(in_height, in_width, out_tile_size);
}

void CalcK3x3S1WinogradOutTileCount(const index_t in_height,
                                    const index_t in_width,
                                    const index_t out_height,
                                    const index_t out_width,
                                    int *pad_in_height,
                                    int *pad_in_width,
                                    int *pad_out_height,
                                    int *pad_out_width,
                                    int *total_out_tile_count,
                                    int *out_tile_size) {
  index_t *out_tile_size_ptr = reinterpret_cast<index_t*>(out_tile_size);
  CalcWinogradOutTileSize(in_height, in_width, out_tile_size_ptr);
  if (pad_out_height != nullptr) {
    *pad_out_height = RoundUpDiv(out_height, *out_tile_size_ptr) * *out_tile_size_ptr;
  }
  if (pad_out_width != nullptr) {
    *pad_out_width = RoundUpDiv(out_width, *out_tile_size_ptr) * *out_tile_size_ptr;
  }
  if (pad_in_height != nullptr) {
    *pad_in_height = *pad_out_height + 2;
  }
  if (pad_in_width != nullptr) {
    *pad_in_width = *pad_out_width + 2;
  }
  if (total_out_tile_count != nullptr) {
    *total_out_tile_count = (*pad_out_height * *pad_out_width) / (*out_tile_size_ptr * *out_tile_size_ptr);
  }
}

void GetOpenCLWidthBlockSize(const index_t k,
                             index_t *block_size) {
  MACE_CHECK(block_size != nullptr);
  if (k == 1) {
    *block_size = 4;
  } else if (k == 3) {
    *block_size = 5;
  } else {
    *block_size = 4;
  }
}

void CalcOpenCLBlockCount(const index_t ow,
                          const index_t ic,
                          const index_t oc,
                          const index_t k,
                          int *owb,
                          int *icb,
                          int *ocb) {
  const index_t c_block_size = 4;
  index_t ow_block_size;
  GetOpenCLWidthBlockSize(k, &ow_block_size);

  if (owb != nullptr) {
    *owb = RoundUpDiv(ow, ow_block_size);
  }
  if (icb != nullptr) {
    *icb = RoundUpDiv(ic, c_block_size);
  }
  if (ocb != nullptr) {
    *ocb = RoundUpDiv(oc, c_block_size);
  }
}

void CalcOpenCLDefaultLWS(OpenCLRuntime *runtime,
                          const uint32_t *gws,
                          const uint32_t kwg_size,
                          const uint32_t owb,
                          std::vector<uint32_t> &lws) {
  if (lws.empty()) {
    lws = std::vector<uint32_t>(4, 0);
  }

  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    const uint32_t base_gpu_mem_cache_size = 16384;
    const uint32_t kernel_cache_size = (owb + 4 + owb) * 4 * 4;
    const uint64_t cache_size = runtime->device_global_mem_cache_size();
    const uint32_t device_compute_units = runtime->device_compute_units();
    const uint32_t compute_units = std::max<uint32_t>(device_compute_units / 2, 1);
    const uint32_t base =
        std::max<uint32_t>(
            std::min<uint32_t>(cache_size / base_gpu_mem_cache_size, 4), 1);
    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    lws[0] =
        std::min<uint32_t>(std::min<uint32_t>(gws[0], base), kwg_size / lws[1]);
    const uint32_t lws_size = lws[0] * lws[1];
    const uint32_t lws_limit = RoundUp<uint32_t>(
        cache_size / kernel_cache_size / lws_size / compute_units, base);
    lws[2] = std::min<uint32_t>(lws_limit, gws[2]);
    if (lws[2] == 0) {
      lws[2] = std::min<uint32_t>(gws[2], base);
    }
    lws[2] = std::max<uint32_t>(std::min<uint32_t>(lws[2], kwg_size / lws_size),
                                1);
  }
}

void CalcOpenCLConv2dK1x1LWS(OpenCLRuntime *runtime,
                             const uint32_t *gws,
                             const uint32_t kwg_size,
                             const uint32_t owb,
                             std::vector<uint32_t> &lws) {
  const uint32_t kernel_cache_size = (owb + 4 + owb) * 4 * 4;
  const uint32_t lws_limit = 128;

  lws = std::vector<uint32_t>(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint32_t base_gpu_mem_cache_size = 16384;
    uint64_t cache_size = runtime->device_global_mem_cache_size();
    uint32_t compute_units = runtime->device_compute_units();
    const uint32_t base =
        std::max<uint32_t>(cache_size / base_gpu_mem_cache_size, 1);

    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    if (lws[1] >= base) {
      lws[0] = std::min<uint32_t>(gws[0], base);
    } else if ((1 < lws[1] && lws[1] < base) && gws[0] >= lws_limit) {
      lws[0] = std::min<uint32_t>(gws[0], base);
    } else {
      lws[0] = gws[0] / 8;
      if (lws[0] < base) {
        lws[0] = std::max<uint32_t>(gws[0] / 4, base);
      }
    }
    lws[0] = std::min<uint32_t>(lws[0], kwg_size / lws[1]);
    const uint32_t lws_size = lws[0] * lws[1];
    lws[2] = std::min<uint32_t>(
        (cache_size / kernel_cache_size / lws_size / compute_units) * 8,
        gws[2]);
    if (lws[2] == 0) {
      lws[2] = std::min<uint32_t>(gws[2], base);
    }
    lws[2] = std::max<uint32_t>(std::min<uint32_t>(lws[2], kwg_size / lws_size),
                                1);
  }
}

void CalcOpenCLConv2dK3x3LWS(OpenCLRuntime *runtime,
                             const uint32_t *gws,
                             const uint32_t kwg_size,
                             const uint32_t owb,
                             std::vector<uint32_t> &lws) {
  const uint32_t kernel_cache_size = (owb + 4 + owb) * 4 * 4;

  lws = std::vector<uint32_t>(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint32_t base_gpu_mem_cache_size = 16384;
    uint64_t cache_size = runtime->device_global_mem_cache_size();
    uint32_t device_compute_units = runtime->device_compute_units();
    uint32_t compute_units = std::max<uint32_t>(device_compute_units / 2, 1);
    const uint32_t base =
        std::max<uint32_t>(
            std::min<uint32_t>(cache_size / base_gpu_mem_cache_size, 4), 1);
    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    lws[0] =
        std::min<uint32_t>(std::min<uint32_t>(gws[0], base), kwg_size / lws[1]);
    const uint32_t lws_size = lws[0] * lws[1];
#if 0
    LOG(INFO) << "cache_size " << cache_size << ", kernel_cache_size " << kernel_cache_size
              << ", lws_size " << lws_size << ", compute_units " << compute_units
              << ", base " << base;
#endif
    const uint32_t lws_limit = RoundUp<uint32_t>(
        cache_size / kernel_cache_size / lws_size / compute_units, base);

#if 0
    LOG(INFO) << "lws_limit " << lws_limit << ", gws[2] " << gws[2]
              << ", base " << base << ", kwg_size " << kwg_size
              << ", lws_size " << lws_size;
#endif

    lws[2] = std::min<uint32_t>(lws_limit, gws[2]);
    if (lws[2] == 0) {
      lws[2] = std::min<uint32_t>(gws[2], base);
    }
    lws[2] = std::max<uint32_t>(std::min<uint32_t>(lws[2], kwg_size / lws_size),
                                1);
  }
}

void CalcOpenCLConv2dGeneralLWS(OpenCLRuntime *runtime,
                                const uint32_t *gws,
                                const uint32_t kwg_size,
                                const uint32_t owb,
                                const uint32_t kernel_size,
                                std::vector<uint32_t> &lws) {
  const uint32_t kernel_cache_size = (owb + 4 + owb) * 4 * 4;
  const uint32_t lws_limit = 20;

  lws = std::vector<uint32_t>(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint32_t base_gpu_mem_cache_size = 16384;
    uint64_t cache_size = runtime->device_global_mem_cache_size();
    uint32_t compute_units = runtime->device_compute_units();
    const uint32_t base =
        std::max<uint32_t>(cache_size / base_gpu_mem_cache_size, 1);

    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    lws[0] = gws[0] / 4;
    if (lws[0] == 0) {
      lws[0] = gws[0];
    }
    lws[0] = std::min<uint32_t>(lws[0], kwg_size / lws[1]);
    const uint32_t lws_size = lws[0] * lws[1];
    lws[2] = std::min<uint32_t>((cache_size / kernel_cache_size / kernel_size /
                                    lws_size / compute_units) *
                                    8,
                                gws[2]);
    if (lws[2] == 0) {
      if (gws[2] < lws_limit) {
        lws[2] = gws[2];
      } else {
        lws[2] = base;
      }
    }
    lws[2] = std::max<uint32_t>(std::min<uint32_t>(lws[2], kwg_size / lws_size),
                                1);
  }
}

}  // namespace ops
}  // namespace mace
