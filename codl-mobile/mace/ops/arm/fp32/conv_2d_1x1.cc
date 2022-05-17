// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/arm/fp32/conv_2d_1x1.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

MaceStatus Conv2dK1x1::Compute(const OpContext *context,
                               const Tensor *input,
                               const Tensor *filter,
                               Tensor *output) {
  //LOG(INFO) << "Conv2dK1x1::Compute";
  //LOG(INFO) << "P1";
  index_t batch = input->dim(0);
  index_t in_height = input->dim(2);
  index_t in_width = input->dim(3);
  index_t in_channels = input->dim(1);

  std::vector<index_t> output_shape;
  std::vector<int> in_pad_size;
  std::vector<int> out_pad_size;
  CalOutputShapeAndPadSize(input,
                           filter,
                           1,
                           1,
                           &output_shape,
                           &in_pad_size,
                           &out_pad_size);
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));
  //LOG(INFO) << "Conv2dK1x1"
  //          << " input shape " << VectorToString<index_t>(input->shape())
  //          << " filter shape " << VectorToString<index_t>(filter->shape())
  //          << " output shape " << VectorToString<index_t>(output_shape);

  //LOG(INFO) << "P2";

  const index_t out_channels = output_shape[1];
  const index_t out_height = output_shape[2];
  const index_t out_width = output_shape[3];
  const index_t padded_in_height = in_height + in_pad_size[0] + in_pad_size[1];
  const index_t padded_in_width = in_width + in_pad_size[2] + in_pad_size[3];

  // pad input and transform input
  const bool is_in_padded =
      in_height != padded_in_height || in_width != padded_in_width;
  //auto scratch_buffer = context->device()->scratch_buffer();
  auto scratch_buffer = context->cpu_device()->scratch_buffer();
  const index_t padded_in_size = is_in_padded ? PadAlignSize(
      sizeof(float) * batch * in_channels * padded_in_height
          * padded_in_width) : 0;
  const index_t pack_filter_size =
      PadAlignSize(sizeof(float) * out_channels * in_channels);
  const index_t pack_input_size =
      PadAlignSize(
          sizeof(float) * in_channels * padded_in_height * padded_in_width);
  const index_t pack_output_size =
      PadAlignSize(
          sizeof(float) * out_channels * padded_in_height * padded_in_width);

  const index_t gemm_pack_size =
      pack_filter_size + pack_input_size + pack_output_size;

  //LOG(INFO) << "P3";

  scratch_buffer->Rewind();
  scratch_buffer->GrowSize(padded_in_size + gemm_pack_size);

  const Tensor *padded_in = input;
  Tensor tmp_padded_in
      (scratch_buffer->Scratch(padded_in_size), DataType::DT_FLOAT);
  if (is_in_padded) {
    tmp_padded_in.Resize({batch, in_channels, padded_in_height,
                          padded_in_width});
    PadInput(*input, in_pad_size[0], in_pad_size[2], &tmp_padded_in);
    padded_in = &tmp_padded_in;
  }

  //LOG(INFO) << "P4";

  return gemm_.Compute(context,
                       filter,
                       padded_in,
                       batch,
                       out_channels,
                       in_channels,
                       in_channels,
                       out_height * out_width,
                       false,
                       false,
                       false,
                       false,
                       true,
                       output);
}

MaceStatus Conv2dK1x1S2::Compute(const OpContext *context,
                                 const Tensor *input,
                                 const Tensor *filter,
                                 Tensor *output) {
  index_t batch = input->dim(0);
  index_t in_height = input->dim(2);
  index_t in_width = input->dim(3);
  index_t in_channels = input->dim(1);

  std::vector<index_t> output_shape;
  std::vector<int> in_pad_size;
  std::vector<int> out_pad_size;
  CalOutputShapeAndPadSize(input,
                           filter,
                           1,
                           1,
                           &output_shape,
                           &in_pad_size,
                           &out_pad_size);
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));

#ifdef MACE_ENABLE_DEBUG_INFO
  LOG(INFO) << "Conv2dK1x1S2:"
            << " input shape " << VectorToString<index_t>(input->shape())
            << " filter shape " << VectorToString<index_t>(filter->shape())
            << " output shape " << VectorToString<index_t>(output->shape());
#endif

  const index_t out_channels = output_shape[1];
  const index_t out_height = output_shape[2];
  const index_t out_width = output_shape[3];
  const index_t padded_in_height = in_height + in_pad_size[0] + in_pad_size[1];
  const index_t padded_in_width = in_width + in_pad_size[2] + in_pad_size[3];

  // pad input and transform input
  const bool is_in_padded =
      in_height != padded_in_height || in_width != padded_in_width;
  //auto scratch_buffer = context->device()->scratch_buffer();
  auto scratch_buffer = context->cpu_device()->scratch_buffer();
  const index_t padded_in_size = is_in_padded ? PadAlignSize(
      sizeof(float) * batch * in_channels * padded_in_height
          * padded_in_width) : 0;
  const index_t pack_filter_size =
      PadAlignSize(sizeof(float) * out_channels * in_channels);
  const index_t pack_input_size =
      PadAlignSize(
          sizeof(float) * in_channels * padded_in_height * padded_in_width);
  const index_t pack_output_size =
      PadAlignSize(
          sizeof(float) * out_channels * padded_in_height * padded_in_width);

  const index_t gemm_pack_size =
      pack_filter_size + pack_input_size + pack_output_size;

  scratch_buffer->Rewind();
  scratch_buffer->GrowSize(padded_in_size + gemm_pack_size);

#ifdef MACE_ENABLE_DEBUG_INFO
  LOG(INFO) << "Padde in size " << padded_in_size;
  LOG(INFO) << "Pack filter size " << pack_filter_size;
  LOG(INFO) << "Pack input size " << pack_input_size;
  LOG(INFO) << "Pack output size " << pack_output_size;
  LOG(INFO) << "Gemm pack size " << gemm_pack_size;
#endif

  // NOTE(fucheng): padded_in_size is 0, which causes resize failure for tmp_padded_in.
  // therefore I change padded_in_size to pack_output_size.
  const Tensor *padded_in = input;
  Tensor tmp_padded_in
      (scratch_buffer->Scratch(pack_output_size), DataType::DT_FLOAT);
  // Assert that we do not need padding for K1x1
  MACE_CHECK(!is_in_padded);
  // Padded will be remove in the future version.
  /**
  if (is_in_padded) {
    tmp_padded_in.Resize({batch, in_channels, padded_in_height,
                          padded_in_width});
    PadInput(*input, in_pad_size[0], in_pad_size[2], &tmp_padded_in);
    padded_in = &tmp_padded_in;
  }**/
  
  // Rearrange input data to scratch buffer
  tmp_padded_in.Resize({batch, in_channels, out_height, out_width});
  RearrangeInputBlock4(*input, &tmp_padded_in);
  padded_in = &tmp_padded_in;

#ifdef MACE_ENABLE_DEBUG_INFO
  LOG(INFO) << "Gemm Parameters:";
  LOG(INFO) << "lhs_rows: " << out_channels;
  LOG(INFO) << "lhs_cols: " << in_channels;
  LOG(INFO) << "rhs_rows: " << in_channels;
  LOG(INFO) << "rhs_cols: " << out_height * out_width;
#endif
  
  return gemm_.Compute(context,
                       filter,
                       padded_in,
                       batch,
                       out_channels,
                       in_channels,
                       in_channels,
                       out_height * out_width,
                       false,
                       false,
                       false,
                       false,
                       true,
                       output);
}

void Conv2dK1x1S2::RearrangeInput(const Tensor &src,
                                  Tensor *dst) {
  if (dst == &src) return;
  const index_t batch = src.dim(0);
  const index_t channels = src.dim(1);
  const index_t height = src.dim(2);
  const index_t width = src.dim(3);
  const index_t rearranged_height = dst->dim(2);
  const index_t rearranged_width = dst->dim(3);
  auto in_data = src.data<float>();
  auto rearranged_in_data = dst->mutable_data<float>();
  
  const index_t img_size = height * width;
  const index_t rearranged_img_size = rearranged_height * rearranged_width;
  
  const index_t strides_h = strides_[0];
  const index_t strides_w = strides_[1];
  
  for (index_t b = 0; b < batch; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const index_t bc = b * channels + c;
      const float *in_base = in_data + bc * img_size;
      float *rearranged_in_base = rearranged_in_data + bc * rearranged_img_size;

      for (index_t h = 0; h < height; h += strides_h) {
        //LOG(INFO) << "dh " << h / strides_h
        //          << " sh " << h;
        for (index_t w = 0; w < width; w += strides_w) {
          //LOG(INFO) << "dw " << w / strides_w
          //          << " sw " << w;
          rearranged_in_base[w / strides_w] = in_base[w];
        }
        
        in_base += (width * strides_h);
        rearranged_in_base += rearranged_width;
      }
    }
  }
}

void Conv2dK1x1S2::RearrangeInputBlock4(const Tensor &src,
                                        Tensor *dst) {
  if (dst == &src) return;
  const index_t batch = src.dim(0);
  const index_t channels = src.dim(1);
  const index_t height = src.dim(2);
  const index_t width = src.dim(3);
  const index_t rearranged_height = dst->dim(2);
  const index_t rearranged_width = dst->dim(3);
  auto in_data = src.data<float>();
  auto rearranged_in_data = dst->mutable_data<float>();
  
  const index_t img_size = height * width;
  const index_t rearranged_img_size = rearranged_height * rearranged_width;
  
  const index_t strides_h = strides_[0];
  //const index_t strides_w = strides_[1];

  for (index_t b = 0; b < batch; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const index_t bc = b * channels + c;
      const float *in_base = in_data + bc * img_size;
      float *rearranged_in_base = rearranged_in_data + bc * rearranged_img_size;
      float tmp_data[4];

      for (index_t h = 0; h < height; h += strides_h) {
        index_t src_bw = 0;
        index_t dst_bw = 0;
        
        for (; (src_bw + 8) < width; src_bw += 8, dst_bw += 4) {
          tmp_data[0] = in_base[src_bw];
          tmp_data[1] = in_base[src_bw+2];
          tmp_data[2] = in_base[src_bw+4];
          tmp_data[3] = in_base[src_bw+6];

          memcpy(rearranged_in_base + dst_bw, &tmp_data, sizeof(float) * 4);
        }

        for (; src_bw < width; src_bw += 2, dst_bw ++) {
          rearranged_in_base[dst_bw] = in_base[src_bw];
        }

        in_base += (width * strides_h);
        rearranged_in_base += rearranged_width;
      }
    }
  }
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
