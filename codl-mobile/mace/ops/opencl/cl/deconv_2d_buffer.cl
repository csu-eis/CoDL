#include <common.h>

__kernel void deconv_2d(BUFFER_OUT_OF_RANGE_PARAMS
                        GLOBAL_WORK_GROUP_SIZE_DIM3
                        __global DATA_TYPE *input,
                        __global DATA_TYPE *weights,
#ifdef BIAS
                        __global DATA_TYPE *bias,
#endif
                        __global DATA_TYPE *output,
                        __private const float relux_max_limit,
                        __private const float leakyrelu_coefficient,
                        __private const int in_height,
                        __private const int in_width,
                        __private const int in_channel,
                        __private const int out_height,
                        __private const int out_width,
                        __private const int out_channel,
                        __private const int stride_h,
                        __private const int stride_w,
                        __private const float stride_h_r,
                        __private const float stride_w_r,
                        __private const int align_h,
                        __private const int align_w,
                        __private const int padding_h,
                        __private const int padding_w,
                        __private const int kernel_h,
                        __private const int kernel_w,
                        __private const int kernel_size,
                        __private const int in_channel_blocks,
                        __private const int out_channel_blocks) {
  const int c = get_global_id(0);
  const int w_id = get_global_id(1);
  const int hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (c >= global_size_dim0 || w_id >= global_size_dim1
      || hb >= global_size_dim2) {
    return;
  }
#endif

  const int out_channel_idx = c << 2;

#ifdef BIAS
  DATA_TYPE4 out0 = CONVERT4(vload4(0, bias + out_channel_idx));
  DATA_TYPE4 out1 = out0;
  DATA_TYPE4 out2 = out0;
  DATA_TYPE4 out3 = out0;
  DATA_TYPE4 out4 = out0;
#else
  DATA_TYPE4 out0 = 0;
  DATA_TYPE4 out1 = 0;
  DATA_TYPE4 out2 = 0;
  DATA_TYPE4 out3 = 0;
  DATA_TYPE4 out4 = 0;
#endif

  const int n_stride = mad(w_id, stride_w_r, 0);
  const int mod_stride = w_id - mul24(n_stride, stride_w);
  const int w = mad24(mul24(n_stride, 5), stride_w, mod_stride);
  const int b = hb / out_height;
  const int h = hb - mul24(b, out_height);
  if (w < out_width) {
    int start_x = floor((float) (w + align_w) * stride_w_r);
    int start_y = (h + align_h) * stride_h_r;
    start_y = max(0, start_y);

    int f_start_x = mad24(start_x, stride_w, padding_w) - w;
    int f_start_y = mad24(start_y, stride_h, padding_h) - h;
    f_start_x = kernel_w - 1 - f_start_x;
    f_start_y = kernel_h - 1 - f_start_y;

    DATA_TYPE4 in0, in1, in2, in3, in4;
    DATA_TYPE4 w0, w1, w2, w3;
    int idx_w0, idx_w1, idx_w2, idx_w3, idx_w4;

    for (int fy = f_start_y, idx_h = start_y; fy >= 0; fy -= stride_h, ++idx_h) {
      //int in_offset_base = mad24(b, in_height, idx_h) * in_width * in_channel;
      int in_offset_base = (b * in_height + idx_h) * in_width * in_channel;
      for (int fx = f_start_x, idx_w = start_x; fx >= 0; fx -= stride_w, ++idx_w) {
        idx_w0 = idx_w;
        idx_w1 = idx_w + 1;
        idx_w2 = idx_w + 2;
        idx_w3 = idx_w + 3;
        idx_w4 = idx_w + 4;
        for (int ic = 0; ic < in_channel_blocks; ++ ic) {
          int in_channel_idx = ic << 2;
          int f_offset = mad24(mad24(mad24(fy, kernel_w, fx),
                                out_channel_blocks, c),
                                in_channel, in_channel_idx) << 2;
          //int f_offset = (((fy * kernel_w + fx)
          //    * out_channel_blocks + c) * in_channel + in_channel_idx) << 2;

          // 4 ic (w0 to w3) and 4 oc (w0.x to w0.w) from kernel.
          w0 = CONVERT4(vload4(0, weights + f_offset));
          w1 = CONVERT4(vload4(0, weights + f_offset + 4));
          w2 = CONVERT4(vload4(0, weights + f_offset + 8));
          w3 = CONVERT4(vload4(0, weights + f_offset + 12));

          // 5 width and 4 ic from input.
          int in_offset;
          //in_offset = in_offset_base + mad24(idx_w##i, in_channel, in_channel_idx);
#define READ_INPUT(i)                                                          \
          in_offset = in_offset_base + idx_w##i * in_channel + in_channel_idx; \
          in##i = CONVERT4(vload4(0, input + in_offset));

          READ_INPUT(0);
          READ_INPUT(1);
          READ_INPUT(2);
          READ_INPUT(3);
          READ_INPUT(4);
#undef READ_INPUT

#define CALC_OUTPUT(i)                                     \
          out##i = mad((DATA_TYPE4)(in##i.x), w0, out##i); \
          out##i = mad((DATA_TYPE4)(in##i.y), w1, out##i); \
          out##i = mad((DATA_TYPE4)(in##i.z), w2, out##i); \
          out##i = mad((DATA_TYPE4)(in##i.w), w3, out##i);
          
          CALC_OUTPUT(0);
          CALC_OUTPUT(1);
          CALC_OUTPUT(2);
          CALC_OUTPUT(3);
          CALC_OUTPUT(4);
#undef CALC_OUTPUT
        }
      }
    }

#if  defined(USE_RELU) || defined(USE_LEAKYRELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
    out0 = do_activation(out0, relux_max_limit, leakyrelu_coefficient);
    out1 = do_activation(out1, relux_max_limit, leakyrelu_coefficient);
    out2 = do_activation(out2, relux_max_limit, leakyrelu_coefficient);
    out3 = do_activation(out3, relux_max_limit, leakyrelu_coefficient);
    out4 = do_activation(out4, relux_max_limit, leakyrelu_coefficient);
#endif

    //int out_offset = mad24(mad24(mad24(b, out_height, h),
    //    out_width, w), out_channel, out_channel_idx);
    int out_offset = ((b * out_height + h) * out_width + w)
                        * out_channel + out_channel_idx;

#define WRITE_OUTPUT(i)                                             \
    if (out_channel_idx + 4 > out_channel) {                        \
      const int diff = out_channel - out_channel_idx;               \
      switch(diff) {                                                \
        case 3:                                                     \
          output[out_offset + 2] = CONVERT_TO(out##i.z, DATA_TYPE); \
        case 2:                                                     \
          output[out_offset + 1] = CONVERT_TO(out##i.y, DATA_TYPE); \
        case 1:                                                     \
          output[out_offset] = CONVERT_TO(out##i.x, DATA_TYPE);     \
      }                                                             \
      CHECK_OUT_OF_RANGE_FOR_BUFFER(out_offset + diff - 1);         \
    } else {                                                        \
      VSTORE4(CONVERT_TO(out##i, DATA_TYPE4), output, out_offset);  \
    }

    WRITE_OUTPUT(0);
    if (w + 1 * stride_w >= out_width) return;
    out_offset += out_channel * stride_w;
    WRITE_OUTPUT(1);
    if (w + 2 * stride_w >= out_width) return;
    out_offset += out_channel * stride_w;
    WRITE_OUTPUT(2);
    if (w + 3 * stride_w >= out_width) return;
    out_offset += out_channel * stride_w;
    WRITE_OUTPUT(3);
    if (w + 4 * stride_w >= out_width) return;
    out_offset += out_channel * stride_w;
    WRITE_OUTPUT(4);
#undef WRITE_OUTPUT
  }
}
