#include <common.h>

#define ENABLE_SELECT

#define READ_INPUT_FROM_IMAGE
#define READ_FILTER_FROM_IMAGE
#define COMPUTE
#define WRITE_OUTPUT_TO_IMAGE

// 3x3, w_blk_size=8
__kernel void conv_2d_3x3(OUT_OF_RANGE_PARAMS
                          GLOBAL_WORK_GROUP_SIZE_DIM3
                          __read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                          __read_only image2d_t filter, /* cout%4 * cin , kh * kw * cout/4 */
#ifdef BIAS
                          __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                          __write_only image2d_t output,
                          __private const float relux_max_limit,
                          __private const float leakyrelu_coefficient,
                          __private const int in_height,
                          __private const int in_width,
                          __private const int in_ch_blks,
                          __private const int out_height,
                          __private const int out_width,
                          __private const int stride_h,
                          __private const int stride_w,
                          __private const int padding_top,
                          __private const int padding_left,
                          __private const int dilation_h,
                          __private const int dilation_w) {
  const int out_ch_blk = get_global_id(0);
  const int out_w_blk = get_global_id(1);
  const int out_hb = get_global_id(2);

#if 0
  printf("gid [%d,%d,%d], lid [%lu,%lu,%lu], lsize [%lu,%lu,%lu], wid [%lu,%lu,%lu]\n",
      out_ch_blk, out_w_blk, out_hb,
      get_local_id(0), get_local_id(1), get_local_id(2),
      get_local_size(0), get_local_size(1), get_local_size(2),
      get_global_id(0) / get_local_size(0),
      get_global_id(1) / get_local_size(1),
      get_global_id(2) / get_local_size(2));

  barrier(CLK_LOCAL_MEM_FENCE);
#endif

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_ch_blk >= global_size_dim0 || out_w_blk >= global_size_dim1
      || out_hb >= global_size_dim2) {
    return;
  }
#endif
  const int out_w_blks = global_size_dim1;
  const int w_blk_size = WIDTH_BLOCK_SIZE;

#ifdef BIAS
  DATA_TYPE4 out0 = READ_IMAGET(bias, SAMPLER, (int2)(out_ch_blk, 0));
#if WIDTH_BLOCK_SIZE > 1
  DATA_TYPE4 out1 = out0;
  DATA_TYPE4 out2 = out0;
  DATA_TYPE4 out3 = out0;
#endif
#if WIDTH_BLOCK_SIZE > 4
  DATA_TYPE4 out4 = out0;
#endif
#if WIDTH_BLOCK_SIZE > 5
  DATA_TYPE4 out5 = out0;
  DATA_TYPE4 out6 = out0;
  DATA_TYPE4 out7 = out0;
#endif
#else  // BIAS
  DATA_TYPE4 out0 = 0;
#if WIDTH_BLOCK_SIZE > 1
  DATA_TYPE4 out1 = 0;
  DATA_TYPE4 out2 = 0;
  DATA_TYPE4 out3 = 0;
#endif
#if WIDTH_BLOCK_SIZE > 4
  DATA_TYPE4 out4 = 0;
#endif
#if WIDTH_BLOCK_SIZE > 5
  DATA_TYPE4 out5 = 0;
  DATA_TYPE4 out6 = 0;
  DATA_TYPE4 out7 = 0;
#endif
#endif  // BIAS

#ifdef USE_LWS
  int in_width_stride = mul24(out_w_blks, stride_w);
  int in_width0 = mad24(out_w_blk, stride_w, -padding_left);
#endif
#ifdef USE_SWS
  int in_width_stride = stride_w;
  int in_width0 = mad24(mul24(out_w_blk, w_blk_size), stride_w, -padding_left);
#endif
#if WIDTH_BLOCK_SIZE > 1
  int in_width1 = in_width0 + in_width_stride;
  int in_width2 = in_width1 + in_width_stride;
  int in_width3 = in_width2 + in_width_stride;
#endif
#if WIDTH_BLOCK_SIZE > 4
  int in_width4 = in_width3 + in_width_stride;
#endif
#if WIDTH_BLOCK_SIZE > 5
  int in_width5 = in_width4 + in_width_stride;
  int in_width6 = in_width5 + in_width_stride;
  int in_width7 = in_width6 + in_width_stride;
#endif

  const int height_start = mad24((out_hb % out_height), stride_h, -padding_top);
  int in_height_gap = select(
      0,
      (-height_start + dilation_h - 1) / dilation_h,
      height_start < 0);
  int in_height_start = mad24(in_height_gap, dilation_h, height_start);
  int in_height_end = min(mad24(3, dilation_h, height_start),
                          in_height);

  const int batch_idx = mul24((out_hb / out_height), in_height);
  const int filter_y_idx_start = mul24(out_ch_blk, 9) + mul24(in_height_gap, 3);

  DATA_TYPE4 in0;
#if WIDTH_BLOCK_SIZE > 1
  DATA_TYPE4 in1, in2, in3;
#endif
#if WIDTH_BLOCK_SIZE > 4
  DATA_TYPE4 in4;
#endif
#if WIDTH_BLOCK_SIZE > 5
  DATA_TYPE4 in5, in6, in7;
#endif

  DATA_TYPE4 weights0, weights1, weights2, weights3;
  for (short in_ch_blk = 0; in_ch_blk < in_ch_blks; ++in_ch_blk) {
    const int in_idx = mul24(in_ch_blk, in_width);
    int filter_x_idx = in_ch_blk << 2;
    int filter_y_idx = filter_y_idx_start;
    for (int hb_idx = in_height_start; hb_idx < in_height_end; hb_idx += dilation_h) {
      int in_hb_value = hb_idx + batch_idx;
      int in_width_idx = 0;
      for (short width_idx = 0; width_idx < 3; ++width_idx) {
        int in_width_value;

#ifdef READ_INPUT_FROM_IMAGE
#ifdef ENABLE_SELECT
#define READ_INPUT(i)                                                                \
        in_width_value = in_width##i + in_width_idx;                                 \
        in_width_value = select(in_idx + in_width_value,                             \
                                -1,                                                  \
                                (in_width_value < 0 || in_width_value >= in_width)); \
        in##i = READ_IMAGET(input, SAMPLER, (int2)(in_width_value, in_hb_value));
#else  // ENABLE_SELECT
#define READ_INPUT(i)                                                                \
        in_width_value = in_width##i + in_width_idx;                                 \
        in_width_value = in_idx + in_width_value;                                    \
        in##i = READ_IMAGET(input, SAMPLER, (int2)(in_width_value, in_hb_value));
#endif  // ENABLE_SELECT
#else  // READ_INPUT_FROM_IMAGE
#define READ_INPUT(i)                                                                \
        in##i = READ_IMAGET(input, SAMPLER, (int2)(0, 0));
#endif  // READ_INPUT_FROM_IMAGE

        READ_INPUT(0);
#if WIDTH_BLOCK_SIZE > 1
        READ_INPUT(1);
        READ_INPUT(2);
        READ_INPUT(3);
#endif
#if WIDTH_BLOCK_SIZE > 4
        READ_INPUT(4);
#endif
#if WIDTH_BLOCK_SIZE > 5
        READ_INPUT(5);
        READ_INPUT(6);
        READ_INPUT(7);
#endif

#undef READ_INPUT

#ifdef READ_FILTER_FROM_IMAGE
        // int filter_idx = (hb_idx * 3 + width_idx) * in_ch + (in_ch_blk << 2);
        weights0 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 0, filter_y_idx));
        weights1 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 1, filter_y_idx));
        weights2 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 2, filter_y_idx));
        weights3 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 3, filter_y_idx));
#else
        // weights0 = READ_IMAGET(filter, SAMPLER, (int2)(0, 0));
        // weights1 = READ_IMAGET(filter, SAMPLER, (int2)(0, 0));
        // weights2 = READ_IMAGET(filter, SAMPLER, (int2)(0, 0));
        // weights3 = READ_IMAGET(filter, SAMPLER, (int2)(0, 0));
        weights0 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 0, filter_y_idx));
        weights1 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 1, filter_y_idx));
        weights2 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 2, filter_y_idx));
        weights3 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 3, filter_y_idx));
#endif

#ifdef COMPUTE
        out0 = mad((DATA_TYPE4)(in0.x), weights0, out0);
        out0 = mad((DATA_TYPE4)(in0.y), weights1, out0);
        out0 = mad((DATA_TYPE4)(in0.z), weights2, out0);
        out0 = mad((DATA_TYPE4)(in0.w), weights3, out0);

#if WIDTH_BLOCK_SIZE > 1
        out1 = mad((DATA_TYPE4)(in1.x), weights0, out1);
        out1 = mad((DATA_TYPE4)(in1.y), weights1, out1);
        out1 = mad((DATA_TYPE4)(in1.z), weights2, out1);
        out1 = mad((DATA_TYPE4)(in1.w), weights3, out1);

        out2 = mad((DATA_TYPE4)(in2.x), weights0, out2);
        out2 = mad((DATA_TYPE4)(in2.y), weights1, out2);
        out2 = mad((DATA_TYPE4)(in2.z), weights2, out2);
        out2 = mad((DATA_TYPE4)(in2.w), weights3, out2);

        out3 = mad((DATA_TYPE4)(in3.x), weights0, out3);
        out3 = mad((DATA_TYPE4)(in3.y), weights1, out3);
        out3 = mad((DATA_TYPE4)(in3.z), weights2, out3);
        out3 = mad((DATA_TYPE4)(in3.w), weights3, out3);
#endif

#if WIDTH_BLOCK_SIZE > 4
        out4 = mad(in4.x, weights0, out4);
        out4 = mad(in4.y, weights1, out4);
        out4 = mad(in4.z, weights2, out4);
        out4 = mad(in4.w, weights3, out4);
#endif

#if WIDTH_BLOCK_SIZE > 5
        out5 = mad(in5.x, weights0, out5);
        out5 = mad(in5.y, weights1, out5);
        out5 = mad(in5.z, weights2, out5);
        out5 = mad(in5.w, weights3, out5);

        out6 = mad(in6.x, weights0, out6);
        out6 = mad(in6.y, weights1, out6);
        out6 = mad(in6.z, weights2, out6);
        out6 = mad(in6.w, weights3, out6);

        out7 = mad(in7.x, weights0, out7);
        out7 = mad(in7.y, weights1, out7);
        out7 = mad(in7.z, weights2, out7);
        out7 = mad(in7.w, weights3, out7);
#endif

#endif // COMPUTE

        in_width_idx += dilation_w;
        filter_y_idx += 1;
      }
    }
  }

#if  defined(USE_RELU) || defined(USE_LEAKYRELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  out0 = do_activation(out0, relux_max_limit, leakyrelu_coefficient);
#if WIDTH_BLOCK_SIZE > 1
  out1 = do_activation(out1, relux_max_limit, leakyrelu_coefficient);
  out2 = do_activation(out2, relux_max_limit, leakyrelu_coefficient);
  out3 = do_activation(out3, relux_max_limit, leakyrelu_coefficient);
#endif
#if WIDTH_BLOCK_SIZE > 4
  out4 = do_activation(out4, relux_max_limit, leakyrelu_coefficient);
#endif
#if WIDTH_BLOCK_SIZE > 5
  out5 = do_activation(out5, relux_max_limit, leakyrelu_coefficient);
  out6 = do_activation(out6, relux_max_limit, leakyrelu_coefficient);
  out7 = do_activation(out7, relux_max_limit, leakyrelu_coefficient);
#endif
#endif  // USE_RELU

  const int out_x_base = mul24(out_ch_blk, out_width);
#ifdef USE_LWS
  const int out_width_stride = out_w_blks;
  int w = out_w_blk;
#endif
#ifdef USE_SWS
  const int out_width_stride = 1;
  int w = out_w_blk * w_blk_size;
#endif

#ifdef WRITE_OUTPUT_TO_IMAGE
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out0);

#if WIDTH_BLOCK_SIZE > 1
  w += out_width_stride;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out1);

  w += out_width_stride;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out2);

  w += out_width_stride;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out3);
#endif

#if WIDTH_BLOCK_SIZE > 4
  w += out_width_stride;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out4);
#endif

#if WIDTH_BLOCK_SIZE > 5
  w += out_width_stride;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out5);

  w += out_width_stride;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out6);

  w += out_width_stride;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out7);
#endif

#else  // WRITE_OUTPUT_TO_IMAGE
  //WRITE_IMAGET(output, (int2)(0, 0), out0);
  //WRITE_IMAGET(output, (int2)(0, 0), out0);
  //WRITE_IMAGET(output, (int2)(0, 0), out0);
  //WRITE_IMAGET(output, (int2)(0, 0), out0);
  //WRITE_IMAGET(output, (int2)(0, 0), out0);
#endif  // WRITE_OUTPUT_TO_IMAGE
}
