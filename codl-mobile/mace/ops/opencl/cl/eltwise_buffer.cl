#include <common.h>

__kernel void eltwise(BUFFER_OUT_OF_RANGE_PARAMS
                      GLOBAL_WORK_GROUP_SIZE_DIM3
                      __global DATA_TYPE *input0,
#if defined(INPUT_SCALAR)
                      __private const float value,
#else
                      __global DATA_TYPE *input1,
#endif
                      __private const int height,
                      __private const int width,
                      __private const int channel,
#ifdef COEFF_SUM
                      __private const float coeff0,
                      __private const float coeff1,
#endif
                      __global DATA_TYPE *output) {
  const int chan_blk_idx = get_global_id(0);
  const int width_idx    = get_global_id(1);
  const int hb           = get_global_id(2);
  const int chan_idx     = chan_blk_idx << 2;

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_blk_idx >= global_size_dim0 ||
      width_idx    >= global_size_dim1 ||
      hb           >= global_size_dim2)
    return;
#endif
  
  int input_offset0 = mad24(mad24(hb, width, width_idx), channel, chan_idx);
  DATA_TYPE4 in0 = CONVERT4(vload4(0, input0 + input_offset0));
#if defined(INPUT_SCALAR)
  DATA_TYPE4 in1 = (DATA_TYPE4)(value, value, value, value);
#elif defined(INPUT_VECTOR)
  int input_offset1 = chan_idx;
  DATA_TYPE4 in1 = CONVERT4(vload4(0, input1 + input_offset1));
#elif defined(INPUT_BATCH_VECTOR)
  const int batch_idx = hb / height;
  int input_offset1 = mad24(batch_idx, channel, chan_idx);
  DATA_TYPE4 in1 = CONVERT4(vload4(0, input1 + input_offset1));
#elif defined(INPUT_TENSOR_BC_CHAN)
  int input_offset1 = mad24(hb, width, width_idx);
  DATA_TYPE4 tmp = CONVERT4(vload4(0, input1 + input_offset1));
  DATA_TYPE4 in1 = (DATA_TYPE4)(tmp.x, tmp.x, tmp.x, tmp.x);
#else
  int input_offset1 = mad24(mad24(hb, width, width_idx), channel, chan_idx);
  DATA_TYPE4 in1 = CONVERT4(vload4(0, input1 + input_offset1));
#endif

  DATA_TYPE4 out;
#if ELTWISE_TYPE == 0
  #ifdef COEFF_SUM
    out = mad(coeff0, in0, mad(coeff1, in1, 0));
  #else
    out = in0 + in1;
  #endif
#elif ELTWISE_TYPE == 1
  #ifdef SWAPPED
    out = in1 - in0;
  #else
    out = in0 - in1;
  #endif
#elif ELTWISE_TYPE == 2
  out = in0 * in1;
#elif ELTWISE_TYPE == 3
  #ifdef SWAPPED
    out = in1 / in0;
  #else
    out = in0 / in1;
  #endif
#elif ELTWISE_TYPE == 4
  out = fmin(in0, in1);
#elif ELTWISE_TYPE == 5
  out = fmax(in0, in1);
#elif ELTWISE_TYPE == 6
  in1 = (DATA_TYPE4)(0, 0, 0, 0);
  out = in1 - in0;
#elif ELTWISE_TYPE == 7
  out = fabs(in0);
#elif ELTWISE_TYPE == 8
  DATA_TYPE4 diff = in0 - in1;
  out = diff * diff;
#elif ELTWISE_TYPE == 9
  #ifdef SWAPPED
    out = pow(in1, in0);
  #else
    out = pow(in0, in1);
  #endif
#elif ELTWISE_TYPE == 11
  #ifdef SWAPPED
    out = floor(in1 / in0);
  #else
    out = floor(in0 / in1);
  #endif
#elif ELTWISE_TYPE == 12
  out = fmax(coeff0, fmin(coeff1, in0));
#endif

#if defined(NOT_DIVISIBLE_FOUR) &&                                       \
    ((ELTWISE_TYPE == 3 || ELTWISE_TYPE == 9 || ELTWISE_TYPE == 11)      \
     || ((defined(INPUT_SCALAR) || defined(INPUT_TENSOR_BC_CHAN)) &&     \
         (ELTWISE_TYPE == 0 || ELTWISE_TYPE == 1 || ELTWISE_TYPE == 4 || \
          ELTWISE_TYPE == 5 || ELTWISE_TYPE == 8 || ELTWISE_TYPE == 12)))
  const int remain_channel = channel - 4 * chan_blk_idx;
  if (remain_channel < 4) {
    switch (remain_channel) {
      case 1:
        out.y = 0;
      case 2:
        out.z = 0;
      case 3:
        out.w = 0;
    }
  }
#endif

  int output_offset = mad24(mad24(hb, width, width_idx), channel, chan_idx);
  if (chan_idx + 4 <= channel) {
    VSTORE4(CONVERT_TO(out, DATA_TYPE4), output, output_offset);
  } else {
    const int diff = channel - chan_idx;
    switch (diff) {
      case 3:
        output[output_offset + 2] = CONVERT_TO(out.z, DATA_TYPE);
      case 2:
        output[output_offset + 1] = CONVERT_TO(out.y, DATA_TYPE);
      case 1:
        output[output_offset] = CONVERT_TO(out.x, DATA_TYPE);
    }
    CHECK_OUT_OF_RANGE_FOR_BUFFER(output_offset + diff - 1);
  }
}
