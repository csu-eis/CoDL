#include <common.h>

__kernel void pad(BUFFER_OUT_OF_RANGE_PARAMS
                  GLOBAL_WORK_GROUP_SIZE_DIM3
                  __global DATA_TYPE *input,
                  __global DATA_TYPE *output,
#if PAD_TYPE == 0
                  __private const float constant_value,
#endif
                  __private const int input_height,
                  __private const int input_width,
                  __private const int output_height,
                  __private const int height_padding,
                  __private const int width_padding) {
  const int chan_blk_idx = get_global_id(0);
  const int width_idx    = get_global_id(1);
  const int hb_idx       = get_global_id(2);
  const int chan_idx     = chan_blk_idx << 2;
  const int batch_idx    = hb_idx / output_height;
  const int height_idx   = hb_idx - mul24(batch_idx, output_height);
  const int input_padded_height = input_height + height_padding;
  const int input_padded_width  = input_width + width_padding;

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_blk_idx >= global_size_dim0 ||
      width_idx    >= global_size_dim1 ||
      hb_idx       >= global_size_dim2) {
    return;
  }
#endif
  const int channel = global_size_dim0 << 2;
  const int width   = global_size_dim1;

#if PAD_TYPE == 0
  DATA_TYPE4 data = constant_value;
  if ((height_padding <= height_idx &&
       height_idx     <  input_padded_height) &&
      (width_padding <= width_idx &&
       width_idx     <  input_padded_width)) {
    const int input_offset = mad24(mad24(mad24(batch_idx,
                                input_height, (height_idx - height_padding)),
                                input_height, (width_idx - width_padding)),
                                channel, chan_idx);
    data = CONVERT4(vload4(0, input + input_offset));
  }
#elif PAD_TYPE == 1 || PAD_TYPE == 2
  const int diff_left = width_padding - width_idx;
  int w;

  if (diff_left > 0) {
#if PAD_TYPE == 1
    w = diff_left;
#else
    w = diff_left - 1;
#endif
  } else {
    const int diff_right = width_idx - input_padded_width;

    w = select(
        -diff_left,
#if PAD_TYPE == 1
        input_width - diff_right - 2,
#else
        input_width - diff_right - 1,
#endif
      diff_right >= 0
    );
  }

  const int diff_up = height_padding - height_idx;
  int h;

  if (diff_up > 0) {
#if PAD_TYPE == 1
    h = diff_up;
#else
    h = diff_up - 1;
#endif
  } else {
    const int diff_down = height_idx - input_padded_height;

    h = select(
        -diff_up,
#if PAD_TYPE == 1
        input_height - diff_down - 2,
#else
        input_height - diff_down - 1,
#endif
        diff_down >= 0
    );
  }

  const input_offset = mad24(mad24(mad24(batch_idx,
                          input_height, h),
                          input_width, w),
                          channel, chan_idx);
  const DATA_TYPE4 data = CONVERT4(vload4(0, input + input_offset));
#endif

  const int output_offset = mad24(mad24(hb_idx,
                                width, width_idx),
                                channel, chan_idx);
  if (chan_idx + 4 <= channel) {
    VSTORE4(CONVERT_TO(data, DATA_TYPE4), output, output_offset);
  } else {
    const int diff = channel - chan_idx;
    switch (diff) {
      case 3:
        output[output_offset + 2] = CONVERT_TO(data.z, DATA_TYPE);
      case 2:
        output[output_offset + 1] = CONVERT_TO(data.y, DATA_TYPE);
      case 1:
        output[output_offset] = CONVERT_TO(data.x, DATA_TYPE);
    }
    CHECK_OUT_OF_RANGE_FOR_BUFFER(output_offset + diff - 1);
  }
}
