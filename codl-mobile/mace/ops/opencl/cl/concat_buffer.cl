#include <common.h>

DATA_TYPE4 stitch_vector(DATA_TYPE4 left,
                         DATA_TYPE4 right,
                         const int pos,
                         const bool reversed) {
  if (!reversed) {
    switch (pos) {
      case 1:return (DATA_TYPE4)(left.x, right.x, right.y, right.z);
      case 2:return (DATA_TYPE4)(left.x, left.y, right.x, right.y);
      case 3:return (DATA_TYPE4)(left.x, left.y, left.z, right.x);
      default:return (DATA_TYPE4) 0;
    }
  } else {
    switch (pos) {
      case 1:return (DATA_TYPE4)(left.w, right.x, right.y, right.z);
      case 2:return (DATA_TYPE4)(left.z, left.w, right.x, right.y);
      case 3:return (DATA_TYPE4)(left.y, left.z, left.w, right.x);
      default:return (DATA_TYPE4) 0;
    }
  }
}

// Supported data type: half/float
__kernel void concat_channel(BUFFER_OUT_OF_RANGE_PARAMS
                             GLOBAL_WORK_GROUP_SIZE_DIM3
                             __global DATA_TYPE *input0,
                             __global DATA_TYPE *input1,
                             __private const int input0_chan,
                             __private const int input1_chan,
                             __global DATA_TYPE *output) {
  const int chan_blk_idx = get_global_id(0);
  const int width_idx    = get_global_id(1);
  const int hb_idx       = get_global_id(2);
  const int chan_idx     = chan_blk_idx << 2;

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_blk_idx >= global_size_dim0 ||
      width_idx    >= global_size_dim1 ||
      hb_idx       >= global_size_dim2) {
    return;
  }
#endif
  const int channel = global_size_dim0 << 2;
  const int width   = global_size_dim1;

  const int output_chan = input0_chan + input1_chan;
  const int input0_chan_blk = (input0_chan + 3) >> 2;
  const int output_chan_blk = (output_chan + 3) >> 2;

  DATA_TYPE4 data = 0;
#ifdef DIVISIBLE_FOUR
  if (chan_blk_idx + 1 <= input0_chan_blk) {
    const int input_offset0 = mad24(mad24(hb_idx,
                                  width, width_idx),
                                  channel, chan_idx);
    data = CONVERT4(vload4(0, input0 + input_offset0));
  } else {
    const int input_offset1 = mad24(mad24(hb_idx,
                                width, width_idx),
                                channel, (chan_blk_idx - input0_chan_blk) << 2);
    data = CONVERT4(vload4(0, input1 + input_offset1));
  }
#else
  if (chan_blk_idx < input0_chan_blk - 1) {
    const int input_offset0 = mad24(mad24(hb_idx,
                                  width, width_idx),
                                  channel, chan_idx);
    data = CONVERT4(vload4(0, input0 + input_offset0));
  } else if (chan_blk_idx >= input0_chan_blk) {
    const int in_chan_idx = chan_blk_idx - input0_chan_blk;
    const int input_offset1 = mad24(mad24(hb_idx,
                                  width, width_idx),
                                  channel, in_chan_idx << 2);
    DATA_TYPE4 data0 = CONVERT4(vload4(0, input1 + input_offset1));
    DATA_TYPE4 data1 = 0;
    if (((in_chan_idx + 1) << 2) < input1_chan) {
      const int input_offset1 = mad24(mad24(hb_idx,
                                  width, width_idx),
                                  channel, (in_chan_idx + 1) << 2);
      data1 = CONVERT4(vload4(0, input1 + input_offset1));
    }
    data = stitch_vector(data0, data1, input0_chan % 4, true);
  } else {  // if (chan_blk_idx == input0_chan_blk - 1)
    const int input_offset0 = mad24(mad24(hb_idx,
                                  width, width_idx),
                                  channel, chan_idx);
    DATA_TYPE4 data0 = CONVERT4(vload4(0, input0 + input_offset0));
    const int input_offset1 = mad24(hb_idx, width, width_idx);
    DATA_TYPE4 data1 = CONVERT4(vload4(0, input1 + input_offset1));
    data = stitch_vector(data0, data1, input0_chan % 4, false);
  }
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

// Required: All input channels are divisible by 4
__kernel void concat_channel_multi(BUFFER_OUT_OF_RANGE_PARAMS
                                   GLOBAL_WORK_GROUP_SIZE_DIM3
                                   __global DATA_TYPE *input,
                                   __private const int chan_blk_offset,
                                   __global DATA_TYPE *output) {
  const int chan_blk_idx = get_global_id(0);
  const int width_idx    = get_global_id(1);
  const int hb_idx       = get_global_id(2);
  const int chan_idx     = chan_blk_idx << 2;

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_blk_idx >= global_size_dim0 ||
      width_idx    >= global_size_dim1 ||
      hb_idx       >= global_size_dim2) {
    return;
  }
#endif
  const int channel = global_size_dim0 << 2;
  const int width = global_size_dim1;

  DATA_TYPE4 data = 0;
  const int input_offset = mad24(mad24(hb_idx, width, width_idx), channel, chan_idx);
  data = CONVERT4(vload4(0, input + input_offset));

  const int out_chan_idx = (chan_blk_idx + chan_blk_offset) << 2;
  const int output_offset = mad24(mad24(hb_idx,
                                width, width_idx),
                                channel, out_chan_idx);
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
