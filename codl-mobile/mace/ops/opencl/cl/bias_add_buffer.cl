#include <common.h>
// Supported data types: half/float
__kernel void bias_add(BUFFER_OUT_OF_RANGE_PARAMS
                       GLOBAL_WORK_GROUP_SIZE_DIM3
                       __global DATA_TYPE *input,
                       __global DATA_TYPE *bias,
                       __private const int height,
                       __private const int channels,
                       __global DATA_TYPE *output) {
  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (ch_blk >= global_size_dim0 || w >= global_size_dim1
      || hb >= global_size_dim2) {
    return;
  }
#endif
  const int width = global_size_dim1;

  const int batch_idx = hb / height;
  const int height_idx = mad24(batch_idx, -height, hb);
  const int channel_idx = ch_blk << 2;
  const int in_offset = mad24(mad24(mad24(batch_idx, height, height_idx),
      width, w), channels, channel_idx);
  DATA_TYPE4 in = CONVERT4(vload4(0, input + in_offset));
  const int bias_offset = channel_idx;
  DATA_TYPE4 bias_value = CONVERT4(vload4(0, bias + bias_offset));
  DATA_TYPE4 out = in + bias_value;

  const int out_offset = mad24(mad24(mad24(batch_idx, height, height_idx),
      width, w), channels, channel_idx);
  if (channel_idx + 4 > channels) {
    const int diff = channels - channel_idx;
    switch(diff) {
      case 3:
        output[out_offset + 2] = CONVERT_TO(out.z, DATA_TYPE);
      case 2:
        output[out_offset + 1] = CONVERT_TO(out.y, DATA_TYPE);
      case 1:
        output[out_offset] = CONVERT_TO(out.x, DATA_TYPE);
    }
    CHECK_OUT_OF_RANGE_FOR_BUFFER(out_offset + diff - 1);
  } else {
    VSTORE4(CONVERT_TO(out, DATA_TYPE4), output, out_offset);
  }
}
