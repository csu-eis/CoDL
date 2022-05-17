#include <common.h>

__kernel void activation(BUFFER_OUT_OF_RANGE_PARAMS
                         GLOBAL_WORK_GROUP_SIZE_DIM3
                         __global DATA_TYPE *input,
#ifdef USE_PRELU
                         __global DATA_TYPE *alpha,
#endif
                         __private const float relux_max_limit,
                         __private const float leakyrelu_coefficient,
                         __global DATA_TYPE *output) {
  const int ch_blk = get_global_id(0);
  const int w      = get_global_id(1);
  const int hb     = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (ch_blk >= global_size_dim0 ||
      w      >= global_size_dim1 ||
      hb     >= global_size_dim2) {
    return;
  }
#endif
  const int channel = global_size_dim0 << 2;
  const int width   = global_size_dim1;
  const int ch      = ch_blk << 2;

  const int pos = mad24(mad24(hb, width, w), channel, ch);
  DATA_TYPE4 in = CONVERT4(vload4(0, input + pos));
#ifdef USE_PRELU
  DATA_TYPE4 prelu_alpha = CONVERT4(vload4(0, alpha + ch));
  DATA_TYPE4 out = do_activation(in,
                                 prelu_alpha,
                                 relux_max_limit,
                                 leakyrelu_coefficient);
#else
  DATA_TYPE4 out = do_activation(in, relux_max_limit, leakyrelu_coefficient);
#endif

  if (ch + 4 <= channel) {
    VSTORE4(CONVERT_TO(out, DATA_TYPE4), output, pos);
  } else {
    const int diff = channel - ch;
    switch (diff) {
      case 3:
        output[pos + 2] = CONVERT_TO(out.z, DATA_TYPE);
      case 2:
        output[pos + 1] = CONVERT_TO(out.y, DATA_TYPE);
      case 1:
        output[pos] = CONVERT_TO(out.x, DATA_TYPE);
    }
    CHECK_OUT_OF_RANGE_FOR_BUFFER(pos + diff - 1);
  }
}
