#include <common.h>

__kernel void sqrdiff_mean(BUFFER_OUT_OF_RANGE_PARAMS
                           GLOBAL_WORK_GROUP_SIZE_DIM3
                           __global DATA_TYPE *input,
                           __global DATA_TYPE *input1,
                           __local float4 *group_sum,
                           __private const int group_size,
                           __private const int partial_len,
                           __private const int remain_index,
                           __private const int batch,
                           __private const int in_height,
                           __private const int in_width,
                           __private const int channels,
                           __private const float image_size_reciprocal,
                           __private const int channel_blocks,
                           __global DATA_TYPE *output) {
  const int i = get_local_id(0);
  const int j = get_local_id(1);
  const int k = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (k >= global_size_dim2)
    return;
#endif
  const int dim0_size = get_local_size(0);
  float4 tmp = (float4){0, 0, 0, 0};
  const int index = mad24(j, dim0_size, i);
  const int b = k / channel_blocks;
  const int ch_blk = mad24(b, -channel_blocks, k);
  const int ch = ch_blk << 2;

  DATA_TYPE4 in;
  const int valid_part_len = select(partial_len,
                                    partial_len - 1,
                                    remain_index > 0 && index >= remain_index);
  const int full_offset = mul24(index, partial_len);
  const int base_offset = select(full_offset,
                               full_offset - (index - remain_index),
                               valid_part_len < partial_len);
  float4 diff = (float4){0, 0, 0, 0};
  const int in1_offset = mad24(b, channels, ch);
  DATA_TYPE4 in1 = CONVERT4(vload4(0, input1 + in1_offset));
#pragma unroll
  for (int l = 0; l < valid_part_len; ++l) {
    int offset = base_offset + l;
    int h_id = offset / in_width;
    int w_id = mad24(h_id, -in_width, offset);
    int in_offset = mad24(mad24(mad24(b, in_height, h_id), in_width, w_id), channels, ch);
    in = CONVERT4(vload4(0, input + in_offset));
    diff = in- in1;
    tmp = tmp + diff * diff;
  }
  group_sum[index] = tmp * image_size_reciprocal;

#ifdef NON_QUALCOMM_ADRENO
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

  if (i == 0 && j == 0) {
    DATA_TYPE4 out = (DATA_TYPE4){0, 0, 0, 0};
#pragma unroll
    for (int l = 0; l < group_size; ++l) {
      out = out + group_sum[l];
    }
    const int pos = mad24(b, channels, ch);
    if (ch + 4 <= channels) {
      VSTORE4(CONVERT_TO(out, DATA_TYPE4), output, pos);
    } else {
      const int diff = channels - ch;
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
}
