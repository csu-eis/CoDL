#include <common.h>

__kernel void reduce(BUFFER_OUT_OF_RANGE_PARAMS
                     GLOBAL_WORK_GROUP_SIZE_DIM3
                     __global DATA_TYPE *input,
                     __local float4 *local_buffer,
                     __private const int group_num,
                     __private const int compute_size,
                     __private const int last_index,
                     __private const int in_height,
                     __private const int in_width,
                     __private const float scale,
                     __private const int channel_blocks,
                     __global DATA_TYPE *output) {
  const int w  = get_local_id(0);
  const int h  = get_local_id(1);
  const int bc = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (bc >= global_size_dim2)
    return;
#endif
  const int width   = get_local_size(0);
  const int channel = channel_blocks << 2;
  const int index   = mad24(h, width, w);
  const int b       = bc / channel_blocks;
  const int ch      = mad24(b, -channel_blocks, bc) << 2;

  DATA_TYPE4 in;
#if REDUCE_TYPE == 1
  DATA_TYPE4 part_result = (DATA_TYPE4){MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT};
#elif REDUCE_TYPE == 2
  DATA_TYPE4 part_result = (DATA_TYPE4){-MAXFLOAT, -MAXFLOAT, -MAXFLOAT, -MAXFLOAT};
#elif REDUCE_TYPE == 3
  DATA_TYPE4 part_result = (DATA_TYPE4){1, 1, 1, 1};
#else
  DATA_TYPE4 part_result = (DATA_TYPE4){0, 0, 0, 0};
#endif
  const bool after_last = (last_index > 0 && index >= last_index);
  // After last index, each kernel only computes (compute_size - 1) elements.
  const int actual_compute_size = select(compute_size,
                                         compute_size - 1,
                                         after_last);
  const int base_offset = mul24(index, actual_compute_size);
  const int offset= select(base_offset,
                           base_offset + last_index,
                           after_last);
#pragma unroll
  for (int i = 0; i < actual_compute_size; ++i) {
    int element_idx = offset + i;
    int h_idx = element_idx / in_width;
    int w_idx = mad24(h_idx, -in_width, element_idx);
    int input_offset = mad24(mad24(mad24(b,
                          in_height, h_idx),
                          in_width, w_idx),
                          channel, ch);
    in = CONVERT4(vload4(0, input + input_offset));
// MIN
#if REDUCE_TYPE == 1
   part_result = fmin(part_result, in);
// MAX
#elif REDUCE_TYPE == 2
   part_result = fmax(part_result, in);
// PROD
#elif REDUCE_TYPE == 3
    part_result = part_result * in;
// MEAN or SUM
#else
    part_result = part_result + in;
#endif
  }

#if REDUCE_TYPE == 0
  part_result = part_result * scale;
#endif
  local_buffer[index] = part_result;

#ifdef NON_QUALCOMM_ADRENO
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

  if (w == 0 && h == 0) {
#if REDUCE_TYPE == 1
    DATA_TYPE4 out = (DATA_TYPE4){MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT};
#elif REDUCE_TYPE == 2
    DATA_TYPE4 out = (DATA_TYPE4){-MAXFLOAT, -MAXFLOAT, -MAXFLOAT, -MAXFLOAT};
#elif REDUCE_TYPE == 3
    DATA_TYPE4 out = (DATA_TYPE4){1, 1, 1, 1};
#else
    DATA_TYPE4 out = (DATA_TYPE4){0, 0, 0, 0};
#endif
#pragma unroll
    for (int i = 0; i < group_num; ++i) {
#if REDUCE_TYPE == 1
      out = fmin(out, local_buffer[i]);
#elif REDUCE_TYPE == 2
      out = fmax(out, local_buffer[i]);
#elif REDUCE_TYPE == 3
      out = out * local_buffer[i];
#else
      out = out + local_buffer[i];
#endif
    }
    //int output_offset = mad24(mad24(mad24(b, 1, 0), 1, 0), channel, ch);
    int output_offset = mad24(b, channel, ch);
    if (ch + 4 <= channel) {
      VSTORE4(CONVERT_TO(out, DATA_TYPE4), output, output_offset);
    } else {
      const int diff = channel - ch;
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
}
