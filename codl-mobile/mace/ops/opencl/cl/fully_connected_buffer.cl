#include <common.h>

// output = weight * input + bias
__kernel void fully_connected(BUFFER_OUT_OF_RANGE_PARAMS
                              GLOBAL_WORK_GROUP_SIZE_DIM2
                              __global IN_DATA_TYPE *input,  // NHWC
                              __global IN_DATA_TYPE *weight,  // [H,W,OB,I,4(OB)]
#ifdef BIAS
                              __global IN_DATA_TYPE *bias,
#endif
                              __private const int input_height,
                              __private const int input_width,
                              __private const int input_channel,
                              __private const int weight_chan_size,
                              __private const int output_channel,
                              __private const float relux_max_limit,
                              __private const float leakyrelu_coefficient,
                              __global OUT_DATA_TYPE *output) {
  const int batch_idx = get_global_id(0);
  const int out_blk_idx = get_global_id(1);
  
#ifndef NON_UNIFORM_WORK_GROUP
  if (batch_idx >= global_size_dim0 || out_blk_idx >= global_size_dim1)
    return;
#endif

  const int out_chan_idx = out_blk_idx << 2;

#ifdef BIAS
  DATA_TYPE4 result = CONVERT4(vload4(0, bias + out_chan_idx));
#else
  DATA_TYPE4 result = 0;
#endif

  int in_offset = mad24(mad24(mad24(batch_idx, input_height, 0),
      input_width, 0), input_channel, 0);

  int weight_offset_base = mul24(out_blk_idx, input_channel) << 2;

  DATA_TYPE4 input_value;
  DATA_TYPE4 w0, w1, w2, w3;
  for (int h_idx = 0; h_idx < input_height; ++h_idx) {
    for (int w_idx = 0; w_idx < input_width; ++w_idx) {
      int weight_offset = weight_offset_base;
      for (int chan_idx = 0; chan_idx < input_channel; chan_idx += 4) {
        input_value = CONVERT4(vload4(0, input + in_offset));
        
        w0 = CONVERT4(vload4(0, weight + weight_offset));
        w1 = CONVERT4(vload4(0, weight + weight_offset + 4));
        w2 = CONVERT4(vload4(0, weight + weight_offset + 8));
        w3 = CONVERT4(vload4(0, weight + weight_offset + 12));

        result = mad((DATA_TYPE4)(input_value.x), w0, result);
        result = mad((DATA_TYPE4)(input_value.y), w1, result);
        result = mad((DATA_TYPE4)(input_value.z), w2, result);
        result = mad((DATA_TYPE4)(input_value.w), w3, result);

        weight_offset += 16;
        in_offset += 4;
      }
      weight_offset_base += weight_chan_size;
    }
  }
  
#if  defined(USE_RELU) || defined(USE_LEAKYRELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  result = do_activation(result, relux_max_limit, leakyrelu_coefficient);
#endif
  
  int out_offset = mad24(mad24(mad24(batch_idx, 1, 0),
      1, 0), output_channel, out_chan_idx);
  
  if (out_chan_idx + 4 > output_channel) {
    const int diff = output_channel - out_chan_idx;
    switch(diff) {
      case 3:
        output[out_offset + 2] = CONVERT_TO(result.z, OUT_DATA_TYPE);
      case 2:
        output[out_offset + 1] = CONVERT_TO(result.y, OUT_DATA_TYPE);
      case 1:
        output[out_offset] = CONVERT_TO(result.x, OUT_DATA_TYPE);
    }
    CHECK_OUT_OF_RANGE_FOR_BUFFER(out_offset + diff - 1);
  } else {
    VSTORE4(CONVERT_TO(result, OUT_DATA_TYPE4), output, out_offset);
  }
}
