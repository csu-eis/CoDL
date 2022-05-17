
#include "mace/ops/arm/fp32/conv_2d_9x9.h"

#include <arm_neon.h>
#include <memory>

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

#define MACE_Conv2dArmv8NeonK9x9SnLoadCalc4        \
  /* load filter (4 outch x 1 height x 4 width) */ \
  float32x4_t vf00, vf01, vf02;                    \
  float32x4_t vf10, vf11, vf12;                    \
  float32x4_t vf20, vf21, vf22;                    \
  float32x4_t vf30, vf31, vf32;                    \
  vf00 = vld1q_f32(filter_ptr0);                   \
  vf01 = vld1q_f32(filter_ptr0 + 4);               \
  vf02 = vld1q_f32(filter_ptr0 + 5);               \
  vf10 = vld1q_f32(filter_ptr1);                   \
  vf11 = vld1q_f32(filter_ptr1 + 4);               \
  vf12 = vld1q_f32(filter_ptr1 + 5);               \
  vf20 = vld1q_f32(filter_ptr2);                   \
  vf21 = vld1q_f32(filter_ptr2 + 4);               \
  vf22 = vld1q_f32(filter_ptr2 + 5);               \
  vf30 = vld1q_f32(filter_ptr3);                   \
  vf31 = vld1q_f32(filter_ptr3 + 4);               \
  vf32 = vld1q_f32(filter_ptr3 + 5);               \
                                                   \
  /* outch 0 */                                    \
  vo0 = vfmaq_laneq_f32(vo0, vi0, vf00, 0);        \
  vo0 = vfmaq_laneq_f32(vo0, vi1, vf00, 1);        \
  vo0 = vfmaq_laneq_f32(vo0, vi2, vf00, 2);        \
  vo0 = vfmaq_laneq_f32(vo0, vi3, vf00, 3);        \
  vo0 = vfmaq_laneq_f32(vo0, vi4, vf01, 0);        \
  vo0 = vfmaq_laneq_f32(vo0, vi5, vf01, 1);        \
  vo0 = vfmaq_laneq_f32(vo0, vi6, vf01, 2);        \
  vo0 = vfmaq_laneq_f32(vo0, vi7, vf01, 3);        \
  vo0 = vfmaq_laneq_f32(vo0, vi8, vf02, 3);        \
                                                   \
  /* outch 1 */                                    \
  vo1 = vfmaq_laneq_f32(vo1, vi0, vf10, 0);        \
  vo1 = vfmaq_laneq_f32(vo1, vi1, vf10, 1);        \
  vo1 = vfmaq_laneq_f32(vo1, vi2, vf10, 2);        \
  vo1 = vfmaq_laneq_f32(vo1, vi3, vf10, 3);        \
  vo1 = vfmaq_laneq_f32(vo1, vi4, vf11, 0);        \
  vo1 = vfmaq_laneq_f32(vo1, vi5, vf11, 1);        \
  vo1 = vfmaq_laneq_f32(vo1, vi6, vf11, 2);        \
  vo1 = vfmaq_laneq_f32(vo1, vi7, vf11, 3);        \
  vo1 = vfmaq_laneq_f32(vo1, vi8, vf12, 3);        \
                                                   \
  /* outch 2 */                                    \
  vo2 = vfmaq_laneq_f32(vo2, vi0, vf20, 0);        \
  vo2 = vfmaq_laneq_f32(vo2, vi1, vf20, 1);        \
  vo2 = vfmaq_laneq_f32(vo2, vi2, vf20, 2);        \
  vo2 = vfmaq_laneq_f32(vo2, vi3, vf20, 3);        \
  vo2 = vfmaq_laneq_f32(vo2, vi4, vf21, 0);        \
  vo2 = vfmaq_laneq_f32(vo2, vi5, vf21, 1);        \
  vo2 = vfmaq_laneq_f32(vo2, vi6, vf21, 2);        \
  vo2 = vfmaq_laneq_f32(vo2, vi7, vf21, 3);        \
  vo2 = vfmaq_laneq_f32(vo2, vi8, vf22, 3);        \
                                                   \
  /* outch 3 */                                    \
  vo3 = vfmaq_laneq_f32(vo3, vi0, vf30, 0);        \
  vo3 = vfmaq_laneq_f32(vo3, vi1, vf30, 1);        \
  vo3 = vfmaq_laneq_f32(vo3, vi2, vf30, 2);        \
  vo3 = vfmaq_laneq_f32(vo3, vi3, vf30, 3);        \
  vo3 = vfmaq_laneq_f32(vo3, vi4, vf31, 0);        \
  vo3 = vfmaq_laneq_f32(vo3, vi5, vf31, 1);        \
  vo3 = vfmaq_laneq_f32(vo3, vi6, vf31, 2);        \
  vo3 = vfmaq_laneq_f32(vo3, vi7, vf31, 3);        \
  vo3 = vfmaq_laneq_f32(vo3, vi8, vf32, 3);

#define MACE_Conv2dArmv8NeonK9x9SnLoadCalc1        \
  /* load filter (1 outch x 1 height x 4 width) */ \
  float32x4_t vf00, vf01, vf02;                    \
  vf00 = vld1q_f32(filter_ptr0);                   \
  vf01 = vld1q_f32(filter_ptr0 + 4);               \
  vf02 = vld1q_f32(filter_ptr0 + 5);               \
                                                   \
  /* outch 0 */                                    \
  vo0 = vfmaq_laneq_f32(vo0, vi0, vf00, 0);        \
  vo0 = vfmaq_laneq_f32(vo0, vi1, vf00, 1);        \
  vo0 = vfmaq_laneq_f32(vo0, vi2, vf00, 2);        \
  vo0 = vfmaq_laneq_f32(vo0, vi3, vf00, 3);        \
  vo0 = vfmaq_laneq_f32(vo0, vi4, vf01, 0);        \
  vo0 = vfmaq_laneq_f32(vo0, vi5, vf01, 1);        \
  vo0 = vfmaq_laneq_f32(vo0, vi6, vf01, 2);        \
  vo0 = vfmaq_laneq_f32(vo0, vi7, vf01, 3);        \
  vo0 = vfmaq_laneq_f32(vo0, vi8, vf02, 3);

#define MACE_Conv2dArmv7NeonK9x9SnLoadCalc4               \
  /* load filter (4 outch x 1 height x 4 width) */        \
  float32x4_t vf00, vf01, vf02;                           \
  float32x4_t vf10, vf11, vf12;                           \
  float32x4_t vf20, vf21, vf22;                           \
  float32x4_t vf30, vf31, vf32;                           \
  vf00 = vld1q_f32(filter_ptr0);                          \
  vf01 = vld1q_f32(filter_ptr0 + 4);                      \
  vf02 = vld1q_f32(filter_ptr0 + 5);                      \
  vf10 = vld1q_f32(filter_ptr1);                          \
  vf11 = vld1q_f32(filter_ptr1 + 4);                      \
  vf12 = vld1q_f32(filter_ptr1 + 5);                      \
  vf20 = vld1q_f32(filter_ptr2);                          \
  vf21 = vld1q_f32(filter_ptr2 + 4);                      \
  vf22 = vld1q_f32(filter_ptr2 + 5);                      \
  vf30 = vld1q_f32(filter_ptr3);                          \
  vf31 = vld1q_f32(filter_ptr3 + 4);                      \
  vf32 = vld1q_f32(filter_ptr3 + 5);                      \
                                                          \
  /* outch 0 */                                           \
  vo0 = vmlaq_lane_f32(vo0, vi0, vget_low_f32(vf00), 0);  \
  vo0 = vmlaq_lane_f32(vo0, vi1, vget_low_f32(vf00), 1);  \
  vo0 = vmlaq_lane_f32(vo0, vi2, vget_high_f32(vf00), 0); \
  vo0 = vmlaq_lane_f32(vo0, vi3, vget_high_f32(vf00), 1); \
  vo0 = vmlaq_lane_f32(vo0, vi4, vget_low_f32(vf01), 0);  \
  vo0 = vmlaq_lane_f32(vo0, vi5, vget_low_f32(vf01), 1);  \
  vo0 = vmlaq_lane_f32(vo0, vi6, vget_high_f32(vf01), 0); \
  vo0 = vmlaq_lane_f32(vo0, vi7, vget_high_f32(vf01), 1); \
  vo0 = vmlaq_lane_f32(vo0, vi8, vget_high_f32(vf02), 1); \
                                                          \
  /* outch 1 */                                           \
  vo1 = vmlaq_lane_f32(vo1, vi0, vget_low_f32(vf10), 0);  \
  vo1 = vmlaq_lane_f32(vo1, vi1, vget_low_f32(vf10), 1);  \
  vo1 = vmlaq_lane_f32(vo1, vi2, vget_high_f32(vf10), 0); \
  vo1 = vmlaq_lane_f32(vo1, vi3, vget_high_f32(vf10), 1); \
  vo1 = vmlaq_lane_f32(vo1, vi4, vget_low_f32(vf11), 0);  \
  vo1 = vmlaq_lane_f32(vo1, vi5, vget_low_f32(vf11), 1);  \
  vo1 = vmlaq_lane_f32(vo1, vi6, vget_high_f32(vf11), 0); \
  vo1 = vmlaq_lane_f32(vo1, vi7, vget_high_f32(vf11), 1); \
  vo1 = vmlaq_lane_f32(vo1, vi8, vget_high_f32(vf12), 1); \
                                                          \
  /* outch 2 */                                           \
  vo2 = vmlaq_lane_f32(vo2, vi0, vget_low_f32(vf20), 0);  \
  vo2 = vmlaq_lane_f32(vo2, vi1, vget_low_f32(vf20), 1);  \
  vo2 = vmlaq_lane_f32(vo2, vi2, vget_high_f32(vf20), 0); \
  vo2 = vmlaq_lane_f32(vo2, vi3, vget_high_f32(vf20), 1); \
  vo2 = vmlaq_lane_f32(vo2, vi4, vget_low_f32(vf21), 0);  \
  vo2 = vmlaq_lane_f32(vo2, vi5, vget_low_f32(vf21), 1);  \
  vo2 = vmlaq_lane_f32(vo2, vi6, vget_high_f32(vf21), 0); \
  vo2 = vmlaq_lane_f32(vo2, vi7, vget_high_f32(vf21), 1); \
  vo2 = vmlaq_lane_f32(vo2, vi8, vget_high_f32(vf22), 1); \
                                                          \
  /* outch 3 */                                           \
  vo3 = vmlaq_lane_f32(vo3, vi0, vget_low_f32(vf30), 0);  \
  vo3 = vmlaq_lane_f32(vo3, vi1, vget_low_f32(vf30), 1);  \
  vo3 = vmlaq_lane_f32(vo3, vi2, vget_high_f32(vf30), 0); \
  vo3 = vmlaq_lane_f32(vo3, vi3, vget_high_f32(vf30), 1); \
  vo3 = vmlaq_lane_f32(vo3, vi4, vget_low_f32(vf31), 0);  \
  vo3 = vmlaq_lane_f32(vo3, vi5, vget_low_f32(vf31), 1);  \
  vo3 = vmlaq_lane_f32(vo3, vi6, vget_high_f32(vf31), 0); \
  vo3 = vmlaq_lane_f32(vo3, vi7, vget_high_f32(vf31), 1); \
  vo3 = vmlaq_lane_f32(vo3, vi8, vget_high_f32(vf32), 1);

#define MACE_Conv2dArmv7NeonK9x9SnLoadCalc1               \
  /* load filter (1 outch x 1 height x 4 width) */        \
  float32x4_t vf00, vf01, vf02;                           \
  vf00 = vld1q_f32(filter_ptr0);                          \
  vf01 = vld1q_f32(filter_ptr0 + 4);                      \
  vf02 = vld1q_f32(filter_ptr0 + 5);                      \
                                                          \
  /* outch 0 */                                           \
  vo0 = vmlaq_lane_f32(vo0, vi0, vget_low_f32(vf00), 0);  \
  vo0 = vmlaq_lane_f32(vo0, vi1, vget_low_f32(vf00), 1);  \
  vo0 = vmlaq_lane_f32(vo0, vi2, vget_high_f32(vf00), 0); \
  vo0 = vmlaq_lane_f32(vo0, vi3, vget_high_f32(vf00), 1); \
  vo0 = vmlaq_lane_f32(vo0, vi4, vget_low_f32(vf01), 0);  \
  vo0 = vmlaq_lane_f32(vo0, vi5, vget_low_f32(vf01), 1);  \
  vo0 = vmlaq_lane_f32(vo0, vi6, vget_high_f32(vf01), 0); \
  vo0 = vmlaq_lane_f32(vo0, vi7, vget_high_f32(vf01), 1); \
  vo0 = vmlaq_lane_f32(vo0, vi8, vget_high_f32(vf02), 1);

MaceStatus Conv2dK9x9S1::Compute(const OpContext *context,
                                 const Tensor *input,
                                 const Tensor *filter,
                                 Tensor *output) {
  int64_t t0;
  std::vector<double> delays;
  if (context->dura_collector() != nullptr) {
#ifdef CODL_ENABLE_NANOS
    t0 = NowNanos();
#else
    t0 = NowMicros();
#endif
  }
  std::unique_ptr<const Tensor> padded_input;
  std::unique_ptr<Tensor> padded_output;
  t0 = NowMicros();
  ResizeOutAndPadInOut(context,
                       input,
                       filter,
                       output,
                       1,
                       4,
                       &padded_input,
                       &padded_output);
  delays.push_back((NowMicros() - t0) / 1000.0);
  const Tensor *in_tensor = input;
  if (padded_input != nullptr) {
    in_tensor = padded_input.get();
  }
  Tensor *out_tensor = output;
  if (padded_output != nullptr) {
    out_tensor = padded_output.get();
  }
  out_tensor->Clear();

  //Tensor::MappingGuard in_guard(input);
  //Tensor::MappingGuard filter_guard(filter);
  //Tensor::MappingGuard out_guard(output);
  auto filter_data = filter->data<float>();
  auto input_data = in_tensor->data<float>();
  auto output_data = out_tensor->mutable_data<float>();

  auto &in_shape = in_tensor->shape();
  auto &out_shape = out_tensor->shape();

  const index_t batch = in_shape[0];
  const index_t in_channels = in_shape[1];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_channels = out_shape[1];
  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];

  const index_t in_image_size = in_height * in_width;
  const index_t out_image_size = out_height * out_width;
  const index_t in_batch_size = in_channels * in_image_size;
  const index_t out_batch_size = out_channels * out_image_size;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  t0 = NowMicros();
  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        if (m + 3 < out_channels) {
          float *out_ptr0_base =
              output_data + b * out_batch_size + m * out_image_size;
          float *out_ptr1_base =
              output_data + b * out_batch_size + (m + 1) * out_image_size;
          float *out_ptr2_base =
              output_data + b * out_batch_size + (m + 2) * out_image_size;
          float *out_ptr3_base =
              output_data + b * out_batch_size + (m + 3) * out_image_size;
          for (index_t c = 0; c < in_channels; ++c) {
            const float *in_ptr_base =
                input_data + b * in_batch_size + c * in_image_size;
            const float
                *filter_ptr0 = filter_data + m * in_channels * 81 + c * 81;
            const float *filter_ptr1 =
                filter_data + (m + 1) * in_channels * 81 + c * 81;
            const float *filter_ptr2 =
                filter_data + (m + 2) * in_channels * 81 + c * 81;
            const float *filter_ptr3 =
                filter_data + (m + 3) * in_channels * 81 + c * 81;
            for (index_t h = 0; h < out_height; ++h) {
              for (index_t w = 0; w + 3 < out_width; w += 4) {
                // input offset
                index_t in_offset = h * in_width + w;
                // output (4 outch x 1 height x 4 width): vo_outch_height
                float32x4_t vo0, vo1, vo2, vo3;
                // load output
                index_t out_offset = h * out_width + w;
                vo0 = vld1q_f32(out_ptr0_base + out_offset);
                vo1 = vld1q_f32(out_ptr1_base + out_offset);
                vo2 = vld1q_f32(out_ptr2_base + out_offset);
                vo3 = vld1q_f32(out_ptr3_base + out_offset);
                for (index_t r = 0; r < 9; ++r) {
                  // input (3 slide)
                  float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7, vi8;
                  // load input
                  vi0 = vld1q_f32(in_ptr_base + in_offset);
                  vi4 = vld1q_f32(in_ptr_base + in_offset + 4);
                  vi8 = vld1q_f32(in_ptr_base + in_offset + 8);
                  vi1 = vextq_f32(vi0, vi4, 1);
                  vi2 = vextq_f32(vi0, vi4, 2);
                  vi3 = vextq_f32(vi0, vi4, 3);
                  vi5 = vextq_f32(vi4, vi8, 1);
                  vi6 = vextq_f32(vi4, vi8, 2);
                  vi7 = vextq_f32(vi4, vi8, 3);

#if defined(__aarch64__)
                  MACE_Conv2dArmv8NeonK9x9SnLoadCalc4;
#else
                  MACE_Conv2dArmv7NeonK9x9SnLoadCalc4;
#endif

                  in_offset += in_width;
                  filter_ptr0 += 9;
                  filter_ptr1 += 9;
                  filter_ptr2 += 9;
                  filter_ptr3 += 9;
                }  // r

                vst1q_f32(out_ptr0_base + out_offset, vo0);
                vst1q_f32(out_ptr1_base + out_offset, vo1);
                vst1q_f32(out_ptr2_base + out_offset, vo2);
                vst1q_f32(out_ptr3_base + out_offset, vo3);

                filter_ptr0 -= 81;
                filter_ptr1 -= 81;
                filter_ptr2 -= 81;
                filter_ptr3 -= 81;
              }  // w
            }    // h
          }  // c
        } else {
          for (index_t mm = m; mm < out_channels; ++mm) {
            float *out_ptr0_base =
                output_data + b * out_batch_size + mm * out_image_size;
            for (index_t c = 0; c < in_channels; ++c) {
              const float *in_ptr_base =
                  input_data + b * in_batch_size + c * in_image_size;
              const float
                  *filter_ptr0 = filter_data + mm * in_channels * 81 + c * 81;
              for (index_t h = 0; h < out_height; ++h) {
                for (index_t w = 0; w + 3 < out_width; w += 4) {
                  // input offset
                  index_t in_offset = h * in_width + w;
                  // output (1 outch x 1 height x 4 width): vo_outch_height
                  float32x4_t vo0;
                  // load output
                  index_t out_offset = h * out_width + w;
                  vo0 = vld1q_f32(out_ptr0_base + out_offset);
                  for (index_t r = 0; r < 9; ++r) {
                    // input (3 slide)
                    float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7, vi8;
                    // load input
                    vi0 = vld1q_f32(in_ptr_base + in_offset);
                    vi4 = vld1q_f32(in_ptr_base + in_offset + 4);
                    vi8 = vld1q_f32(in_ptr_base + in_offset + 8);
                    vi1 = vextq_f32(vi0, vi4, 1);
                    vi2 = vextq_f32(vi0, vi4, 2);
                    vi3 = vextq_f32(vi0, vi4, 3);
                    vi5 = vextq_f32(vi4, vi8, 1);
                    vi6 = vextq_f32(vi4, vi8, 2);
                    vi7 = vextq_f32(vi4, vi8, 3);

#if defined(__aarch64__)
                    MACE_Conv2dArmv8NeonK9x9SnLoadCalc1;
#else
                    MACE_Conv2dArmv7NeonK9x9SnLoadCalc1;
#endif

                    in_offset += in_width;
                    filter_ptr0 += 9;
                  }  // r

                  vst1q_f32(out_ptr0_base + out_offset, vo0);
                  filter_ptr0 -= 81;
                }  // w
              }    // h
            }  // c
          }    // mm
        }      // if
      }        // m
    }          // b
  }, 0, batch, 1, 0, out_channels, 4);
  delays.push_back((NowMicros() - t0) / 1000.0);

  t0 = NowMicros();
  UnPadOutput(*out_tensor, output);
  delays.push_back((NowMicros() - t0) / 1000.0);

  if (context->dura_collector() != nullptr) {
#ifdef CODL_ENABLE_NANOS
    context->dura_collector()->Add((NowNanos() - t0) / 1000000.0);
#else
    context->dura_collector()->Add(delays);
#endif
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
