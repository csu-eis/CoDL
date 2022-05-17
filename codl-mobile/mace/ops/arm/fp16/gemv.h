// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_OPS_ARM_FP16_GEMV_H_
#define MACE_OPS_ARM_FP16_GEMV_H_

#include "mace/core/types.h"

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
#include <arm_neon.h>
#endif

#if defined(MACE_ENABLE_NEON) && !defined(__aarch64__) && defined(__ANDROID__)
#define vaddvq_f32(v) ((v)[0] + (v)[1] + (v)[2] + (v)[3])
#endif

namespace mace {
namespace ops {

template<typename INPUT_TYPE_LEFT,
         typename INPUT_TYPE_RIGHT,
         typename OUTPUT_TYPE>
void FP16Gemv(const INPUT_TYPE_LEFT *m_ptr,
              const INPUT_TYPE_RIGHT *v_ptr,
              const index_t height,
              const index_t width,
              OUTPUT_TYPE *result);

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
template<>
void FP16Gemv<float16_t, float, float>(const float16_t *m_ptr,
                                       const float *v_ptr,
                                       const index_t height,
                                       const index_t width,
                                       float *out_ptr);
#endif

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FP16_GEMV_H_
