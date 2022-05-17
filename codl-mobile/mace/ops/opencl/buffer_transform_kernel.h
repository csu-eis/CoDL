// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_OPENCL_BUFFER_TRANSFORM_KERNEL_H_
#define MACE_OPS_OPENCL_BUFFER_TRANSFORM_KERNEL_H_

#include "mace/core/runtime/opencl/opencl_util.h"
#include "mace/ops/common/transpose.h"
#include "mace/public/mace.h"
#include "mace/utils/math.h"

namespace mace {
class OpContext;
class Tensor;
namespace ops {

inline int ExtractDim0(const OdimRanges &odim_ranges) {
  for (size_t i = 0; i < odim_ranges.size(); i ++) {
    if (odim_ranges[i].size() > 0) {
      return static_cast<int>(i);
    }
  }

  return -1;
}

class OpenCLBufferTransformKernel {
 public:
  virtual MaceStatus Compute(OpContext *context,
                             const Tensor *input,
                             const OpenCLBufferType type,
                             const int wino_blk_size,
                             Tensor *output) = 0;

  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLBufferTransformKernel)
};

#ifdef MACE_ENABLE_CODL

class OpenCLPartBufferTransformKernel {
public:
  virtual MaceStatus Compute(OpContext *context,
                             const Tensor *input,
                             const OpenCLBufferType type,
                             const int wino_blk_size,
                             const OdimRanges &odim_ranges,
                             Tensor *output) = 0;

  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLPartBufferTransformKernel)
};

#endif  // MACE_ENABLE_CODL

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_TRANSFORM_KERNEL_H_
