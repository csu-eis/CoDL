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

#ifndef MACE_OPS_COMMON_CONV_POOL_2D_UTIL_H_
#define MACE_OPS_COMMON_CONV_POOL_2D_UTIL_H_

#include <vector>
#include "mace/core/tensor.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {

enum Padding {
  VALID = 0,  // No padding
  SAME = 1,   // Pads with half the filter size (rounded down) on both sides
  FULL = 2,   // Pads with one less than the filter size on both sides
};

std::string PaddingTypeToString(const Padding padding_type);

enum RoundType {
  FLOOR = 0,
  CEIL = 1,
};

enum Conv2dImplType {
  CONV2D_DIRECT = 0,
  CONV2D_GEMM = 1,
  CONV2D_WINOGRAD = 2
};

namespace ops {

Conv2dImplType GetConv2dCpuImplType(const index_t input_channel,
                                    const index_t output_channel,
                                    const index_t kernel_size,
                                    const index_t stride);

Conv2dImplType GetConv2dGpuImplType();

void CalcPaddingAndOutputSize(const index_t *input_shape,
                              const DataFormat input_format,
                              const index_t *filter_shape,
                              const DataFormat filter_format,
                              const int *dilations,
                              const int *strides,
                              Padding padding,
                              index_t *output_shape,
                              int *padding_size);

void CalcNCHWPaddingAndOutputSize(const index_t *input_shape,
                                  const index_t *filter_shape,
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size);

void CalcNHWCPaddingAndOutputSize(const index_t *input_shape,
                                  const index_t *filter_shape,
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size);

void CalcOutputSize(const index_t *input_shape,
                    const DataFormat input_format,
                    const index_t *filter_shape,
                    const DataFormat filter_format,
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape);

void CalcOutputSize(const index_t *input_shape,   // NHWC
                    const index_t *filter_shape,  // OIHW
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape);

void CalcNCHWOutputSize(const index_t *input_shape,
                        const index_t *filter_shape,
                        const int *padding_size,
                        const int *dilations,
                        const int *strides,
                        const RoundType round_type,
                        index_t *output_shape);

void CalDeconvOutputShapeAndPadSize(const std::vector<index_t> &input_shape,
                                    const std::vector<index_t> &filter_shape,
                                    const std::vector<int> &strides,
                                    Padding padding_type,
                                    const std::vector<int> &paddings,
                                    const std::vector<int> &output_paddings,
                                    int group,
                                    std::vector<index_t> *output_shape,
                                    std::vector<int> *in_pad_size,
                                    std::vector<int> *out_pad_size,
                                    std::vector<index_t> *padded_out_shape,
                                    FrameworkType framework_type,
                                    DataFormat data_format);

void CalcWinogradOutTileSize(const index_t in_height,
                             const index_t in_width,
                             index_t *out_tile_size);

void CalcK3x3S1WinogradOutTileCount(const index_t in_height,
                                    const index_t in_width,
                                    const index_t out_height,
                                    const index_t out_width,
                                    int *pad_in_height,
                                    int *pad_in_width,
                                    int *pad_out_height,
                                    int *pad_out_width,
                                    int *total_out_tile_count,
                                    int *out_tile_size);

void GetOpenCLWidthBlockSize(const index_t k,
                             index_t *block_size);

void CalcOpenCLBlockCount(const index_t ow,
                          const index_t ic,
                          const index_t oc,
                          const index_t k,
                          int *owb,
                          int *icb,
                          int *ocb);

void CalcOpenCLDefaultLWS(OpenCLRuntime *runtime,
                          const uint32_t *gws,
                          const uint32_t kwg_size,
                          const uint32_t owb,
                          std::vector<uint32_t> &lws);

void CalcOpenCLConv2dK1x1LWS(OpenCLRuntime *runtime,
                             const uint32_t *gws,
                             const uint32_t kwg_size,
                             const uint32_t owb,
                             std::vector<uint32_t> &lws);

void CalcOpenCLConv2dK3x3LWS(OpenCLRuntime *runtime,
                             const uint32_t *gws,
                             const uint32_t kwg_size,
                             const uint32_t owb,
                             std::vector<uint32_t> &lws);

void CalcOpenCLConv2dGeneralLWS(OpenCLRuntime *runtime,
                                const uint32_t *gws,
                                const uint32_t kwg_size,
                                const uint32_t owb,
                                const uint32_t kernel_size,
                                std::vector<uint32_t> &lws);
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_CONV_POOL_2D_UTIL_H_
