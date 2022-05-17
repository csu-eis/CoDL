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

#include "mace/core/runtime/opencl/opencl_util.h"

#include <utility>

#include "mace/utils/logging.h"
#include "mace/utils/math.h"

namespace mace {

namespace {
// [(C + 3) / 4 * W, N * H]
void CalInOutputImageShape(const std::vector<index_t> &shape, /* NHWC */
                           std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4 || shape.size() == 5 ||
             shape.size() == 3 || shape.size() == 1);
  image_shape->resize(2);
  if (shape.size() == 4) {
    (*image_shape)[0] = RoundUpDiv4(shape[3]) * shape[2];
    (*image_shape)[1] = shape[0] * shape[1];
  } else if (shape.size() == 5) {
    (*image_shape)[0] = RoundUpDiv4(shape[4]) * shape[3];
    (*image_shape)[1] = shape[0] * shape[1] * shape[2];
  } else if (shape.size() == 3) {
    (*image_shape)[0] = RoundUpDiv4(shape[2]) * shape[1];
    (*image_shape)[1] = shape[0];
  } else if (shape.size() == 1) {
    (*image_shape)[0] = RoundUpDiv4(shape[0]);
    (*image_shape)[1] = 1;
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

// [Ic, H * W * (Oc + 3) / 4]
void CalConv2dFilterImageShape(const std::vector<index_t> &shape, /* OIHW */
                               std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4 || shape.size() == 2);
  image_shape->resize(2);
  if (shape.size() == 4) {
    (*image_shape)[0] = shape[1];
    (*image_shape)[1] = shape[2] * shape[3] * RoundUpDiv4(shape[0]);
  } else if (shape.size() == 2) {
    (*image_shape)[0] = shape[1];
    (*image_shape)[1] = RoundUpDiv4(shape[0]);
  }
}

// [H * W * M, (Ic + 3) / 4]
void CalDepthwiseConv2dFilterImageShape(
    const std::vector<index_t> &shape, /* MIHW */
    std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[0] * shape[2] * shape[3];
  (*image_shape)[1] = RoundUpDiv4(shape[1]);
}

// [(size + 3) / 4, 1]
void CalArgImageShape(const std::vector<index_t> &shape,
                      std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 1 ||
             shape.size() == 3 && shape[1] == 1 && shape[2] == 1 ||
             shape.size() == 4 && shape[1] == 1 && shape[2] == 1 && shape[3] == 1);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[0]);
  (*image_shape)[1] = 1;
}

// Only support 3x3 now
// [ (Ic + 3) / 4, 16 * Oc]
void CalWinogradFilterImageShape(
    const std::vector<index_t> &shape, /* Oc, Ic, H, W*/
    std::vector<size_t> *image_shape,
    const int blk_size) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[1]);
  (*image_shape)[1] = (shape[0] * (blk_size + 2) * (blk_size + 2));
}


// [W * C, N * RoundUp<4>(H)]
void CalInOutHeightImageShape(const std::vector<index_t> &shape, /* NHWC */
                              std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[2] * shape[3];
  (*image_shape)[1] = shape[0] * RoundUpDiv4(shape[1]);
}

// [RoundUp<4>(W) * C, N * H]
void CalInOutWidthImageShape(const std::vector<index_t> &shape, /* NHWC */
                             std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[2]) * shape[3];
  (*image_shape)[1] = shape[0] * shape[1];
}

// [Ic * H * W, (Oc + 3) / 4]
void CalWeightHeightImageShape(const std::vector<index_t> &shape, /* OIHW */
                               std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[1] * shape[2] * shape[3];
  (*image_shape)[1] = RoundUpDiv4(shape[0]);
}

// [(Ic + 3) / 4 * H * W, Oc]
void CalWeightWidthImageShape(const std::vector<index_t> &shape, /* OIHW */
                              std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[1]) * shape[2] * shape[3];
  (*image_shape)[1] = shape[0];
}

// [W * H, N * (C + 3) / 4]
void CalGemmInOutputImageShape(const std::vector<index_t> &shape, /* NHWC */
                               std::vector<size_t> *image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[2] * shape[1];
  (*image_shape)[1] = shape[0] * RoundUpDiv4(shape[3]);
}
}  // namespace

void OpenCLUtil::CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                                 const OpenCLBufferType type,
                                 std::vector<size_t> *image_shape,
                                 const int wino_block_size) {
  MACE_CHECK_NOTNULL(image_shape);
  switch (type) {
    case CONV2D_FILTER:
      CalConv2dFilterImageShape(shape, image_shape);
      break;
    case DW_CONV2D_FILTER:
      CalDepthwiseConv2dFilterImageShape(shape, image_shape);
      break;
    case IN_OUT_CHANNEL:
      CalInOutputImageShape(shape, image_shape);
      break;
    case ARGUMENT:
      CalArgImageShape(shape, image_shape);
      break;
    case IN_OUT_HEIGHT:
      CalInOutHeightImageShape(shape, image_shape);
      break;
    case IN_OUT_WIDTH:
      CalInOutWidthImageShape(shape, image_shape);
      break;
    case WINOGRAD_FILTER:
      CalWinogradFilterImageShape(shape, image_shape, wino_block_size);
      break;
    case WEIGHT_HEIGHT:
      CalWeightHeightImageShape(shape, image_shape);
      break;
    case WEIGHT_WIDTH:
      CalWeightWidthImageShape(shape, image_shape);
      break;
    case GEMM_IN_OUT:
      CalGemmInOutputImageShape(shape, image_shape);
      break;
    default:
      LOG(FATAL) << "Mace not supported yet.";
  }
}

void OpenCLUtil::CalcWarpNumber(
    OpenCLRuntime *runtime,
    const uint32_t *gws,
    const uint32_t *lws,
    int *num_warps) {
  MACE_CHECK(num_warps != nullptr);

  const uint32_t device_compute_units = runtime->device_compute_units();
  const uint32_t warp_size = runtime->warp_size();
  
  const uint32_t len_gws = gws[0] * gws[1] * gws[2];
  const uint32_t len_lws = lws[0] * lws[1] * lws[2];
  const uint32_t num_wg = RoundUpDiv(len_gws, len_lws);
  const uint32_t num_wg_per_unit = RoundUpDiv(num_wg, device_compute_units);
  const uint32_t num_warps_per_wg = RoundUpDiv(len_lws, warp_size);
  const uint32_t num_warps_per_unit = num_warps_per_wg * num_wg_per_unit;
  *num_warps = static_cast<int>(num_warps_per_unit);

#if 0
  const std::vector<uint32_t> vgws = {gws[0], gws[1], gws[2]};
  const std::vector<uint32_t> vlws = {lws[0], lws[1], lws[2]};

  LOG(INFO) << "compute_units " << device_compute_units
            << ", warp_size " << warp_size;
  LOG(INFO) << "gws " << VectorToString<uint32_t>(vgws)
            << ", lws " << VectorToString<uint32_t>(vlws);
  LOG(INFO) << "len_gws " << len_gws << ", len_lws " << len_lws;
  LOG(INFO) << "num_wg " << num_wg << ", num_wg_per_unit " << num_wg_per_unit;
  LOG(INFO) << "num_warps_per_wg " << num_warps_per_wg
            << ", num_warps_per_unit " << num_warps_per_unit;
  LOG(INFO) << "num_warps " << *num_warps;
#endif
}

void OpenCLUtil::BuildTransformOpDef(
    const std::string &input_name,
    const std::vector<mace::index_t> &input_shape,
    const std::string &output_name,
    const mace::DataType dt,
    const OpenCLBufferType buffer_type,
    const mace::MemoryType mem_type,
    DataFormat data_format,
    OperatorDef *op_def) {
  std::string op_name = "mace_node_" + output_name;
  op_def->set_name(op_name);
  op_def->set_type("BufferTransform");
  op_def->add_input(input_name);
  op_def->add_output(output_name);
  op_def->set_device_type(DeviceType::GPU);
  Argument *arg = op_def->add_arg();
  arg->set_name("buffer_type");
  arg->set_i(static_cast<int32_t>(buffer_type));
  arg = op_def->add_arg();
  arg->set_name("mem_type");
  arg->set_i(static_cast<int32_t>(mem_type));
  arg = op_def->add_arg();
  arg->set_name("T");
  arg->set_i(static_cast<int32_t>(dt));
  arg = op_def->add_arg();
  arg->set_name("data_format");
  arg->set_i(static_cast<int>(data_format));
  if (!input_shape.empty()) {
    OutputShape *shape = op_def->add_output_shape();
    for (auto value : input_shape) {
      shape->add_dims(value);
    }
  }
}

}  // namespace mace
