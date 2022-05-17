
#include <unistd.h>
#include <vector>
#include <string.h>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/core/tensor_manage_util.h"
#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/common/transpose.h"
#include "mace/ops/ref/conv_2d.h"
#include "mace/ops/conv_2d_part_plan.h"
#include "mace/utils/statistics.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#endif

#include "test/codl_run/utils/io_util.h"
#include "test/codl_run/utils/device_util.h"
#include "test/codl_run/utils/tensor_buffer_util.h"
#include "test/codl_run/utils/conv_2d_util.h"
#include "test/codl_run/conv2d_test_param.h"
#ifndef MACE_BUILD_LIBRARY
#include "test/codl_run/conv2d_test_task.h"
#endif
#include "test/codl_run/conv2d_test.h"

namespace mace {

CollectGranularity ReadCollectGranularityFromEnv() {
  std::string collect_granularity;
  GetEnv("MACE_DURATION_COLLECT_GRANULARITY", &collect_granularity);
  if (!collect_granularity.empty()) {
    if (collect_granularity.compare("Fine") == 0) {
      return GRANULARITY_FINE;
    } else if (collect_granularity.compare("Coarse") == 0) {
      return GRANULARITY_COARSE;
    }
  }

  return GRANULARITY_NONE;
}

int Conv2dArgumentUtils::Parse(int argc, char *argv[], TestParam *param_out) {
  int ch;
  int cu_hint;
  std::string optarg_str;
  std::vector<std::vector<index_t>> shapes;
  Conv2dTestParam *conv2d_test_param =
      reinterpret_cast<Conv2dTestParam *>(param_out);
  while ((ch = getopt(argc, argv, "c:t:d:r:o:w:m:s:u:f:e:p:")) != -1) {
    switch (ch) {
      case 'c':
        conv2d_test_param->cpu_affinity_policy = StringToInt(optarg);
        break;
      case 't':
        conv2d_test_param->num_threads = StringToInt(optarg);
        break;
      case 'd':
        conv2d_test_param->part_dim = StringToInt(optarg);
        break;
      case 'r':
        conv2d_test_param->part_ratio = StringToFloat(optarg);
        break;
      case 'o':
        conv2d_test_param->total_rounds = StringToInt(optarg);
        break;
      case 'w':
        conv2d_test_param->wino_block_size = StringToInt(optarg);
        break;
      case 'm':
        conv2d_test_param->gpu_memory_type
            = static_cast<MemoryType>(StringToInt(optarg));
        break;
      case 's':
        optarg_str = std::string(optarg);
        shapes = ParseInputShapeString(optarg_str);
        conv2d_test_param->input_shape = shapes[0];
        conv2d_test_param->filter_shape = shapes[1];
        conv2d_test_param->strides = std::vector<int>{
            static_cast<int>(shapes[2][0]),
            static_cast<int>(shapes[2][1])};
        break;
      case 'u':
        cu_hint = StringToInt(optarg);
        conv2d_test_param->compute_unit_hint = static_cast<ComputeUnitHint>(cu_hint);
        break;
      case 'p':
        conv2d_test_param->test_program_idx = StringToInt(optarg);
        break;
      case '?':
      default:
        break;
    }
  }
  
  const char *do_data_transform = getenv("MACE_DO_DATA_TRANSFORM");
  conv2d_test_param->do_data_transform = (do_data_transform != nullptr &&
                                          strlen(do_data_transform) == 1 &&
                                          do_data_transform[0] == '1');

  const char *do_compute = getenv("MACE_DO_COMPUTE");
  if (do_compute != nullptr) {
    conv2d_test_param->do_compute = (strlen(do_compute) == 1 &&
                                     do_compute[0] == '1');
  } else {
    conv2d_test_param->do_compute = true;
  }
  
  const char *is_debug_on = getenv("MACE_IS_DEBUG_ON");
  conv2d_test_param->is_debug_on = (is_debug_on != nullptr &&
                                    strlen(is_debug_on) == 1 &&
                                    is_debug_on[0] == '1');
  
  return 0;
}

int Conv2dArgumentUtils::Check(const TestParam *param) {
  const Conv2dTestParam *conv2d_param =
      reinterpret_cast<const Conv2dTestParam *>(param);
  MACE_CHECK(conv2d_param->cpu_affinity_policy
                >= CPUAffinityPolicy::AFFINITY_NONE);
  MACE_CHECK(conv2d_param->cpu_affinity_policy
                <= CPUAffinityPolicy::AFFINITY_POWER_SAVE);
  MACE_CHECK(conv2d_param->num_threads > 0);
  MACE_CHECK(conv2d_param->total_rounds >= 0);
  MACE_CHECK(conv2d_param->part_ratio >= 0.0f &&
             conv2d_param->part_ratio <= 2.0f);
  MACE_CHECK(conv2d_param->wino_block_size == 0 ||
             conv2d_param->wino_block_size == 2 ||
             conv2d_param->wino_block_size == 4);
  MACE_CHECK(conv2d_param->gpu_memory_type == 0 ||
             conv2d_param->gpu_memory_type == 1);
  return 0;
}

int Conv2dArgumentUtils::Print(const TestParam *param) {
  const Conv2dTestParam *conv2d_param =
      reinterpret_cast<const Conv2dTestParam *>(param);
  LOG(INFO) << "===== Test Parameters =====";
  LOG(INFO) << "Do data transform: " << conv2d_param->do_data_transform;
  LOG(INFO) << "Do compute: " << conv2d_param->do_compute;
  LOG(INFO) << "CPU affinity policy: " << conv2d_param->cpu_affinity_policy;
  LOG(INFO) << "Number of threads: " << conv2d_param->num_threads;
  LOG(INFO) << "GPU memory type: " << conv2d_param->gpu_memory_type;
  LOG(INFO) << "Winogard block size: " << conv2d_param->wino_block_size;
  LOG(INFO) << "Partition pimension: " << conv2d_param->part_dim;
  LOG(INFO) << "Partition ratio: " << conv2d_param->part_ratio;
  LOG(INFO) << "Total rounds: " << conv2d_param->total_rounds;
  LOG(INFO) << "Compute unit hint: " << conv2d_param->compute_unit_hint;
  LOG(INFO) << "Input shape: " << ShapeToString(conv2d_param->input_shape);
  LOG(INFO) << "Filter shape: " << ShapeToString(conv2d_param->filter_shape);
  LOG(INFO) << "Filter strides: " << VectorToString<int>(conv2d_param->strides);
  LOG(INFO) << "===========================";
  return 0;
}

}  // namespace mace
