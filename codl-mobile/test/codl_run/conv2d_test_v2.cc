
#include <memory>
#include <vector>

#include "gflags/gflags.h"
#include "mace/utils/statistics.h"

#include "test/codl_run/conv2d_test_param.h"
#include "test/codl_run/conv2d_test_task.h"
#include "test/codl_run/conv2d_test.h"
#include "test/codl_run/pooling_test_param.h"
#include "test/codl_run/pooling_test_task.h"
#include "test/codl_run/fully_connected_test_param.h"
#include "test/codl_run/fully_connected_test_task.h"
#include "test/codl_run/deconv2d_test_param.h"
#include "test/codl_run/deconv2d_test_task.h"
#include "test/codl_run/matmul_test_param.h"
#include "test/codl_run/matmul_test_task.h"

namespace mace {

CodlTestTask *BuildTask(CodlTestTaskType type) {
  switch (type) {
    case CONV2D_CPU_GPU_TEST_TASK:
      return new CodlConv2dCpuGpuTestTask();
    case POOLING_CPU_GPU_TEST_TASK:
      return new CodlPoolingCpuGpuTestTask();
    case FC_CPU_GPU_TEST_TASK:
      return new CodlFullyConnectedCpuGpuTestTask();
    case DECONV2D_CPU_GPU_TEST_TASK:
      return new CodlDeconv2dCpuGpuTestTask();
    case MATMUL_CPU_GPU_TEST_TASK:
      return new CodlMatMulCpuGpuTestTask();
    default:
      LOG(ERROR) << "Unsupported task type: " << static_cast<int>(type);
      return nullptr;
  }
}

}  // namespace mace

DEFINE_string(op_type, "", "OP type");
DEFINE_bool(debug, false, "enable debug");
DEFINE_bool(data_transform, false, "enable data transforming");
DEFINE_bool(compute, false, "enable computation");
DEFINE_bool(warmup, false, "enable warm up");
DEFINE_int32(cpu_affinity_policy, 1,
             "0, 1, 2");
DEFINE_int32(num_threads, -1, "number of threads");
DEFINE_int32(gpu_memory_type, 0, "0, 1");
DEFINE_int32(rounds, 1, "rounds");
DEFINE_int32(cpu_dtype, 1, "cpu data type");
DEFINE_int32(gpu_dtype, 1, "gpu data type");
DEFINE_int32(part_dim, 0, "0, 1, 2, 3, 4");
DEFINE_double(part_ratio, 1, "partition ratio");
DEFINE_int32(cu_hint, 0, "0, 1, 2");
DEFINE_string(input_shape, "", "input shape");
DEFINE_string(weight_shape, "", "weight shape");
DEFINE_string(strides, "", "strides");
DEFINE_int32(wino_blk_size, 0, "0, 2, 4");
DEFINE_int32(pooling_type, 1, "1=AVG/2=MAX");
DEFINE_bool(transpose_a, false, "enable transpose A matmul");
DEFINE_bool(transpose_b, false, "enable transpose B matmul");

void ParseCommonTestParam(TestParam *param) {
  param->is_debug_on = FLAGS_debug;
  param->do_data_transform = FLAGS_data_transform;
  param->do_compute = FLAGS_compute;
  param->do_warmup = FLAGS_warmup;
  param->cpu_affinity_policy = FLAGS_cpu_affinity_policy;
  param->num_threads = FLAGS_num_threads;
  param->gpu_memory_type = static_cast<MemoryType>(FLAGS_gpu_memory_type);
  param->compute_unit_hint = static_cast<mace::ComputeUnitHint>(FLAGS_cu_hint);
  param->cpu_dtype = static_cast<mace::DataType>(FLAGS_cpu_dtype);
  param->gpu_dtype = static_cast<mace::DataType>(FLAGS_gpu_dtype);
  param->part_dim = FLAGS_part_dim;
  param->part_ratio = FLAGS_part_ratio;
  param->input_shape = mace::ArgumentUtils::ParseValueString<index_t>(FLAGS_input_shape);
  param->total_rounds = FLAGS_rounds;
}

void PrintCommonTestParam() {
  LOG(INFO) << "is_debug_on: " << FLAGS_debug;
  LOG(INFO) << "do_data_transform: " << FLAGS_data_transform;
  LOG(INFO) << "do_compute: " << FLAGS_compute;
  LOG(INFO) << "cpu_affinity_policy: " << FLAGS_cpu_affinity_policy;
  LOG(INFO) << "num_threads: " << FLAGS_num_threads;
  LOG(INFO) << "cpu_dtype: " << FLAGS_cpu_dtype;
  LOG(INFO) << "gpu_memory_type: " << FLAGS_gpu_memory_type;
  LOG(INFO) << "gpu_dtype: " << FLAGS_gpu_dtype;
  LOG(INFO) << "compute_unit_hint: " << FLAGS_cu_hint;
  LOG(INFO) << "part_dim: " << FLAGS_part_dim;
  LOG(INFO) << "part_ratio: " << FLAGS_part_ratio;
  LOG(INFO) << "total_rounds: " << FLAGS_rounds;
}

void ParseConv2dTestParam(Conv2dTestParam *param) {
  param->wino_block_size = FLAGS_wino_blk_size;
  param->filter_shape
      = mace::ArgumentUtils::ParseValueString<index_t>(FLAGS_weight_shape);
  param->strides = mace::ArgumentUtils::ParseValueString<int>(FLAGS_strides);
}

void ParsePoolingTestParam(PoolingTestParam *param) {
  param->filter_shape
      = mace::ArgumentUtils::ParseValueString<index_t>(FLAGS_weight_shape);
  param->strides = mace::ArgumentUtils::ParseValueString<int>(FLAGS_strides);
  param->pooling_type = FLAGS_pooling_type;
}

void ParseFullyConnectedTestParam(FullyConnectedTestParam *param) {
  param->weight_shape
      = mace::ArgumentUtils::ParseValueString<index_t>(FLAGS_weight_shape);
}

void ParseDeconv2dTestParam(Deconv2dTestParam *param) {
  param->filter_shape
      = mace::ArgumentUtils::ParseValueString<index_t>(FLAGS_weight_shape);
  param->strides = mace::ArgumentUtils::ParseValueString<int>(FLAGS_strides);
}

void ParseMatMulTestParam(MatMulTestParam *param) {
  param->transpose_a = FLAGS_transpose_a;
  param->transpose_b = FLAGS_transpose_b;
  param->rhs_shape
      = mace::ArgumentUtils::ParseValueString<index_t>(FLAGS_weight_shape);
}

int main(int argc, char *argv[]) {

#if 0
  Conv2dTestParam test_param;
  Conv2dArgumentUtils arg_utils;
  arg_utils.Parse(argc, argv, &test_param);
  arg_utils.Check(&test_param);
  arg_utils.Print(&test_param);
#endif

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CodlTestTaskType task_type = NONE_CPU_GPU_TEST_TASK;
  TestParam *test_param = nullptr;

  if (FLAGS_op_type.compare("Conv2D") == 0) {
    task_type = CONV2D_CPU_GPU_TEST_TASK;
    test_param = new Conv2dTestParam();
    ParseConv2dTestParam(reinterpret_cast<Conv2dTestParam *>(test_param));
  } else if (FLAGS_op_type.compare("Pooling") == 0) {
    task_type = POOLING_CPU_GPU_TEST_TASK;
    test_param = new PoolingTestParam();
    ParsePoolingTestParam(reinterpret_cast<PoolingTestParam *>(test_param));
  } else if (FLAGS_op_type.compare("FullyConnected") == 0) {
    task_type = FC_CPU_GPU_TEST_TASK;
    test_param = new FullyConnectedTestParam();
    ParseFullyConnectedTestParam(
        reinterpret_cast<FullyConnectedTestParam *>(test_param));
  } else if (FLAGS_op_type.compare("Deconv2D") == 0) {
    task_type = DECONV2D_CPU_GPU_TEST_TASK;
    test_param = new Deconv2dTestParam();
    ParseDeconv2dTestParam(reinterpret_cast<Deconv2dTestParam *>(test_param));
  } else if (FLAGS_op_type.compare("MatMul") == 0) {
    task_type = MATMUL_CPU_GPU_TEST_TASK;
    test_param = new MatMulTestParam();
    ParseMatMulTestParam(reinterpret_cast<MatMulTestParam *>(test_param));
  } else {
    LOG(ERROR) << "Unsupported op type: " << FLAGS_op_type;
  }

  ParseCommonTestParam(test_param);
  PrintCommonTestParam();

#ifdef MACE_ENABLE_OPENCL
  std::unique_ptr<CodlTestTask> task;
  // Create and run task.
  CodlTestTask *task_ptr = mace::BuildTask(task_type);
  if (task_ptr != nullptr) {
    task.reset(task_ptr);
    task->Prepare(test_param);
    const CollectGranularity granularity =
        mace::ReadCollectGranularityFromEnv();
    mace::DurationCollector<double> dura_collector(granularity);
    mace::DurationCollector<double> *dura_collector_ptr = nullptr;
    if (granularity == GRANULARITY_FINE) {
      dura_collector_ptr = &dura_collector;
    }
    //dura_collector.RunExample();
    for (int i = 0; i < test_param->total_rounds; i ++) {
      const int64_t t0 = NowMicros();
      task->Run(dura_collector_ptr);
      const double run_duration = (NowMicros() - t0) / 1000.0;
      if (granularity == GRANULARITY_COARSE) {
        dura_collector.Add(run_duration);
      }
      LOG(INFO) << "Round " << i
                << ", latency " << run_duration << " ms";
    }
    task->Destroy();
    task.reset(nullptr);
    
    const int si = 10;
    const StatMetric stat_metric = StatMetric::SM_AVERAGE;
    std::vector<double> stat;
    std::vector<double> stat_std_pect;
    if (stat_metric == StatMetric::SM_AVERAGE) {
      stat = dura_collector.StatAvg(si);
    } else if (stat_metric == StatMetric::SM_MEDIAN) {
      stat = dura_collector.StatMedian(si);
    }
    stat_std_pect = dura_collector.StatStdDevPect(si);
    dura_collector.DebugPrint();
    LOG(INFO) << "Stat: si " << si
              << ", stat " << VectorToString<double>(stat)
              << ", std_pect " << VectorToString<double>(stat_std_pect);
  }
#endif  // MACE_ENABLE_OPENCL
  
  return 0;
}
