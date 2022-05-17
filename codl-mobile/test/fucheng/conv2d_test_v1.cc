
#include <unistd.h>
#include <string>
#include <vector>

#include "mace/public/mace.h"
#include "mace/utils/logging.h"
#include "mace/utils/statistics.h"

#include "test/fucheng/io_util.h"
#include "test/fucheng/device_util.h"
#include "test/fucheng/tensor_buffer_util.h"
#include "test/fucheng/statistics_util.h"
#include "test/fucheng/conv_2d_util.h"
#include "test/fucheng/conv2d_test_param.h"
#include "test/fucheng/conv2d_test_task.h"
#include "test/fucheng/conv2d_test.h"

namespace mace {

// Test functions.
MaceStatus Conv2dGpuTestV1(TestDeviceContext *device_context);
MaceStatus Conv2dGpuTestV2(TestDeviceContext *dev_context,
                           const TestParam *test_param);
MaceStatus Conv2dCpuGpuTest(TestDeviceContext *device_context,
                            const TestParam *test_param);
// Refer to buffer_image_transform_test.cc.
MaceStatus BufferImageTransformTest(TestDeviceContext *device_context);
// Refer to buffer_image_transform_test_v2.cc.
MaceStatus BufferImageTransformTestV2(TestDeviceContext *device_context,
                                      const TestParam *test_param);
// Refer to buffer_map_copy_test.cc.
MaceStatus BufferMapAndCopyTest(TestDeviceContext *device_context);
// Refer to buffer_map_unmap_test.cc.
MaceStatus BufferMapUnmapTest(TestDeviceContext *device_context);
// Refer to make_partition_plan.cc.
MaceStatus MakePartitionPlanTest();

} // namcespace mace

#ifdef MACE_BUILD_LIBRARY

int jni_main(TestParam *param) {
  const Conv2dTestParam *conv2d_param =
      reinterpret_cast<const Conv2dTestParam *>(param);
  const TestProgramEnum test_program =
      static_cast<TestProgramEnum>(conv2d_param->test_program_idx);
  const int num_threads = conv2d_param->num_threads;
  const CPUAffinityPolicy policy
      = static_cast<CPUAffinityPolicy>(conv2d_param->cpu_affinity_policy);
      
  LOG(INFO) << "Test program: " << GetTestProgramName(test_program);
  
  TestDeviceContext device_context(num_threads, policy);
  device_context.InitCpuDevice();
  
#ifdef MACE_ENABLE_OPENCL
  device_context.InitGpuDevice();
  
  switch (test_program) {
    case TestProgramEnum::CONV2D_GPU_TEST:
      //mace::Conv2dGpuTestV1(&device_context);
      mace::Conv2dGpuTestV2(&device_context, param);
      break;
    case TestProgramEnum::CONV2D_CPU_GPU_TEST:
      mace::Conv2dCpuGpuTest(&device_context, param);
      break;
    default:
      MACE_NOT_IMPLEMENTED;
  }
#endif // MACE_ENABLE_OPENCL

  return 0;
}

#else // MACE_BUILD_LIBRARY

int main(int argc, char *argv[]) {
  Conv2dTestParam test_param;
  Conv2dArgumentUtils arg_utils;
  arg_utils.Parse(argc, argv, &test_param);
  arg_utils.Check(&test_param);
  arg_utils.Print(&test_param);
  
  const int num_threads = test_param.num_threads;
  const CPUAffinityPolicy policy
      = static_cast<CPUAffinityPolicy>(test_param.cpu_affinity_policy);
  
  std::unique_ptr<TestDeviceContext> dev_context(
      new TestDeviceContext(num_threads, policy));
  dev_context->InitCpuDevice();
  
#ifdef MACE_ENABLE_OPENCL
  dev_context->InitGpuDevice();
  
  const TestProgramEnum test_program =
      static_cast<TestProgramEnum>(test_param.test_program_idx);
  
  LOG(INFO) << "Test Program: " << GetTestProgramName(test_program);
  
  CodlTestTask *task_ptr = nullptr;
  std::unique_ptr<CodlTestTask> task;
  switch (test_program) {
    case TestProgramEnum::CONV2D_GPU_TEST:
      //Conv2dGpuTestV1(dev_context.get());
      Conv2dGpuTestV2(dev_context.get(), &test_param);
      break;
    case TestProgramEnum::CONV2D_CPU_GPU_TEST:
      //Conv2dCpuGpuTest(dev_context.get(), &test_param);
      
      // This case does not need device context.
      dev_context.reset(nullptr);
      // Create and run task.
      task_ptr = CodlTestTask::Build(CONV2D_CPU_GPU_TEST_TASK);
      if (task_ptr != nullptr) {
        task.reset(task_ptr);
        task->Prepare(&test_param);
        std::vector<double> duration_list;
        const CollectGranularity granularity =
            mace::ReadCollectGranularityFromEnv();
        mace::DurationCollector<double> dura_collector(granularity);
        mace::DurationCollector<double> *dura_collector_ptr = nullptr;
        if (granularity == GRANULARITY_FINE) {
          dura_collector_ptr = &dura_collector;
        }
        for (int i = 0; i < test_param.total_rounds; i ++) {
          const int64_t t0 = NowMicros();
          task->Run(dura_collector_ptr);
          const double run_duration = (NowMicros() - t0) / 1000.0;
          duration_list.push_back(run_duration);
          LOG(INFO) << "Round " << i
                    << " latency " << run_duration << " ms";
        }
        task->Destroy();
        task.reset(nullptr);
        
        const int si = 10;
        const StatMetric stat_metric = StatMetric::SM_AVERAGE;
        std::vector<double> stat;
        if (granularity == GRANULARITY_FINE) {
          if (stat_metric == StatMetric::SM_AVERAGE) {
            stat = dura_collector_ptr->stat_avg(si);
          } else if (stat_metric == StatMetric::SM_MEDIAN) {
            stat = dura_collector_ptr->stat_median(si);
          }
          //dura_collector_ptr->DebugPrint();
        } else {
          if (stat_metric == StatMetric::SM_AVERAGE) {
            stat = mace::benchmark::StatRunDuration(duration_list, si);
          } else if (stat_metric == StatMetric::SM_MEDIAN) {
            stat = mace::benchmark::StatRunDurationMedian(duration_list, si);
          }
        }
        LOG(INFO) << "Stat: si " << si
                  << " stat " << VectorToString<double>(stat);
      }
      break;
    case TestProgramEnum::BUFFER_IMAGE_TRANSFROM_TEST:
      BufferImageTransformTest(dev_context.get());
      break;
    case TestProgramEnum::BUFFER_IMAGE_TRANSFROM_TEST_V2:
      BufferImageTransformTestV2(dev_context.get(), &test_param);
      break;
    case TestProgramEnum::BUFFER_MAP_AND_COPY_TEST:
      BufferMapAndCopyTest(dev_context.get());
      break;
    case TestProgramEnum::BUFFER_MAP_UNMAP_TEST:
      BufferMapUnmapTest(dev_context.get());
      break;
    case TestProgramEnum::MAKE_PARTITION_PLAN_TEST:
      MakePartitionPlanTest();
      break;
    default:
      MACE_NOT_IMPLEMENTED;
  }
#endif // MACE_ENABLE_OPENCL
  
  return 0;
}

#endif // MACE_BUILD_LIBRARY
