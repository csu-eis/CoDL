
#include <unistd.h>
#include <stdlib.h>
#include <string>

#include "mace/utils/logging.h"
#include "mace/utils/statistics.h"
#include "test/fucheng/gemm_test_param.h"
#include "test/fucheng/gemm_test_task.h"
#include "test/fucheng/gemm_test.h"

namespace mace {

class GemmUtils {
public:
  static void Set(float *matrix,
                  const int rows,
                  const int cols,
                  const float v) {
    for (int i = 0; i < rows; i ++) {
      for (int j = 0; j < cols; j ++) {
        matrix[i * cols + j] = v;
      }
    }
  }

  static void Print(const float *matrix,
                    const int rows,
                    const int cols,
                    const std::string &name) {
    LOG(INFO) << "Matrix: " << name;
    std::vector<float> row;
    for (int i = 0; i < rows; i ++) {
      row.clear();
      for (int j = 0; j < cols; j ++) {
        row.push_back(matrix[i * cols + j]);
      }
      LOG(INFO) << VectorToString<float>(row);
    }
  }
};

MaceStatus RunGemmTest(GemmTestParam *param,
                       CodlGemmTestTask *task) {
  std::vector<double> duration_list;

  task->Prepare(param);
  
  for (int i = 0; i < param->warmup_rounds; i ++) {
    task->Run();
  }

  for (int i = 0; i < param->rounds; i ++) {
    int64_t t0 = NowMicros();
    task->Run();
    const double run_duration = (NowMicros() - t0) / 1000.0;
    duration_list.push_back(run_duration);
  }

  const std::vector<double> stat =
      mace::benchmark::StatRunDuration(duration_list, 0);
  LOG(INFO) << "Stat: si " << 0
            << " stat " << VectorToString<double>(stat);
  
  task->Destroy();

  return MaceStatus::MACE_SUCCESS;
}

} // namespace mace

int jni_main(const float *lhs,
             const float *rhs,
             const int rows,
             const int depth,
             const int cols,
             const int cpu_affinity_policy,
             const int num_threads,
             const int device_idx,
             const int memory_type_idx,
             const int warmup_rounds,
             const int rounds,
             float *output) {
  GemmTestParam param;
  param.lhs = lhs;
  param.rhs = rhs;
  param.rows = rows;
  param.depth = depth;
  param.cols = cols;
  param.cpu_affinity_policy = cpu_affinity_policy;
  param.num_threads = num_threads;
  param.warmup_rounds = warmup_rounds;
  param.rounds = rounds;
  param.memory_type_idx = memory_type_idx;
  param.output = output;

  enum Device {
    CPU = 0,
    GPU = 1
  };

  enum MemoryType {
    BUFFER = 0,
    IMAGE = 1
  };

  const Device device = static_cast<Device>(device_idx);
  const MemoryType memory_type = static_cast<MemoryType>(memory_type_idx);
  std::unique_ptr<CodlGemmTestTask> task;
  
  switch (device) {
    case Device::CPU:
      task.reset(new CodlGemmCpuTestTask());
      break;
    case Device::GPU:
      task.reset(new CodlGemmGpuTestTask());
      break;
  }

  if (task != nullptr) {
    LOG(INFO) << "device " << device_idx
              << " memory_type " << memory_type_idx;
    mace::RunGemmTest(&param, task.get());
  }

  return 0;
}

#ifndef MACE_BUILD_LIBRARY

class GemmArgumentUtils : public ArgumentUtils {
public:
  int Parse(int argc, char *argv[], TestParam *param_out) override {
    int ch;
    std::string *optarg_str;
    GemmTestParam *test_param =
        reinterpret_cast<GemmTestParam *>(param_out);
    while ((ch = getopt(argc, argv, "m:k:n:t:d:e:w:r:")) != -1) {
      switch (ch) {
        case 'm':
          test_param->rows = StringToInt(optarg);
          break;
        case 'k':
          test_param->depth = StringToInt(optarg);
          break;
        case 'n':
          test_param->cols = StringToInt(optarg);
          break;
        case 't':
          test_param->num_threads = StringToInt(optarg);
          break;
        case 'd':
          test_param->device_idx = StringToInt(optarg);
          break;
        case 'e':
          test_param->memory_type_idx = StringToInt(optarg);
          break;
        case 'w':
          test_param->warmup_rounds = StringToInt(optarg);
          break;
        case 'r':
          test_param->rounds = StringToInt(optarg);
          break;
        case '?':
        default:
          break;
      }
    }
    
    return 0;
  }

  int Check(const TestParam *param) override {
    MACE_UNUSED(param);
    return 0;
  }

  int Print(const TestParam *param) override {
    MACE_UNUSED(param);
    return 0;
  }

private:
};

int main(int argc, char *argv[]) {
  MACE_UNUSED(argc);
  MACE_UNUSED(argv);

  GemmArgumentUtils arg_utils;
  GemmTestParam param;
  arg_utils.Parse(argc, argv, &param);
  
  const int rows = param.rows;
  const int depth = param.depth;
  const int cols = param.cols;
  const int cpu_affinity_policy = 0;
  const int num_threads = param.num_threads;
  const int device_idx = param.device_idx;
  const int memory_type_idx = param.memory_type_idx;
  const int warmup_rounds = param.warmup_rounds;
  const int rounds = param.rounds;

  const int lhs_raw_size = rows * depth * sizeof(float);
  const int rhs_raw_size = depth * cols * sizeof(float);
  const int output_raw_size = rows * cols * sizeof(float);

  float *lhs = reinterpret_cast<float *>(malloc(lhs_raw_size));
  float *rhs = reinterpret_cast<float *>(malloc(rhs_raw_size));
  float *output = reinterpret_cast<float *>(malloc(output_raw_size));

  mace::GemmUtils::Set(lhs, rows, depth, 1);
  mace::GemmUtils::Set(rhs, depth, cols, 2);
  mace::GemmUtils::Set(output, rows, cols, 0);

  jni_main(lhs,
           rhs,
           rows,
           depth,
           cols,
           cpu_affinity_policy,
           num_threads,
           device_idx,
           memory_type_idx,
           warmup_rounds,
           rounds,
           output);

  mace::GemmUtils::Print(lhs, rows, depth, "lhs");
  mace::GemmUtils::Print(rhs, depth, cols, "rhs");
  mace::GemmUtils::Print(output, rows, cols, "output");

  free(lhs);
  free(rhs);
  free(output);

  return 0;
}
#endif
