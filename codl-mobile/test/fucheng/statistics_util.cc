
#include "mace/utils/logging.h"
#include "test/fucheng/statistics_util.h"

void DelayStatisticsResult::Show(const std::string &title) {
  LOG(INFO) << title;
  LOG(INFO) << avg_ << " " << max_ << " " << min_;
}

DelayStatisticsResult* TestDelayStatistics::ComputeResult(
    const std::vector<double> &delays) {
  double sum, max, min;
  sum = 0.0f;
  max = delays.at(0);
  min = delays.at(0);
  
  for (const double &delay : delays) {
    sum += delay;
    if (delay > max) {
      max = delay;
    }
    if (delay < min) {
      min = delay;
    }
  }
  
  return new DelayStatisticsResult(sum / delays.size(), max, min);
}

void TestDelayStatistics::Show() {
  LOG(INFO) << "===== Stastistics =====";
  LOG(INFO) << "Avg Max Min";
  
  std::unique_ptr<DelayStatisticsResult> result;
  if (run_millis_trans_.at(TransposeType::TT_GPU_TO_CPU).size() > 0) {
    result.reset(ComputeResult(run_millis_trans_.at(TransposeType::TT_GPU_TO_CPU)));
    result->Show("Transpose (GPU to CPU)");
  }
  
  if (run_millis_trans_.at(TransposeType::TT_CPU_TO_GPU).size() > 0) {
    result.reset(ComputeResult(run_millis_trans_.at(TransposeType::TT_CPU_TO_GPU)));
    result->Show("Transpose (CPU to GPU)");
  }
  
  if (run_millis_cpu_.size() > 0) {
    result.reset(ComputeResult(run_millis_cpu_));
    result->Show("CPU");
  }
  
  if (run_millis_gpu_.size() > 0) {
    result.reset(ComputeResult(run_millis_gpu_));
    result->Show("GPU");
  }
}
