
#ifndef TEST_FUCHENG_STATISTICS_UTIL_H_
#define TEST_FUCHENG_STATISTICS_UTIL_H_

#include <vector>
#include <memory>

enum TransposeType {
  TT_GPU_TO_CPU = 0,
  TT_CPU_TO_GPU
};

class DelayStatisticsResult {
public:
  DelayStatisticsResult(const double avg, const double max, const double min)
    : avg_(avg), max_(max), min_(min) {};
    
  double avg() { return avg_; }
  double max() { return max_; }
  double min() { return min_; }

  void Show(const std::string &title);
  
private:
  double avg_;
  double max_;
  double min_;
};

class TestDelayStatistics {
public:
  TestDelayStatistics() {
    run_millis_trans_ = std::vector<std::vector<double>>(2);
  }
  
  void AddCPUDelay(double delay) {
    run_millis_cpu_.push_back(delay);
  }
  
  void AddGPUDelay(double delay) {
    run_millis_gpu_.push_back(delay);
  }
  
  void AddTransposeDelay(double delay, TransposeType type) {
    run_millis_trans_.at(type).push_back(delay);
  }
  
  void Show();

private:
  DelayStatisticsResult* ComputeResult(const std::vector<double> &delays);

  std::vector<double> run_millis_cpu_;
  std::vector<double> run_millis_gpu_;
  std::vector<std::vector<double>> run_millis_trans_;
};

#endif  // TEST_FUCHENG_STATISTICS_UTIL_H_
