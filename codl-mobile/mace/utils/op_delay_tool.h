
#ifndef MACE_UTILS_DELAY_TOOL_H_
#define MACE_UTILS_DELAY_TOOL_H_

#include <math.h>
#include <vector>
#include "mace/utils/logging.h"
#include "mace/utils/string_util.h"

namespace mace {

enum CollectGranularity {
  GRANULARITY_NONE   = 0,
  GRANULARITY_COARSE = 1,
  GRANULARITY_FINE   = 2,
};

template<class T>
class DurationCollector {
public:
  DurationCollector()
      : collect_granularity_(GRANULARITY_NONE) {}

  DurationCollector(const CollectGranularity granularity)
      : collect_granularity_(granularity) {}

  inline CollectGranularity collect_granularity() const {
    return collect_granularity_;
  }

  void Add(const std::vector<T> &durations) {
    if (dura_items_.size() == 0 ||
        dura_items_[0].size() == durations.size()) {
      dura_items_.push_back(durations);
#if 0
      LOG(INFO) << "Add durations: " << VectorToString<T>(durations);
#endif
    }
  }

  void Add(T duration) {
    const std::vector<T> durations = {duration};
    Add(durations);
  }

  void Append(T duration) {
    if (dura_items_.size() > 0) {
      dura_items_[dura_items_.size() - 1].push_back(duration);
#if 0
      LOG(INFO) << "Append duration: " << duration;
#endif
    }
  }

  inline size_t Size() const {
    return dura_items_.size();
  }

  inline std::vector<T> Get(const size_t i) const {
    if (i < dura_items_.size()) {
      return dura_items_[i];
    } else {
      return {};
    }
  }

  void Clear() {
    dura_items_.clear();
  }

  std::vector<T> StatSum(const size_t si = 0) {
    const size_t n = dura_items_.size();
    if (n == 0) {
      return std::vector<T>(0);
    }

    const size_t m = dura_items_[0].size();
    std::vector<T> dura_sum(m, 0);
    for (size_t i = si; i < n; i ++) {
      const std::vector<T> durations = dura_items_[i];
      for (size_t j = 0; j < m; j ++) {
        dura_sum[j] += durations[j];
      }
    }

    return dura_sum;
  }

  std::vector<T> StatAvg(const size_t si = 0) {
    const size_t n = dura_items_.size();
    if (n == 0) {
      return std::vector<T>(0);
    }

    std::vector<T> dura_avg = StatSum(si);

    const size_t m = dura_avg.size();
    for (size_t i = 0; i < m; i ++) {
      dura_avg[i] /= (n - si);
    }

    return dura_avg;
  }

  std::vector<T> StatMedian(const size_t si = 0) {
    const size_t n = dura_items_.size();
    if (n == 0) {
      return std::vector<T>(0);
    }

    const size_t m = dura_items_[0].size();
    std::vector<std::vector<T>> dura_arr(m);
    for (size_t i = si; i < n; i ++) {
      const std::vector<T> durations = dura_items_[i];
      for (size_t j = 0; j < m; j ++) {
        dura_arr[j].push_back(durations[j]);
      }
    }

    std::vector<T> dura_median(m, 0);
    for (size_t i = 0; i < m; i ++) {
      // Sort.
      std::sort(dura_arr[i].begin(), dura_arr[i].end());
      // Median number.
      const size_t k = dura_arr[i].size();
      if (k % 2) {
        dura_median[i] = dura_arr[i][k / 2];
      } else {
        dura_median[i] = (dura_arr[i][k / 2] + dura_arr[i][k / 2 - 1]) / 2.0f;
      }
    }

    return dura_median;
  }

  std::vector<double> StatStdDevPect(const size_t si = 0) {
    const size_t n = dura_items_.size();
    if (n == 0) {
      return std::vector<T>(0);
    }
    
    const std::vector<T> avg_values = StatAvg(si);
    
    const size_t m = dura_items_[0].size();
    std::vector<double> out_values = std::vector<T>(m, 0);
    for (size_t i = si; i < n; i ++) {
      const std::vector<T> durations = dura_items_[i];
      for (size_t j = 0; j < m; j ++) {
        const double d = (double) (durations[j] - avg_values[j]);
        out_values[j] += pow(d, 2);
      }
    }

    for (size_t i = 0; i < m; i ++) {
      out_values[i] = (sqrt(out_values[i] / (n - si)) * 100) / avg_values[i];
    }

    return out_values;
  }

  void DebugPrint(const size_t si = 0) {
    if (dura_items_.size() == 0) {
      return;
    }

    for (size_t i = si; i < dura_items_.size(); i ++) {
      const std::vector<T> durations = dura_items_[i];
      LOG(INFO) << VectorToString<T>(durations);
    }
  }

  void RunExample() {
    Add(1); Add(2); Add(3); Add(4); Add(5);
    Add(6); Add(7); Add(8); Add(9); Add(10);
    const std::vector<T> avg_values = StatAvg();
    const std::vector<T> median_values = StatMedian();
    const std::vector<T> sd_pect_values = StatStdDevPect();
    LOG(INFO) << "Average: " << VectorToString<T>(avg_values);
    LOG(INFO) << "Median: " << VectorToString<T>(median_values);
    LOG(INFO) << "SD percent: " << VectorToString<T>(sd_pect_values);
    Clear();
  }

  //std::vector<T> stat_avg_err();
private:
  CollectGranularity collect_granularity_;
  std::vector<std::vector<T>> dura_items_;
};

}  // mace

#endif  // MACE_UTILS_DELAY_TOOL_H_
