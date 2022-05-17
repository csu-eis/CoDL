
#include "mace/utils/spinlock.h"

namespace mace {
namespace utils {

void SpinWait(const std::atomic<int> &variable,
              const int value,
              const int64_t spin_wait_max_time) {
  const auto start_time = std::chrono::high_resolution_clock::now();
  for (size_t k = 1; variable.load(std::memory_order_acquire) == value; ++k) {
    // k % 1000 == 0
    if (spin_wait_max_time > 0 && k > kIterationCount) {
      auto end_time = std::chrono::high_resolution_clock::now();
      int64_t elapse =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              end_time - start_time).count();
      if (elapse > spin_wait_max_time) {
        break;
      }
      k = 1;
    }
  }
}

void SpinWaitUntil(const std::atomic<int> &variable,
                   const int value,
                   const int64_t spin_wait_max_time) {
  const auto start_time = std::chrono::high_resolution_clock::now();
  for (size_t k = 1; variable.load(std::memory_order_acquire) != value; ++k) {
    // k % 1000 == 0
    if (spin_wait_max_time > 0 && k > kIterationCount) {
      auto end_time = std::chrono::high_resolution_clock::now();
      int64_t elapse =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              end_time - start_time).count();
      if (elapse > spin_wait_max_time) {
        break;
      }
      k = 1;
    }
  }
}

void SimpleSpinWaitInternal(const int64_t spin_wait_time) {
  const auto start_time = std::chrono::high_resolution_clock::now();
  for (size_t k = 1; ; ++k) {
    if (k > kIterationCount) {
      const auto end_time = std::chrono::high_resolution_clock::now();
      const int64_t elapse =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              end_time - start_time).count();
      if (elapse > spin_wait_time) {
        break;
      }
      k = 1;
    }
  }
}

}  // namespace utils
}  // namespace mace
