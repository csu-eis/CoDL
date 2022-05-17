
#include "mace/utils/thread_count_down_latch.h"

namespace mace {
namespace utils {

MaceStatus SimpleCountDownLatch::Await() const {
  for (; count_down_.load(std::memory_order_acquire) != 0;) {
    for (size_t k = 0; k < 1000000; k ++);
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SimpleCountDownLatch::Await(const int64_t max_time_ns) const {
  auto start_time = std::chrono::high_resolution_clock::now();
  for (; count_down_.load(std::memory_order_acquire) != 0;) {
    for (size_t k = 0; k < 1000000; k ++);
    auto end_time = std::chrono::high_resolution_clock::now();
    const int64_t time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time).count();
    if (time_ns > max_time_ns) {
      break;
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SimpleCountDownLatch::Await(ThreadPool &thread_pool) const {
  MACE_UNUSED(thread_pool);

  for (; count_down_.load(std::memory_order_acquire) != 0;) {
    for (size_t k = 0; k < 1000000; k ++);
  }

  return MaceStatus::MACE_SUCCESS;
}

SimpleCountDownLatch *GetSimpleCountDownLatch() {
  static SimpleCountDownLatch count_down_latch;
  return &count_down_latch;
}

}  // namespace utils
}  // namespace mace
