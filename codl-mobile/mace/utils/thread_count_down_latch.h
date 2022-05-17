
#ifndef MACE_UTILS_THREAD_COUNT_DOWN_LATCH_H_
#define MACE_UTILS_THREAD_COUNT_DOWN_LATCH_H_

#include "mace/utils/thread_pool.h"

namespace mace {
namespace utils {

class SimpleCountDownLatch {
public:
  SimpleCountDownLatch(int count) : count_down_(count) {}
  SimpleCountDownLatch() : count_down_(0) {}
  ~SimpleCountDownLatch() {}

  inline int count_down() const {
    return count_down_.load(std::memory_order_acquire);
  }

  inline void SetCountDown(const int count) {
    count_down_.store(count, std::memory_order_release);
  }

  MaceStatus CountDown() {
    count_down_.fetch_sub(1, std::memory_order_release);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus Await() const;

  MaceStatus Await(const int64_t max_time_ns) const;

  MaceStatus Await(ThreadPool &thread_pool) const;

private:
  std::atomic<int> count_down_;
};

SimpleCountDownLatch *GetSimpleCountDownLatch();

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_THREAD_COUNT_DOWN_LATCH_H_
