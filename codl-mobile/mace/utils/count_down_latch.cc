
#include "mace/utils/count_down_latch.h"

namespace mace {
namespace utils {

void CountDownLatch::Wait() {
  spinlock_.WaitUntil(count_, 0);
  if (count_.load(std::memory_order_acquire) != 0) {
    std::unique_lock<std::mutex> m(mutex_);
    while (count_.load(std::memory_order_acquire) != 0) {
      cond_.wait(m);
    }
  }
}

}  // namespace utils
}  // namespace mace
