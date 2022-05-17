// Copyright 2019 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_UTILS_COUNT_DOWN_LATCH_H_
#define MACE_UTILS_COUNT_DOWN_LATCH_H_

#include <atomic>  // NOLINT(build/c++11)
#include <condition_variable>  // NOLINT(build/c++11)
#include <mutex>  // NOLINT(build/c++11)

#include "mace/utils/spinlock.h"

namespace mace {
namespace utils {

class CountDownLatch {
 public:
  explicit CountDownLatch(int64_t spin_timeout)
      : spin_timeout_(spin_timeout), count_(0) {
    spinlock_.set_spin_wait_max_time(spin_timeout);
  }
  CountDownLatch(int64_t spin_timeout, int count)
      : spin_timeout_(spin_timeout), count_(count) {
    spinlock_.set_spin_wait_max_time(spin_timeout);
  }

  void Wait();

  void CountDown() {
    if (count_.fetch_sub(1, std::memory_order_release) == 1) {
      std::unique_lock<std::mutex> m(mutex_);
      cond_.notify_all();
    }
  }

  void Reset(int count) {
    count_.store(count, std::memory_order_release);
  }

  int count() const {
    return count_;
  }

  void set_spin_timeout(const int64_t timeout) {
    spin_timeout_ = timeout;
    spinlock_.set_spin_wait_max_time(timeout);
  }

 private:
  int64_t spin_timeout_;
  SpinLock spinlock_;
  std::atomic<int> count_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_COUNT_DOWN_LATCH_H_

