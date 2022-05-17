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

#ifndef MACE_UTILS_SPINLOCK_H_
#define MACE_UTILS_SPINLOCK_H_

#include <thread>  // NOLINT(build/c++11)
#include <chrono>  // NOLINT(build/c++11)
#include <atomic>  // NOLINT(build/c++11)
#include "mace/port/port.h"
#include "mace/port/env.h"
#include "mace/utils/logging.h"

constexpr int kIterationCount = 1000;

namespace mace {
namespace utils {

void SpinWait(const std::atomic<int> &variable,
              const int value,
              const int64_t spin_wait_max_time = -1);

void SpinWaitUntil(const std::atomic<int> &variable,
                   const int value,
                   const int64_t spin_wait_max_time = -1);

void SimpleSpinWaitInternal(const int64_t spin_wait_time);

class SpinLock {
public:
  SpinLock()
    : spin_wait_max_time_(0) {}

  SpinLock(const int64_t t)
    : spin_wait_max_time_(t) {}

  inline void set_spin_wait_max_time(const int64_t t) {
    spin_wait_max_time_ = t;
  }

  inline void Wait(const std::atomic<int> &variable,
                   const int value) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t k = 1; variable.load(std::memory_order_acquire) == value; ++k) {
      if (spin_wait_max_time_ > 0 && k > kIterationCount) {
        auto end_time = std::chrono::high_resolution_clock::now();
        int64_t elapse =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
        if (elapse > spin_wait_max_time_) {
          break;
        }
        k = 1;
      }
    }
  }

  inline void WaitUntil(const std::atomic<int> &variable,
                        const int value) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t k = 1; variable.load(std::memory_order_acquire) != value; ++k) {
      if (spin_wait_max_time_ > 0 && k > kIterationCount) {
        auto end_time = std::chrono::high_resolution_clock::now();
        int64_t elapse =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
        if (elapse > spin_wait_max_time_) {
          break;
        }
        k = 1;
      }
    }
  }

private:
  int64_t spin_wait_max_time_;
};

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_SPINLOCK_H_
