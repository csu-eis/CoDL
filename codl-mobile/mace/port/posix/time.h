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

#ifndef MACE_PORT_POSIX_TIME_H_
#define MACE_PORT_POSIX_TIME_H_

#include <sys/time.h>
#include <time.h>

#include <cstddef>

namespace mace {
namespace port {
namespace posix {

inline int64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

inline int64_t NowNanos() {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1e9 + ts.tv_nsec;
}

}  // namespace posix
}  // namespace port
}  // namespace mace

#endif  // MACE_PORT_POSIX_TIME_H_
