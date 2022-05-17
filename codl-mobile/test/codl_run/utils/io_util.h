
#ifndef TEST_CODL_RUN_UTILS_IO_UTIL_H_
#define TEST_CODL_RUN_UTILS_IO_UTIL_H_

#include "mace/utils/logging.h"

inline void PressEnterKeyToContinue() {
  LOG(INFO) << "Press ENTER key to continue";
  getchar();
}

#endif  // TEST_CODL_RUN_UTILS_IO_UTIL_H_
