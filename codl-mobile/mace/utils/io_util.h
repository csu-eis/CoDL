
#ifndef MACE_UTILS_IO_UTIL_H_
#define MACE_UTILS_IO_UTIL_H_

#include <stdio.h>
#include "mace/utils/logging.h"

class IoUtil {
public:
  static void Pause() {
    LOG(INFO) << "Press ENTER key to continue";
    getchar();
  }
};

#endif  // MACE_UTILS_IO_UTIL_H_
