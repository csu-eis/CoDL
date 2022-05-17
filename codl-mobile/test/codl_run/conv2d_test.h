
#ifndef TEST_CODL_RUN_CONV2D_TEST_H_
#define TEST_CODL_RUN_CONV2D_TEST_H_

#ifdef MACE_BUILD_LIBRARY
class TestParam {};
#else
#include <string>
#include "mace/utils/op_delay_tool.h"
#include "test/codl_run/core/test_param.h"
#endif

#ifdef MACE_BUILD_LIBRARY

int jni_main(TestParam *param);

#else  // MACE_BUILD_LIBRARY

namespace mace {

enum StatMetric {
  SM_AVERAGE = 0,
  SM_MEDIAN = 1,
};

CollectGranularity ReadCollectGranularityFromEnv();

class Conv2dArgumentUtils : public ArgumentUtils {
 public:
  int Parse(int argc, char *argv[], TestParam *param_out) override;
  int Check(const TestParam *param) override;
  int Print(const TestParam *param) override;
};

}  // namespace mace

#endif  // MACE_BUILD_LIBRARY

#endif  // TEST_CODL_RUN_CONV2D_TEST_H_
