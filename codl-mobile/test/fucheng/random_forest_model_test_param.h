
#ifndef TEST_FUCHENG_RANDOM_FOREST_TEST_PARAM_H_
#define TEST_FUCHENG_RANDOM_FOREST_TEST_PARAM_H_

#include "test/codlconv2drun/core/test_param.h"

namespace mace {

class RandomForestModelTestParam : public TestParam {
 public:
  int num_features;
  int num_estimators;
  int max_depth;
  int total_rounds;
};

}  // namespace mace

#endif  // TEST_FUCHENG_RANDOM_FOREST_TEST_PARAM_H_
