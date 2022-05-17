
#ifndef TEST_CODL_RUN_FULLY_CONNECTED_TEST_PARAM_H_
#define TEST_CODL_RUN_FULLY_CONNECTED_TEST_PARAM_H_

#include <vector>
#include "test/codl_run/core/compute_unit.h"
#include "test/codl_run/core/test_param.h"

namespace mace {

class FullyConnectedTestParam : public TestParam {
 public:
  std::vector<index_t> weight_shape;
};

} // namespace mace

#endif // TEST_CODL_RUN_FULLY_CONNECTED_TEST_PARAM_H_
