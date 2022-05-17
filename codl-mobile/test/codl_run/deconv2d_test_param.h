
#ifndef TEST_CODL_RUN_DECONV2D_TEST_PARAM_H_
#define TEST_CODL_RUN_DECONV2D_TEST_PARAM_H_

#include <vector>
#include "test/codl_run/core/compute_unit.h"
#include "test/codl_run/core/test_param.h"

namespace mace {

class Deconv2dTestParam : public TestParam {
 public:
  std::vector<index_t> filter_shape;
  std::vector<int> strides;
};

} // namespace mace

#endif // TEST_CODL_RUN_DECONV2D_TEST_PARAM_H_
