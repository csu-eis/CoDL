
#ifndef TEST_CODL_RUN_CONV2D_TEST_PARAM_H_
#define TEST_CODL_RUN_CONV2D_TEST_PARAM_H_

#include <vector>
#include "test/codl_run/core/compute_unit.h"
#include "test/codl_run/core/test_param.h"

namespace mace {

class Conv2dTestParam : public TestParam {
 public:
  int wino_block_size;
  std::vector<index_t> filter_shape;
  std::vector<int> strides;
};

}  // namespace mace

#endif  // TEST_CODL_RUN_CONV2D_TEST_PARAM_H_
