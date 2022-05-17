
#ifndef TEST_CODL_RUN_MATMUL_TEST_PARAM_H_
#define TEST_CODL_RUN_MATMUL_TEST_PARAM_H_

#include <vector>
#include "test/codl_run/core/compute_unit.h"
#include "test/codl_run/core/test_param.h"

namespace mace {

class MatMulTestParam : public TestParam {
 public:
  bool transpose_a;
  bool transpose_b;
  std::vector<index_t> rhs_shape;
};

} // namespace mace

#endif // TEST_CODL_RUN_MATMUL_TEST_PARAM_H_
