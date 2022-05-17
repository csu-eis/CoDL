
#ifndef TEST_FUCHENG_GEMM_TEST_PARAM_H_
#define TEST_FUCHENG_GEMM_TEST_PARAM_H_

#include "test/fucheng/test_param.h"

namespace mace {

class GemmTestParam : public TestParam {
public:
  int rows;
  int depth;
  int cols;
  int num_tasks;
  int tile_size;
  int cpu_affinity_policy;
  int num_threads;
  int device_idx;
  int memory_type_idx;
  int warmup_rounds;
  int rounds;
  const float *lhs;
  const float *rhs;
  float *output;
};

} // namespace mace

#endif // TEST_FUCHENG_GEMM_TEST_PARAM_H_
