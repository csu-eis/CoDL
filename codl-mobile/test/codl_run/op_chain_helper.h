
#ifndef TEST_CODL_RUN_OP_CHAIN_HELPER_H_
#define TEST_CODL_RUN_OP_CHAIN_HELPER_H_

#include "test/codl_run/op_test_task_chain.h"

namespace mace {

class CodlOpTaskChainHelper {
 public:
  static bool Equal(const CodlOpTaskChain *src_chain,
                    const std::vector<CodlOpTaskChain> &dst_chains);

  static void SoftFree(CodlOpTaskChain &op_chain) {
    op_chain.Destroy(DESTROY_TYPE_SOFT);
  }

  static void SoftFree(std::vector<CodlOpTaskChain> &op_chains) {
    for (auto &chain : op_chains) {
      chain.Destroy(DESTROY_TYPE_SOFT);
    }
  }

  static void HardFree(CodlOpTaskChain &op_chain) {
    op_chain.Destroy(DESTROY_TYPE_HARD);
  }

  static void HardFree(std::vector<CodlOpTaskChain> &op_chains) {
    for (auto &chain : op_chains) {
      chain.Destroy(DESTROY_TYPE_HARD);
    }
  }

  static index_t CalcTotalMemoryUsage(
      const std::vector<CodlOpTaskChain> &op_chains,
      const int debug_level = 0,
      index_t *max_cpu_input_raw_size_ptr = nullptr,
      index_t *max_cpu_output_raw_size_ptr = nullptr,
      index_t *const_raw_size_ptr = nullptr);
};

}  // namespace mace

#endif  // TEST_CODL_RUN_OP_CHAIN_HELPER_H_
