
#ifndef TEST_CODL_RUN_OP_CHAIN_SEARCH_H_
#define TEST_CODL_RUN_OP_CHAIN_SEARCH_H_

#include <vector>
#include <memory>
#include "mace/core/latency_predictor.h"
#include "test/codl_run/op_test_task_chain.h"

namespace mace {

enum LatencyAcquirement {
  LA_PROFILING = 0,
  LA_PREDICTION = 1
};

class ChainSearch {
 public:
  static int OptimalPartitionProfileOrPredict(
      const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
      const LatencyAcquirement acq,
      const LPBackend lp_backend,
      const bool profile_data_transform,
      const bool profile_compute,
      const int pdim_hint,
      const int pratio_hint,
      const int debug_level,
      std::vector<std::shared_ptr<CodlOpChainParam>> &out_op_params);

  static int Serial(const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
                    std::vector<CodlOpTaskChain> &op_chains);

  static int Heuristic(const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
                       std::vector<CodlOpTaskChain> &op_chains);

  static int Greedy(const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
                    const LatencyAcquirement acq,
                    const int baseline_idx,
                    const int pratio_hint,
                    const int debug_level,
                    std::vector<CodlOpTaskChain> &op_chains);

  static int Full(const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
                  std::vector<CodlOpTaskChain> &op_chains);

  static int BuildChainCases(const int op_count,
                             std::vector<std::set<std::vector<int>>> &chain_case_sets);

  static int PrintChainInfo(
      const std::vector<CodlOpTaskChain> &op_chains);

  static int PrintChainCases(
      const std::vector<std::set<std::vector<int>>> &chain_case_sets);
};

}  // namespace mace

#endif  // TEST_CODL_RUN_OP_CHAIN_SEARCH_H_
