
#ifndef TEST_CODL_RUN_OP_CHAIN_EXECUTOR_H_
#define TEST_CODL_RUN_OP_CHAIN_EXECUTOR_H_

#include "test/codl_run/op_test_task_chain.h"

namespace mace {

double CalcOpComputeDuration(const std::vector<double> &stat);

double CalcOpDuration(const std::vector<double> &stat);

std::string OpDurationToString(const std::vector<double> &stat);

class CodlOpTaskChainExecutor {
 public:
  static int Execute(std::vector<CodlOpTaskChain> &op_chains,
                     const int rounds,
                     const std::string &tag,
                     const int debug_level,
                     std::vector<double> &out_lats);

  static int Execute(std::vector<CodlOpTaskChain> &op_chains,
                     const int rounds,
                     const std::string &tag,
                     const int debug_level = 0,
                     double *out_lat = nullptr) {
    std::vector<double> out_lats;
    CodlOpTaskChainExecutor::Execute(op_chains, rounds, tag, debug_level, out_lats);
    if (out_lat != nullptr) {
      *out_lat = out_lats[0];
    }

    return 0;
  }
};

}  // namespace mace

#endif  // TEST_CODL_RUN_OP_CHAIN_EXECUTOR_H_
