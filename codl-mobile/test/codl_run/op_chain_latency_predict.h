
#ifndef TEST_CODL_RUN_OP_CHAIN_LATENCY_PREDICT_H_
#define TEST_CODL_RUN_OP_CHAIN_LATENCY_PREDICT_H_

#include <vector>
#include <memory>
#include "mace/core/latency_predictor.h"
#include "test/codl_run/op_test_task_chain.h"

namespace mace {

class OpChainLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override {
    MACE_UNUSED(context);
    MACE_UNUSED(inputs);
    MACE_NOT_IMPLEMENTED;
    return 0;
  }

  int Predict(std::vector<CodlOpTaskChain> &op_chains,
              const int debug_level,
              double *lat_ptr);

  int Predict(CodlOpTaskChain &op_chain,
              const int debug_level,
              double *lat_ptr);

 private:
  double Predict(OpContext *context,
                 const std::vector<CodlOpType> &op_types,
                 const std::vector<std::vector<double>> &inputs,
                 const std::vector<ops::OpPartPlan *> part_plans,
                 const int debug_level);

 private:
  LPBackend backend_;
  std::shared_ptr<LatencyPredictor> dt_in_predictor_;
  std::shared_ptr<LatencyPredictor> map_in_predictor_;
  std::shared_ptr<LatencyPredictor> map_out_predictor_;
  std::shared_ptr<LatencyPredictor> sync_predictor_;
  std::shared_ptr<LatencyPredictor> dt_out_predictor_;
  std::map<CodlOpType, std::shared_ptr<LatencyPredictor>> cpu_predictors_;
  std::map<CodlOpType, std::shared_ptr<LatencyPredictor>> gpu_predictors_;
};

}  // namespace mace

#endif  // TEST_CODL_RUN_OP_CHAIN_LATENCY_PREDICT_H_
