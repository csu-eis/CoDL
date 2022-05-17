
#ifndef MACE_OPS_OP_LATENCY_PREDICT_H_
#define MACE_OPS_OP_LATENCY_PREDICT_H_

#include "mace/core/latency_predictor.h"
#include "mace/utils/const_model.h"
#include "mace/utils/linear_regression_model.h"

namespace mace {
namespace ops {

class DataSharingLatencyPredictor : public LatencyPredictor {
 public:
  DataSharingLatencyPredictor() : LatencyPredictor(1) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

class SyncLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::ConstModel> c_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OP_LATENCY_PREDICT_H_
