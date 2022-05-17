
#ifndef MACE_OPS_FULLY_CONNECTED_LATENCY_PREDICT_H_
#define MACE_OPS_FULLY_CONNECTED_LATENCY_PREDICT_H_

#include "mace/core/latency_predictor.h"
#include "mace/ops/gemv_latency_predict.h"
#include "mace/utils/linear_regression_model.h"

namespace mace {
namespace ops {

class FullyConnectedCpuLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<GemvCpuLatencyPredictor> cpu_gemv_predictor_;
};

class FullyConnectedGpuLatencyPredictor : public LatencyPredictor {
 public:
  FullyConnectedGpuLatencyPredictor() : LatencyPredictor(1) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

class FullyConnectedFLOPsLatencyPredictor : public LatencyPredictor {
 public:
  FullyConnectedFLOPsLatencyPredictor() : LatencyPredictor(1) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_FULLY_CONNECTED_LATENCY_PREDICT_H_
