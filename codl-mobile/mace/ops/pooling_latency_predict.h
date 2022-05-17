
#ifndef MACE_OPS_POOLING_LATENCY_PREDICT_H_
#define MACE_OPS_POOLING_LATENCY_PREDICT_H_

#include "mace/core/latency_predictor.h"
#include "mace/utils/linear_regression_model.h"

namespace mace {
namespace ops {

class PoolingCpuLatencyPredictor : public LatencyPredictor {
 public:
  PoolingCpuLatencyPredictor() : LatencyPredictor(1) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

class PoolingGpuLatencyPredictor : public LatencyPredictor {
 public:
  PoolingGpuLatencyPredictor() : LatencyPredictor(1) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

class PoolingFLOPsLatencyPredictor : public LatencyPredictor {
 public:
  PoolingFLOPsLatencyPredictor() : LatencyPredictor(1) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_POOLING_LATENCY_PREDICT_H_
