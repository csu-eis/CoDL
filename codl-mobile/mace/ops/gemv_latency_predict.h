
#ifndef MACE_OPS_GEMV_LATENCY_PREDICT_H_
#define MACE_OPS_GEMV_LATENCY_PREDICT_H_

#include "mace/core/latency_predictor.h"
#include "mace/utils/linear_regression_model.h"

namespace mace {
namespace ops {

class GemvCpuLatencyPredictor : public LatencyPredictor {
 public:
  GemvCpuLatencyPredictor() : LatencyPredictor(1) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const index_t h,
                 const index_t w);

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override {
    MACE_CHECK(inputs.size() == 2);
    const index_t h = static_cast<index_t>(inputs[0]);
    const index_t w = static_cast<index_t>(inputs[1]);
    return Predict(context, h, w);
  }

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_GEMV_LATENCY_PREDICT_H_
