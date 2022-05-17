
#ifndef MACE_OPS_GEMM_LATENCY_PREDICT_H_
#define MACE_OPS_GEMM_LATENCY_PREDICT_H_

#include "mace/core/latency_predictor.h"
#include "mace/utils/linear_regression_model.h"

namespace mace {
namespace ops {

class GemmCpuLatencyPredictor : public LatencyPredictor {
 public:
  GemmCpuLatencyPredictor(const index_t model_count)
      : LatencyPredictor(model_count) {}

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override {
    MACE_UNUSED(context);
    MACE_UNUSED(inputs);
    MACE_NOT_IMPLEMENTED;
    return 0;
  }
  
  virtual double Predict(OpContext *context,
                         const index_t m,
                         const index_t k,
                         const index_t n) = 0;
};

class GemmCpuLRLatencyPredictor : public GemmCpuLatencyPredictor {
 public:
  GemmCpuLRLatencyPredictor() : GemmCpuLatencyPredictor(3) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const index_t m,
                 const index_t k,
                 const index_t n) override;

 private:
  std::shared_ptr<utils::LinearRegressionModel> pack_lhs_lr_;
  std::shared_ptr<utils::LinearRegressionModel> pack_rhs_lr_;
  std::shared_ptr<utils::LinearRegressionModel> comp_lr_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_GEMM_LATENCY_PREDICT_H_
