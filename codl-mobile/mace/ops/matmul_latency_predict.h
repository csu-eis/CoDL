
#ifndef MACE_OPS_MATMUL_LATENCY_PREDICT_H_
#define MACE_OPS_MATMUL_LATENCY_PREDICT_H_

#include "mace/core/latency_predictor.h"
#include "mace/ops/gemm_latency_predict.h"
#include "mace/utils/linear_regression_model.h"

namespace mace {
namespace ops {

class MatMulCpuLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init(LPInitContext *context) override {
    gemm_predictor_.reset(new GemmCpuLRLatencyPredictor());
    gemm_predictor_->Init(context);
    return MaceStatus::MACE_SUCCESS;
  }

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override {
    MACE_CHECK_NOTNULL(gemm_predictor_);
    const index_t b = static_cast<index_t>(inputs[0]);
    const index_t m = static_cast<index_t>(inputs[1]);
    const index_t k = static_cast<index_t>(inputs[2]);
    const index_t n = static_cast<index_t>(inputs[3]);
    
    return b * gemm_predictor_->Predict(context, m, k, n);
  }

 private:
  std::shared_ptr<GemmCpuLRLatencyPredictor> gemm_predictor_;
};

class MatMulGpuLatencyPredictor : public LatencyPredictor {
 public:
  MatMulGpuLatencyPredictor() : LatencyPredictor(1) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_MATMUL_LATENCY_PREDICT_H_
