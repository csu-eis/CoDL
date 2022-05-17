
#ifndef MACE_OPS_CONV_2D_LATENCY_PREDICT_H_
#define MACE_OPS_CONV_2D_LATENCY_PREDICT_H_

#include "mace/core/latency_predictor.h"
#include "mace/ops/gemm_latency_predict.h"
#include "mace/utils/const_model.h"
#include "mace/utils/linear_regression_model.h"
#include "mace/utils/random_forest_model.h"

#define Conv2dCpuLatencyPredictor Conv2dCpuLRLatencyPredictor
#define Conv2dGpuLatencyPredictor Conv2dGpuLRLatencyPredictor

namespace mace {
namespace ops {

#if 0

class Conv2dCpuGemmLatencyPredictorBase : public LatencyPredictor {
 public:
  virtual double Predict(OpContext *context,
                         const index_t m,
                         const index_t k,
                         const index_t n) = 0;
  
  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override {
    MACE_CHECK(inputs.size() == 11);

    const index_t ih = static_cast<index_t>(inputs[0]);
    const index_t iw = static_cast<index_t>(inputs[1]);
    const index_t ic = static_cast<index_t>(inputs[2]);
    const index_t oc = static_cast<index_t>(inputs[3]);
    const index_t kh = static_cast<index_t>(inputs[4]);
    const index_t kw = static_cast<index_t>(inputs[5]);
    //const int sh = static_cast<int>(inputs[6]);
    //const int sw = static_cast<int>(inputs[7]);
    //const int dh = static_cast<int>(inputs[8]);
    //const int dw = static_cast<int>(inputs[9]);
    //const int padding_type = static_cast<int>(inputs[10]);

    const index_t input_size = ih * iw * ic;
    const index_t filter_size = oc * ic * kh * kw;
    if (input_size <= 0 || filter_size <= 0) {
      return 0;
    }

    const index_t m = oc;
    const index_t k = ic;
    const index_t n = ih * iw;

    return Predict(context, m, k, n);
  }
};

#endif

class Conv2dCpuGemmLatencyPredictor : public LatencyPredictor {
 public:
  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override {
    MACE_CHECK(inputs.size() == 11);

    const index_t ih = static_cast<index_t>(inputs[0]);
    const index_t iw = static_cast<index_t>(inputs[1]);
    const index_t ic = static_cast<index_t>(inputs[2]);
    const index_t oc = static_cast<index_t>(inputs[3]);
    const index_t kh = static_cast<index_t>(inputs[4]);
    const index_t kw = static_cast<index_t>(inputs[5]);
    //const int sh = static_cast<int>(inputs[6]);
    //const int sw = static_cast<int>(inputs[7]);
    //const int dh = static_cast<int>(inputs[8]);
    //const int dw = static_cast<int>(inputs[9]);
    //const int padding_type = static_cast<int>(inputs[10]);

    const index_t input_size = ih * iw * ic;
    const index_t filter_size = oc * ic * kh * kw;
    if (input_size <= 0 || filter_size <= 0) {
      return 0;
    }

    const index_t m = oc;
    const index_t k = ic;
    const index_t n = ih * iw;

    return gemm_predictor_->Predict(context, m, k, n);
  }

  double Predict(OpContext *context,
                 const index_t m,
                 const index_t k,
                 const index_t n) {
    return gemm_predictor_->Predict(context, m, k, n);
  }

 protected:
  std::shared_ptr<GemmCpuLatencyPredictor> gemm_predictor_;
};

/**
 * Random Forest based Models
 */

#if 0

class Conv2dCpuDirectRFLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init() override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::RandomForestModel> rf_;
};

class Conv2dCpuGemmRFLatencyPredictor : public Conv2dCpuGemmLatencyPredictor {
 public:
  MaceStatus Init() override {
    MACE_NOT_IMPLEMENTED;
    return MaceStatus::MACE_SUCCESS;
  }
};

class Conv2dCpuWinogradRFLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init() override;

  MaceStatus Init(
      std::shared_ptr<Conv2dCpuGemmRFLatencyPredictor> gemm_predictor);

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

  void set_gemm_predictor(
      std::shared_ptr<Conv2dCpuGemmRFLatencyPredictor> predictor) {
    gemm_predictor_ = predictor;
  }

 private:
  std::shared_ptr<utils::RandomForestModel> pad_rf_;
  std::shared_ptr<utils::RandomForestModel> unpad_rf_;
  std::shared_ptr<utils::RandomForestModel> trans_in_rf_;
  std::shared_ptr<utils::RandomForestModel> trans_out_rf_;
  std::shared_ptr<Conv2dCpuGemmRFLatencyPredictor> gemm_predictor_;
};

class Conv2dGpuDirectRFLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init() override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::RandomForestModel> rf_;
};

class Conv2dCpuRFLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init() override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<Conv2dCpuDirectRFLatencyPredictor> cpu_direct_predictor_;
  std::shared_ptr<Conv2dCpuGemmRFLatencyPredictor> cpu_gemm_predictor_;
  std::shared_ptr<Conv2dCpuWinogradRFLatencyPredictor> cpu_winograd_predictor_;
};

class Conv2dGpuRFLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init() override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<Conv2dGpuDirectRFLatencyPredictor> gpu_direct_predictor_;
};

#endif

/**
 * Linear Regression baesd Models
 */

constexpr int kConv2dCpuDirectKernelCount = 10;
constexpr int kConv2dCpuGemmKernelCount = 3;
constexpr int kConv2dCpuWinogradKernelCount = 5;
constexpr int kConv2dGpuDirectKernelCount = 6;

class Conv2dCpuDirectLRLatencyPredictor : public LatencyPredictor {
 public:
  Conv2dCpuDirectLRLatencyPredictor() : LatencyPredictor(10) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::vector<std::shared_ptr<utils::LinearRegressionModel>> lr_models_;
};

class Conv2dCpuGemmLRLatencyPredictor : public Conv2dCpuGemmLatencyPredictor {
 public:
  MaceStatus Init(LPInitContext *context) override {
    gemm_predictor_.reset(new GemmCpuLRLatencyPredictor());
    gemm_predictor_->Init(context);
    return MaceStatus::MACE_SUCCESS;
  }
};

class Conv2dCpuWinogradLRLatencyPredictor : public LatencyPredictor {
 public:
  Conv2dCpuWinogradLRLatencyPredictor() : LatencyPredictor(5) {}

  MaceStatus Init(LPInitContext *context) override;

  MaceStatus Init(
      LPInitContext *context,
      std::shared_ptr<Conv2dCpuGemmLRLatencyPredictor> gemm_predictor);

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

  void set_gemm_predictor(
      std::shared_ptr<Conv2dCpuGemmLRLatencyPredictor> predictor) {
    gemm_predictor_ = predictor;
  }

 private:
  std::shared_ptr<utils::LinearRegressionModel> pad_lr_;
  std::shared_ptr<utils::LinearRegressionModel> unpad_lr_;
  std::shared_ptr<utils::LinearRegressionModel> trans_in_lr_;
  std::shared_ptr<utils::LinearRegressionModel> trans_out_lr_;
  std::shared_ptr<utils::LinearRegressionModel> comp_lr_;
  std::shared_ptr<Conv2dCpuGemmLRLatencyPredictor> gemm_predictor_;
};

class Conv2dGpuDirectLRLatencyPredictor : public LatencyPredictor {
 public:
  Conv2dGpuDirectLRLatencyPredictor() : LatencyPredictor(10) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::vector<std::shared_ptr<utils::LinearRegressionModel>> lr_models_;
};

class Conv2dCpuLRLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<Conv2dCpuDirectLRLatencyPredictor> cpu_direct_predictor_;
  std::shared_ptr<Conv2dCpuGemmLRLatencyPredictor> cpu_gemm_predictor_;
  std::shared_ptr<Conv2dCpuWinogradLRLatencyPredictor> cpu_winograd_predictor_;
};

class Conv2dGpuLRLatencyPredictor : public LatencyPredictor {
 public:
  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<Conv2dGpuDirectLRLatencyPredictor> gpu_direct_predictor_;
};

/**
 * Mulayer Models
 */

class Conv2dFLOPsLatencyPredictor : public LatencyPredictor {
 public:
  Conv2dFLOPsLatencyPredictor() : LatencyPredictor(1) {}

  MaceStatus Init(LPInitContext *context) override;

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override;

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CONV_2D_LATENCY_PREDICT_H_
