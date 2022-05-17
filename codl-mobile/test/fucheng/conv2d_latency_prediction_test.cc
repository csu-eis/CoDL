
#include <vector>
#include "mace/core/latency_predictor.h"
#include "mace/utils/linear_regression_model.h"

namespace mace {

class Conv2dDirectLatencyPredicotr : public LatencyPredictor {
 public:
  MaceStatus Init() override {
    std::vector<double> coefs = {1, 1, 1, 1};
    double inter = 1;
    lr_.reset(new utils::LinearRegressionModel("Conv2dDirect", coefs, inter));
    return MaceStatus::MACE_SUCCESS;
  }

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override {
    MACE_UNUSED(context);
    const index_t ih = static_cast<index_t>(inputs[0]);
    const index_t iw = static_cast<index_t>(inputs[1]);
    const index_t ic = static_cast<index_t>(inputs[2]);
    const index_t oc = static_cast<index_t>(inputs[3]);

    std::vector<double> lr_inputs;
    lr_inputs.emplace_back(static_cast<double>(ih));
    lr_inputs.emplace_back(static_cast<double>(iw));
    lr_inputs.emplace_back(static_cast<double>(ic));
    lr_inputs.emplace_back(static_cast<double>(oc));
    return lr_->Predict(lr_inputs);
  }

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

class Conv2dGemmLatencyPredicotr : public LatencyPredictor {
 public:
  MaceStatus Init() override {
    std::vector<double> coefs = {1, 1, 1};
    double inter = 1;
    lr_.reset(new utils::LinearRegressionModel("Conv2dGemm", coefs, inter));
    return MaceStatus::MACE_SUCCESS;
  }

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override {
    MACE_UNUSED(context);
    const index_t m = static_cast<index_t>(inputs[0]);
    const index_t k = static_cast<index_t>(inputs[1]);
    const index_t n = static_cast<index_t>(inputs[2]);

    std::vector<double> lr_inputs;
    lr_inputs.emplace_back(static_cast<double>(m));
    lr_inputs.emplace_back(static_cast<double>(k));
    lr_inputs.emplace_back(static_cast<double>(n));
    return lr_->Predict(lr_inputs);
  }

 private:
  std::shared_ptr<utils::LinearRegressionModel> lr_;
};

class Conv2dWinogradLatencyPredicotr : public LatencyPredictor {
 public:
  MaceStatus Init() override {
    std::vector<double> coefs = {1, 1, 1};
    double inter = 1;
    tr_in_lr_.reset(new utils::LinearRegressionModel(
        "Conv2dWinogradTransformInput", coefs, inter));
    tr_out_lr_.reset(new utils::LinearRegressionModel(
        "Conv2dWinogradTransformOutput", coefs, inter));
    gemm_predictor_.reset(new Conv2dGemmLatencyPredicotr());
    gemm_predictor_->Init();
    return MaceStatus::MACE_SUCCESS;
  }

  double Predict(OpContext *context,
                 const std::vector<double> &inputs) override {
    MACE_UNUSED(context);
    const index_t ih = static_cast<index_t>(inputs[0]);
    const index_t iw = static_cast<index_t>(inputs[1]);
    const index_t ic = static_cast<index_t>(inputs[2]);
    const index_t oh = static_cast<index_t>(inputs[3]);
    const index_t ow = static_cast<index_t>(inputs[4]);
    const index_t oc = static_cast<index_t>(inputs[5]);
    const index_t ot_size = static_cast<index_t>(inputs[6]);

    std::vector<double> tr_in_lr_inputs;
    tr_in_lr_inputs.emplace_back(static_cast<double>(ih));
    tr_in_lr_inputs.emplace_back(static_cast<double>(iw));
    tr_in_lr_inputs.emplace_back(static_cast<double>(ic));
    double t_tr_in =  tr_in_lr_->Predict(tr_in_lr_inputs);

    std::vector<double> tr_out_lr_inputs;
    tr_out_lr_inputs.emplace_back(static_cast<double>(oh));
    tr_out_lr_inputs.emplace_back(static_cast<double>(ow));
    tr_out_lr_inputs.emplace_back(static_cast<double>(oc));
    double t_tr_out =  tr_out_lr_->Predict(tr_out_lr_inputs);

    const index_t m = oc;
    const index_t n = (oh / ot_size) * (ow / ot_size);
    const index_t k = ic;

    std::vector<double> gemm_inputs;
    gemm_inputs.emplace_back(static_cast<double>(m));
    gemm_inputs.emplace_back(static_cast<double>(k));
    gemm_inputs.emplace_back(static_cast<double>(n));
    double t_gemm = gemm_predictor_->Predict(context, gemm_inputs);

    return t_tr_in + t_tr_out + t_gemm;
  }

 private:
  std::shared_ptr<utils::LinearRegressionModel> tr_in_lr_;
  std::shared_ptr<utils::LinearRegressionModel> tr_out_lr_;
  std::shared_ptr<Conv2dGemmLatencyPredicotr> gemm_predictor_;
};

}  // namespace mace

using namespace mace;

constexpr int kRounds = 1000000;

int RunConv2dDirectPredictorOnceTest(
    Conv2dDirectLatencyPredicotr *predictor,
    const std::vector<double> &inputs,
    const index_t k,
    const index_t s) {
  if (k == 3 && s == 1) {
    predictor->Predict(nullptr, inputs);
  } else if (k == 3 && s == 2) {
    predictor->Predict(nullptr, inputs);
  } else if (k == 5 && s == 1) {
    predictor->Predict(nullptr, inputs);
  } else if (k == 5 && s == 2) {
    predictor->Predict(nullptr, inputs);
  } else if (k == 7 && s == 1) {
    predictor->Predict(nullptr, inputs);
  } else if (k == 7 && s == 2) {
    predictor->Predict(nullptr, inputs);
  } else if (k == 9 && s == 1) {
    predictor->Predict(nullptr, inputs);
  } else if (k == 9 && s == 2) {
    predictor->Predict(nullptr, inputs);
  } else {
    predictor->Predict(nullptr, inputs);
  }
  return 0;
}

int RunConv2dDirectPredictorTest() {
  const index_t ih = 224;
  const index_t iw = 224;
  const index_t ic = 64;
  const index_t oc = 64;
  const index_t k = 3;
  const index_t s = 1;

  const std::vector<double> inputs = {ih, iw, ic, oc};
  
  Conv2dDirectLatencyPredicotr predictor;
  predictor.Init();

  const int64_t t0 = NowMicros();
  for (int i = 0; i < kRounds; i ++) {
    RunConv2dDirectPredictorOnceTest(&predictor, inputs, k, s);
  }
  LOG(INFO) << "Conv2dDirect: " << (NowMicros() - t0) / 1000.0 / kRounds << " ms";

  return 0;
}

int RunConv2dGemmPredictorTest() {
  const index_t m = 64;
  const index_t k = 64;
  const index_t n = 224 * 224;

  const std::vector<double> inputs = {m, k, n};
  
  Conv2dGemmLatencyPredicotr predictor;
  predictor.Init();

  const int64_t t0 = NowMicros();
  for (int i = 0; i < kRounds; i ++) {
    predictor.Predict(nullptr, inputs);
  }
  LOG(INFO) << "Conv2dGemm: " << (NowMicros() - t0) / 1000.0 / kRounds << " ms";

  return 0;
}

int RunConv2dWinogradPredictorTest() {
  const index_t ih = 224;
  const index_t iw = 224;
  const index_t ic = 64;
  const index_t oh = 224;
  const index_t ow = 224;
  const index_t oc = 64;
  const index_t ot_size = 6;

  const std::vector<double> inputs = {ih, iw, ic, oh, ow, oc, ot_size};
  
  Conv2dWinogradLatencyPredicotr predictor;
  predictor.Init();

  const int64_t t0 = NowMicros();
  for (int i = 0; i < kRounds; i ++) {
    predictor.Predict(nullptr, inputs);
  }
  LOG(INFO) << "Conv2dWinograd: " << (NowMicros() - t0) / 1000.0 / kRounds << " ms";

  return 0;
}

int main() {
  RunConv2dDirectPredictorTest();
  RunConv2dGemmPredictorTest();
  RunConv2dWinogradPredictorTest();
  return 0;
}
