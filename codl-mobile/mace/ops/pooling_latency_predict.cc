
#include "mace/ops/pooling_latency_predict.h"

namespace mace {
namespace ops {

MaceStatus PoolingCpuLatencyPredictor::Init(LPInitContext *context) {
  const std::vector<double> means = {0};
  const std::vector<double> stds = {1};
  const std::vector<double> coefs = {0};
  const double inter = 0;
  lr_.reset(new utils::LinearRegressionModel("POOLING_CPU", means, stds, coefs, inter));
  
  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  if (codl_config->is_loaded()) {
    const index_t cur_idx = context->cur_idx();
    context->set_cur_idx(cur_idx + model_count_);
    std::string path = MakeString(codl_config->predict_model_path(), "/",
                                  codl_config->predict_model_filenames()[cur_idx]);
    lr_->BuildFromJson(path.c_str());
  }

  return MaceStatus::MACE_SUCCESS;
}

double PoolingCpuLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_UNUSED(context);
  MACE_CHECK_NOTNULL(lr_);
  MACE_CHECK(inputs.size() == 11);
  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t params = ih * iw * ic;
  if (params <= 0) {
    return 0;
  }

  std::vector<double> lr_inputs;
  lr_inputs.emplace_back(static_cast<double>(params));
  return lr_->Predict(lr_inputs);
}

MaceStatus PoolingGpuLatencyPredictor::Init(LPInitContext *context) {
  const std::vector<double> means = {0};
  const std::vector<double> stds = {1};
  const std::vector<double> coefs = {0};
  const double inter = 0;
  lr_.reset(new utils::LinearRegressionModel("POOLING_GPU", means, stds, coefs, inter));
  
  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  if (codl_config->is_loaded()) {
    const index_t cur_idx = context->cur_idx();
    context->set_cur_idx(cur_idx + model_count_);
    std::string path = MakeString(codl_config->predict_model_path(), "/",
                                  codl_config->predict_model_filenames()[cur_idx]);
    lr_->BuildFromJson(path.c_str());
  }

  return MaceStatus::MACE_SUCCESS;
}

double PoolingGpuLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_UNUSED(context);
  MACE_CHECK_NOTNULL(lr_);
  MACE_CHECK(inputs.size() == 11);
  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t params = ih * iw * ic;
  if (params <= 0) {
    return 0;
  }

  std::vector<double> lr_inputs;
  lr_inputs.emplace_back(static_cast<double>(params));
  return lr_->Predict(lr_inputs);
}

MaceStatus PoolingFLOPsLatencyPredictor::Init(LPInitContext *context) {
  const std::vector<double> means = {0, 0};
  const std::vector<double> stds = {1, 1};
  const std::vector<double> coefs = {0, 0};
  const double inter = 0;
  lr_.reset(new utils::LinearRegressionModel("POOLING_FLOPS", means, stds, coefs, inter));
  
  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  if (codl_config->is_loaded()) {
    const index_t cur_idx = context->cur_idx();
    context->set_cur_idx(cur_idx + model_count_);
    std::string path = MakeString(codl_config->predict_model_path(), "/",
                                  codl_config->predict_model_filenames()[cur_idx]);
    lr_->BuildFromJson(path.c_str());
  }

  return MaceStatus::MACE_SUCCESS;
}

double PoolingFLOPsLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_UNUSED(context);
  MACE_CHECK_NOTNULL(lr_);
  MACE_CHECK(inputs.size() == 11);
  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t oc = static_cast<index_t>(inputs[3]);
  const index_t kh = static_cast<index_t>(inputs[4]);
  const index_t kw = static_cast<index_t>(inputs[5]);
  const int sh = static_cast<int>(inputs[6]);
  const int sw = static_cast<int>(inputs[7]);
  
  // Calculate features.
  const double oh = (ih - kh) / sh + 1;
  const double ow = (iw - kw) / sw + 1;
  const double in_size = ih * iw * ic;
  const double out_size = oh * ow * oc;
  if (in_size <= 0 || out_size <= 0) {
    return 0;
  }

  std::vector<double> lr_inputs;
  lr_inputs.emplace_back(static_cast<double>(in_size));
  lr_inputs.emplace_back(static_cast<double>(out_size));
  
  // Predict OP latency.
  const double lat_op = lr_->Predict(lr_inputs);
  
  return lat_op;
}

}  // namespace ops
}  // namespace mace
