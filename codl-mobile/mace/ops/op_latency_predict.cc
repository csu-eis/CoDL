
#include "mace/ops/op_latency_predict.h"

namespace mace {
namespace ops {

MaceStatus DataSharingLatencyPredictor::Init(LPInitContext *context) {
  const std::vector<double> means = {0};
  const std::vector<double> stds = {1};
  const std::vector<double> coefs = {0};
  const double inter = 0;
  lr_.reset(new utils::LinearRegressionModel("DATA_TRANSFORM", means, stds, coefs, inter));
  
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

double DataSharingLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_UNUSED(context);
  MACE_CHECK_NOTNULL(lr_);
  MACE_CHECK(inputs.size() > 3);
  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t in_size = ih * iw * ic;
  VLOG(1) << "ih " << ih << ", iw " << iw << ", ic " << ic
          << ", in_size " << in_size;
  if (in_size <= 0) {
    return 0;
  }

  std::vector<double> lr_inputs;
  lr_inputs.emplace_back(static_cast<double>(in_size));
  double lat =  lr_->Predict(lr_inputs);
  return (lat < 0) ? 0 : lat;
}

MaceStatus SyncLatencyPredictor::Init(LPInitContext *context) {
  MACE_UNUSED(context);
  c_.reset(new utils::ConstModel("SYNC", 1));
  return MaceStatus::MACE_SUCCESS;
}

double SyncLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_UNUSED(context);
  MACE_CHECK_NOTNULL(c_);
  MACE_CHECK(inputs.size() > 3);
  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t in_size = ih * iw * ic;
  if (in_size > 0) {
    return c_->Predict();
  } else {
    return 0;
  }
}

}  // namespace ops
}  // namespace mace
