
#include "mace/core/runtime/opencl/opencl_util.h"
#include "mace/ops/fully_connected_latency_predict.h"
#include "mace/ops/opencl/image/fully_connected.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {

MaceStatus FullyConnectedCpuLatencyPredictor::Init(LPInitContext *context) {
  cpu_gemv_predictor_.reset(new GemvCpuLatencyPredictor());
  cpu_gemv_predictor_->Init(context);
  return MaceStatus::MACE_SUCCESS;
}

double FullyConnectedCpuLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_CHECK(inputs.size() == 6);
  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t oc = static_cast<index_t>(inputs[3]);

  const index_t gemv_h = oc;
  const index_t gemv_w = ih * iw * ic;
  if (gemv_h == 0 || gemv_w == 0) {
    return 0;
  }

  return cpu_gemv_predictor_->Predict(context, gemv_h, gemv_w);
}

MaceStatus FullyConnectedGpuLatencyPredictor::Init(LPInitContext *context) {
  const std::vector<double> means = {0, 0, 0, 0};
  const std::vector<double> stds = {1, 1, 1, 1};
  const std::vector<double> coefs = {0, 0, 0, 0};
  const double inter = 0;
  lr_.reset(new utils::LinearRegressionModel("FULLY_CONNECTED_GPU", means, stds, coefs, inter));
  
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

double FullyConnectedGpuLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_CHECK(inputs.size() == 6);
  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t oc = static_cast<index_t>(inputs[3]);
  if ((ih * iw * ic * oc) == 0) {
    return 0;
  }

  int iwb, iwb_size, icb, ocb;
  int num_warps;
  std::vector<uint32_t> gws;
  std::vector<uint32_t> lws;

  icb = RoundUpDiv4(ic);
  ocb = RoundUpDiv4(oc);

  auto opencl_runtime = context->device()->gpu_runtime()->opencl_runtime();
  const uint32_t kwg_size = opencl_runtime->kwg_size();
  MACE_CHECK(kwg_size > 0);
  gws = opencl::image::FullyConnectedGlobalWS(opencl_runtime, 1, ocb,
                                              opencl_runtime->warp_size());
  lws = opencl::image::FullyConnectedLocalWS(gws.data(), kwg_size);
  OpenCLUtil::CalcWarpNumber(opencl_runtime, gws.data(), lws.data(), &num_warps);

  iwb = gws[1];
  iwb_size = RoundUpDiv(static_cast<int>(iw), iwb);

  std::vector<std::pair<std::string, double>> features;
  features.emplace_back("IH", ih);
  features.emplace_back("IW", iw);
  features.emplace_back("ICB", icb);
  features.emplace_back("OCB", ocb);

  const double t_warp_blk = lr_->Predict(features);
  index_t num_blks = ih * iwb_size * icb;
  if (iwb_size == 1) {
    num_blks = ih * iw * icb;
  }
  const double t_warp = t_warp_blk * num_blks;
  const double lat_op = num_warps * t_warp;

  VLOG(1) << "ih " << ih << ", iw " << iw << ", icb " << icb << ", ocb " << ocb;
  VLOG(1) << "iwb " << iwb << ", iwb_size " << iwb_size;
  VLOG(1) << "num_blks " << num_blks << ", num_warps " << num_warps;
  VLOG(1) << "t_warp_blk " << t_warp_blk << ", t_warp " << t_warp << ", lat_op " << lat_op;

  return lat_op;
}

MaceStatus FullyConnectedFLOPsLatencyPredictor::Init(LPInitContext *context) {
#if 0
  const std::vector<double> means = {0};
  const std::vector<double> stds = {1};
  const std::vector<double> coefs = {0};
#endif
  const std::vector<double> means = {0, 0};
  const std::vector<double> stds = {1, 1};
  const std::vector<double> coefs = {0, 0};
  const double inter = 0;
  lr_.reset(new utils::LinearRegressionModel("FC_FLOPS", means, stds, coefs, inter));

  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  if (codl_config->is_loaded()) {
    const index_t cur_idx = context->cur_idx();
    context->set_cur_idx(cur_idx + model_count_);
    std::string path;
    path = MakeString(codl_config->predict_model_path(), "/",
                      codl_config->predict_model_filenames()[cur_idx]);
    lr_->BuildFromJson(path.c_str());
  }

  return MaceStatus::MACE_SUCCESS;
}

double FullyConnectedFLOPsLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_UNUSED(context);
  MACE_CHECK(inputs.size() == 6);

  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t oc = static_cast<index_t>(inputs[3]);
  const index_t kh = static_cast<index_t>(inputs[4]);
  const index_t kw = static_cast<index_t>(inputs[5]);

  const index_t input_size = ih * iw * ic;
  const index_t filter_size = oc * ic * kh * kw;
  if (input_size <= 0 || filter_size <= 0) {
    return 0;
  }

  const std::vector<index_t> input_shape = {1, ih, iw, ic};
  const std::vector<index_t> kernel_shape = {oc, ic, kh, kw};
  const std::vector<index_t> output_shape = {1, 1, 1, oc};
  
#if 0
  // Calculate FLOPs.
  const double flops = MaceFLOPsStatistics::Compute(
      "FullyConnected", kernel_shape, output_shape);

  std::vector<double> lr_inputs;
  lr_inputs.emplace_back(static_cast<double>(flops));
#endif

  // Calculate features.
  const double in_size = ih * iw * ic;
  const double out_size = oc;

  std::vector<double> lr_inputs;
  lr_inputs.emplace_back(static_cast<double>(in_size));
  lr_inputs.emplace_back(static_cast<double>(out_size));

  // Predict OP latency.
  const double lat_op = lr_->Predict(lr_inputs);

  return lat_op;
}

}  // namespace ops
}  // namespace mace
