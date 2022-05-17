
#include "mace/ops/matmul_latency_predict.h"
#include "mace/ops/opencl/image/matmul.h"

namespace mace {
namespace ops {

MaceStatus MatMulGpuLatencyPredictor::Init(LPInitContext *context) {
  const std::vector<double> means = {0, 0, 0};
  const std::vector<double> stds = {1, 1, 1};
  const std::vector<double> coefs = {0, 0, 0};
  const double inter = 0;
  lr_.reset(new utils::LinearRegressionModel("MATMUL_GPU", means, stds, coefs, inter));
  
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

double MatMulGpuLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_CHECK_NOTNULL(lr_);
  const index_t b = static_cast<index_t>(inputs[0]);
  const index_t m = static_cast<index_t>(inputs[1]);
  const index_t k = static_cast<index_t>(inputs[2]);
  const index_t n = static_cast<index_t>(inputs[3]);
  
  const int m_blocks = RoundUpDiv4(static_cast<int>(m));
  const int k_blocks = RoundUpDiv4(static_cast<int>(k));
  const int n_blocks = RoundUpDiv4(static_cast<int>(n));
  
  int num_warps;
  uint32_t gws[2];
  std::vector<uint32_t> lws;
  
  auto opencl_runtime = context->device()->gpu_runtime()->opencl_runtime();
  const uint32_t kwg_size = opencl_runtime->kwg_size();
  MACE_CHECK(kwg_size > 0);
  opencl::image::MatMulGlobalWS(b, m_blocks, n_blocks, gws);
  lws = opencl::image::MatMulLocalWS(kwg_size);
  OpenCLUtil::CalcWarpNumber(opencl_runtime, gws, lws.data(), &num_warps);
  
  std::vector<std::pair<std::string, double>> features;
  features.emplace_back("M", m);
  features.emplace_back("K", k);
  features.emplace_back("N", n);

  const double t_warp_blk = lr_->Predict(features);
  const double t_warp = t_warp_blk * k_blocks;
  const double lat_op = num_warps * t_warp;

#if 1
  LOG(INFO) << "b " << b << ", m " << m << ", k " << k << ", n " << n;
  LOG(INFO) << "t_warp_blk " << t_warp_blk << ", k_blocks " << k_blocks
            << ", t_warp " << t_warp << ", num_warps " << num_warps
            << ", lat_op " << lat_op;
#endif
  
  return lat_op;
}

}  // namespace ops
}  // namespace mace
