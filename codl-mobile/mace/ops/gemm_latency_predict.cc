
#include "mace/ops/gemm_latency_predict.h"

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/gemm.h"
#endif

namespace mace {
namespace ops {

MaceStatus GemmCpuLRLatencyPredictor::Init(LPInitContext *context) {
  const std::vector<double> pack_means = {0, 0};
  const std::vector<double> pack_stds = {1, 1};
  const std::vector<double> pack_coefs = {0, 0};
  const std::vector<double> comp_means = {0, 0, 0};
  const std::vector<double> comp_stds = {0, 0, 0};
  const std::vector<double> comp_coefs = {0, 0, 0};
  const double inter = 0;
  pack_lhs_lr_.reset(new utils::LinearRegressionModel(
      "GEMM_PACK_LHS", pack_means, pack_stds, pack_coefs, inter));
  pack_rhs_lr_.reset(new utils::LinearRegressionModel(
      "GEMM_PACK_RHS", pack_means, pack_stds, pack_coefs, inter));
  comp_lr_.reset(new utils::LinearRegressionModel(
      "GEMM_COMPUTE", comp_means, comp_stds, comp_coefs, inter));

  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  if (codl_config->is_loaded()) {
    const index_t cur_idx = context->cur_idx();
    context->set_cur_idx(cur_idx + model_count_);
    std::string path;
    path = MakeString(codl_config->predict_model_path(), "/",
                      codl_config->predict_model_filenames()[cur_idx]);
    pack_lhs_lr_->BuildFromJson(path.c_str());
    path = MakeString(codl_config->predict_model_path(), "/",
                      codl_config->predict_model_filenames()[cur_idx + 1]);
    pack_rhs_lr_->BuildFromJson(path.c_str());
    path = MakeString(codl_config->predict_model_path(), "/",
                      codl_config->predict_model_filenames()[cur_idx + 2]);
    comp_lr_->BuildFromJson(path.c_str());
  }

  return MaceStatus::MACE_SUCCESS;
}

double GemmCpuLRLatencyPredictor::Predict(
    OpContext *context,
    const index_t m,
    const index_t k,
    const index_t n) {
  MACE_CHECK_NOTNULL(pack_lhs_lr_);
  MACE_CHECK_NOTNULL(pack_rhs_lr_);
  MACE_CHECK_NOTNULL(comp_lr_);
  int mb, kb, nb;
  arm::fp32::Gemm::CalcBlockCount(m, n, k, &mb, &nb, &kb);

  int tile_size, max_thread_tile_count;
  int max_thread_mb, max_thread_nb;
  utils::ThreadPool &thread_pool
      = context->device()->cpu_runtime()->thread_pool();
  thread_pool.Calculate1DTileCount(0, mb, 1,
                                   &tile_size, &max_thread_tile_count);
  max_thread_mb = tile_size * max_thread_tile_count;
  thread_pool.Calculate1DTileCount(0, nb, 1,
                                   &tile_size, &max_thread_tile_count);
  max_thread_nb = tile_size * max_thread_tile_count;

  std::vector<std::pair<std::string, double>> pack_lhs_features;
  std::vector<std::pair<std::string, double>> pack_rhs_features;
  std::vector<std::pair<std::string, double>> comp_features;
  pack_lhs_features.emplace_back("M", m);
  pack_lhs_features.emplace_back("K", k);
  pack_rhs_features.emplace_back("K", k);
  pack_rhs_features.emplace_back("N", n);
  comp_features.emplace_back("M", m);
  comp_features.emplace_back("K", k);
  comp_features.emplace_back("N", n);
  const double t_pack_lhs = pack_lhs_lr_->Predict(pack_lhs_features);
  const double t_pack_rhs = pack_rhs_lr_->Predict(pack_rhs_features);
  const double t_comp = comp_lr_->Predict(comp_features);

  const double lat_pack_lhs = t_pack_lhs * max_thread_mb * kb;
  const double lat_pack_rhs = t_pack_rhs * max_thread_nb * kb;
  const double lat_comp = t_comp * max_thread_mb * kb * nb;

  const double lat_op = lat_pack_lhs + lat_pack_rhs + lat_comp;

#if 1
  VLOG(1) << "m " << m << ", k " << k << ", n " << n;
  VLOG(1) << "mb " << mb << ", kb " << kb << ", nb " << nb;
  VLOG(1) << "t_pack_lhs " << t_pack_lhs
          << ", t_pack_rhs " << t_pack_rhs
          << ", t_comp " << t_comp;
  VLOG(1) << "max_thread_mb " << max_thread_mb
          << ", max_thread_nb " << max_thread_nb
          << ", kb " << kb
          << ", nb " << nb;
  VLOG(1) << "lat_pack_lhs " << lat_pack_lhs
          << ", lat_pack_rhs " << lat_pack_rhs
          << ", lat_comp " << lat_comp;
#endif

  return lat_op;
}

}  // namespace ops
}  // namespace mace
