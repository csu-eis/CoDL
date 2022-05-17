
#include "mace/ops/arm/fp32/gemv.h"
#include "mace/ops/gemv_latency_predict.h"

namespace mace {
namespace ops {

MaceStatus GemvCpuLatencyPredictor::Init(LPInitContext *context) {
  lr_.reset(new utils::LinearRegressionModel("CPU_GEMV", {0, 0}, {1, 1}, {0, 0}, 0));
  
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

double GemvCpuLatencyPredictor::Predict(OpContext *context,
                                        const index_t h,
                                        const index_t w) {
  MACE_CHECK_NOTNULL(lr_);
  if ((h * w) == 0) {
    return 0;
  }
  int hb, wb;
  arm::fp32::Gemv::CalcBlockCount(h, w, &hb, &wb);

  int tile_size, max_thread_tile_count;
  int max_thread_hb;
  utils::ThreadPool &thread_pool
      = context->device()->cpu_runtime()->thread_pool();
  thread_pool.Calculate2DTileCount(0, 1, 1, 0, hb, 1,
                                   nullptr, &tile_size, &max_thread_tile_count);
  max_thread_hb = tile_size * max_thread_tile_count;

  std::vector<std::pair<std::string, double>> features;
  features.emplace_back("H", h);
  features.emplace_back("W", w);

  const double t_comp = lr_->Predict(features);
  const double lat_comp = t_comp * max_thread_hb * wb;
  const double lat_op = lat_comp;
  return lat_op;
}

}  // namespace ops
}  // namespace mace
