
#if 0

#include "mace/core/runtime/opencl/opencl_util.h"
#include "mace/ops/conv_2d_latency_predict.h"

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/conv_2d.h"
#endif

namespace mace {
namespace ops {

MaceStatus Conv2dCpuDirectRFLatencyPredictor::Init() {
  rf_.reset(new utils::RandomForestModel("CPU_DIRECT"));

  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  std::string path = MakeString(codl_config->predict_model_path(), "/",
                                codl_config->predict_model_filenames()[1]);
  rf_->BuildFromJson(path.c_str());
  return MaceStatus::MACE_SUCCESS;
}

double Conv2dCpuDirectRFLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_CHECK(inputs.size() == 11);

  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t oc = static_cast<index_t>(inputs[3]);
  const index_t kh = static_cast<index_t>(inputs[4]);
  const index_t kw = static_cast<index_t>(inputs[5]);
  const int sh = static_cast<int>(inputs[6]);
  const int sw = static_cast<int>(inputs[7]);
  const int dh = static_cast<int>(inputs[8]);
  const int dw = static_cast<int>(inputs[9]);
  const int padding_type = static_cast<int>(inputs[10]);

  const index_t input_size = ih * iw * ic;
  const index_t filter_size = oc * ic * kh * kw;
  if (input_size <= 0 || filter_size <= 0) {
    return 0;
  }

  const std::vector<index_t> input_shape = {1, ih, iw, ic};
  const std::vector<index_t> kernel_shape = {oc, ic, kh, kw};
  const int dilations[2] = {dh, dw};
  const int strides[2] = {sh, sw};
  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  CalcNHWCPaddingAndOutputSize(
      input_shape.data(), kernel_shape.data(), dilations, strides,
      static_cast<Padding>(padding_type), output_shape.data(), paddings.data());
  
  const double flops = MaceFLOPsStatistics::Compute(
      "Conv2D", kernel_shape, output_shape);

  const int kChannelPerItem = 4;
  int tile_size;
  int max_thread_tiles;
  utils::ThreadPool &thread_pool
      = context->device()->cpu_runtime()->thread_pool();
  thread_pool.Calculate2DTileCount(0, 1, 1, 0, oc, kChannelPerItem,
                                   nullptr, &tile_size, &max_thread_tiles);

  const double flops_per_tile = tile_size * kChannelPerItem * flops / oc;

  std::vector<std::pair<std::string, double>> features;
  features.emplace_back("IH", ih);
  features.emplace_back("IW", iw);
  features.emplace_back("IC", ic);
  features.emplace_back("OC", oc);
  features.emplace_back("KH", kh);
  features.emplace_back("KW", kw);
  features.emplace_back("SH", sh);
  features.emplace_back("SW", sw);
  const double t_flops = rf_->Predict(features);

  const double lat_op = max_thread_tiles * flops_per_tile * t_flops;

#if 0
  LOG(INFO) << "flops " << flops
            << ", tile_size " << tile_size
            << ", max_thread_tiles " << max_thread_tiles
            << ", flops_per_tile " << flops_per_tile
            << ", t_flops " << t_flops;
#endif

  return lat_op;
}

#if 0

MaceStatus Conv2dCpuGemmRFLatencyPredictor::Init() {
  pack_lhs_rf_.reset(new utils::RandomForestModel("GEMM_PACK_LHS"));
  pack_rhs_rf_.reset(new utils::RandomForestModel("GEMM_PACK_RHS"));
  comp_rf_.reset(new utils::RandomForestModel("GEMM_COMPUTE"));

  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  std::string path;
  path = MakeString(codl_config->predict_model_path(), "/",
                    codl_config->predict_model_filenames()[2]);
  pack_lhs_rf_->BuildFromJson(path.c_str());
  path = MakeString(codl_config->predict_model_path(), "/",
                    codl_config->predict_model_filenames()[3]);
  pack_rhs_rf_->BuildFromJson(path.c_str());
  path = MakeString(codl_config->predict_model_path(), "/",
                    codl_config->predict_model_filenames()[4]);
  comp_rf_->BuildFromJson(path.c_str());

  return MaceStatus::MACE_SUCCESS;
}

double Conv2dCpuGemmRFLatencyPredictor::Predict(
    OpContext *context,
    const index_t m,
    const index_t k,
    const index_t n) {
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
  const double t_pack_lhs = pack_lhs_rf_->Predict(pack_lhs_features);
  const double t_pack_rhs = pack_rhs_rf_->Predict(pack_rhs_features);
  const double t_comp = comp_rf_->Predict(comp_features);

  const double lat_pack_lhs = t_pack_lhs * max_thread_mb * kb;
  const double lat_pack_rhs = t_pack_rhs * max_thread_nb * kb;
  const double lat_comp = t_comp * max_thread_mb * kb * nb;

  const double lat_op = lat_pack_lhs + lat_pack_rhs + lat_comp;

#if 0
  LOG(INFO) << "m " << m << ", k " << k << ", n " << n;
  LOG(INFO) << "mb " << mb << ", kb " << kb << ", nb " << nb;
  LOG(INFO) << "t_pack_lhs " << t_pack_lhs
            << ", t_pack_rhs " << t_pack_rhs
            << ", t_comp " << t_comp;
  LOG(INFO) << "max_thread_mb " << max_thread_mb
            << ", max_thread_nb " << max_thread_nb
            << ", kb " << kb
            << ", nb " << nb;
  LOG(INFO) << "lat_pack_lhs " << lat_pack_lhs
            << ", lat_pack_rhs " << lat_pack_rhs
            << ", lat_comp " << lat_comp;
#endif

  return lat_op;
}

#endif

MaceStatus Conv2dCpuWinogradRFLatencyPredictor::Init() {
  pad_rf_.reset(new utils::RandomForestModel("WINOGRAD_PAD"));
  trans_in_rf_.reset(new utils::RandomForestModel("WINOGRAD_TRANS_IN"));
  trans_out_rf_.reset(new utils::RandomForestModel("WINOGRAD_TRANS_OUT"));
  unpad_rf_.reset(new utils::RandomForestModel("WINOGRAD_UNPAD"));

  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  std::string path;
  path = MakeString(codl_config->predict_model_path(), "/",
                    codl_config->predict_model_filenames()[5]);
  pad_rf_->BuildFromJson(path.c_str());
  path = MakeString(codl_config->predict_model_path(), "/",
                    codl_config->predict_model_filenames()[6]);
  trans_in_rf_->BuildFromJson(path.c_str());
  path = MakeString(codl_config->predict_model_path(), "/",
                    codl_config->predict_model_filenames()[7]);
  trans_out_rf_->BuildFromJson(path.c_str());
  path = MakeString(codl_config->predict_model_path(), "/",
                    codl_config->predict_model_filenames()[8]);
  unpad_rf_->BuildFromJson(path.c_str());

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dCpuWinogradRFLatencyPredictor::Init(
    std::shared_ptr<Conv2dCpuGemmRFLatencyPredictor> gemm_predictor) {
  set_gemm_predictor(gemm_predictor);
  return Init();
}

double Conv2dCpuWinogradRFLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_CHECK(inputs.size() == 11);
  MACE_CHECK(gemm_predictor_ != nullptr);

  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t oc = static_cast<index_t>(inputs[3]);
  const index_t kh = static_cast<index_t>(inputs[4]);
  const index_t kw = static_cast<index_t>(inputs[5]);
  const int sh = static_cast<int>(inputs[6]);
  const int sw = static_cast<int>(inputs[7]);
  const int dh = static_cast<int>(inputs[8]);
  const int dw = static_cast<int>(inputs[9]);
  const int padding_type = static_cast<int>(inputs[10]);

  const index_t input_size = ih * iw * ic;
  const index_t filter_size = oc * ic * kh * kw;
  if (input_size <= 0 || filter_size <= 0) {
    return 0;
  }

  const std::vector<index_t> input_shape = {1, ih, iw, ic};
  const std::vector<index_t> kernel_shape = {oc, ic, kh, kw};
  const int dilations[2] = {dh, dw};
  const int strides[2] = {sh, sw};
  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  CalcNHWCPaddingAndOutputSize(
      input_shape.data(), kernel_shape.data(), dilations, strides,
      static_cast<Padding>(padding_type), output_shape.data(), paddings.data());

  const index_t oh = output_shape[1];
  const index_t ow = output_shape[2];

  int pad_ih, pad_iw, pad_oh, pad_ow;
  int total_out_tile_count, out_tile_size;
  CalcK3x3S1WinogradOutTileCount(ih, iw, oh, ow,
                                 &pad_ih, &pad_iw, &pad_oh, &pad_ow,
                                 &total_out_tile_count, &out_tile_size);

  int tile_size, max_thread_tiles;
  int max_thread_ic, max_thread_oc;
  utils::ThreadPool &thread_pool
      = context->device()->cpu_runtime()->thread_pool();
  thread_pool.Calculate2DTileCount(0, 1, 1, 0, ic, 1,
                                   nullptr, &tile_size, &max_thread_tiles);
  max_thread_ic = tile_size * max_thread_tiles;
  thread_pool.Calculate2DTileCount(0, 1, 1, 0, oc, 1,
                                   nullptr, &tile_size, &max_thread_tiles);
  max_thread_oc = tile_size * max_thread_tiles;

  std::vector<std::pair<std::string, double>> pad_features;
  std::vector<std::pair<std::string, double>> trans_in_features;
  std::vector<std::pair<std::string, double>> trans_out_features;
  std::vector<std::pair<std::string, double>> unpad_features;
  pad_features.emplace_back("IH", ih);
  pad_features.emplace_back("IW", iw);
  pad_features.emplace_back("IC", ic);
  trans_in_features.emplace_back("PIH", pad_ih);
  trans_in_features.emplace_back("PIW", pad_iw);
  trans_in_features.emplace_back("OT", total_out_tile_count);
  trans_out_features.emplace_back("POH", pad_oh);
  trans_out_features.emplace_back("POW", pad_ow);
  trans_out_features.emplace_back("OT", total_out_tile_count);
  unpad_features.emplace_back("OH", oh);
  unpad_features.emplace_back("OW", ow);
  unpad_features.emplace_back("OC", oc);
  const double t_pad = pad_rf_->Predict(pad_features);
  const double t_trans_in = trans_in_rf_->Predict(trans_in_features);
  const double t_trans_out = trans_out_rf_->Predict(trans_out_features);
  const double t_unpad = unpad_rf_->Predict(unpad_features);

  const double lat_pad = t_pad * ih * iw * ic;
  const double lat_trans_in
      = t_trans_in * max_thread_ic * total_out_tile_count;
  const double lat_trans_out
      = t_trans_out * max_thread_oc * total_out_tile_count;
  const double lat_unpad = t_unpad * oh * ow * oc;

  const index_t m = oc;
  const index_t k = ic;
  const index_t n = total_out_tile_count;
  const double t_gemm = gemm_predictor_->Predict(context, m, k, n);
  const double lat_gemm = t_gemm * (out_tile_size + 2) * (out_tile_size + 2);

  const double lat_op
      = lat_pad + lat_trans_in + lat_gemm + lat_trans_out + lat_unpad;

#if 0
  LOG(INFO) << "lat_pad " << lat_pad
            << ", lat_trans_in " << lat_trans_in
            << ", lat_gemm " << lat_gemm
            << ", lat_trans_out " << lat_trans_out
            << ", lat_unpad " << lat_unpad;
#endif
  
  return lat_op;
}

MaceStatus Conv2dGpuDirectRFLatencyPredictor::Init() {
  rf_.reset(new utils::RandomForestModel("GPU_DIRECT"));

  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  std::string path = MakeString(codl_config->predict_model_path(), "/",
                                codl_config->predict_model_filenames()[9]);
  rf_->BuildFromJson(path.c_str());

  return MaceStatus::MACE_SUCCESS;
}

double Conv2dGpuDirectRFLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_UNUSED(context);
  MACE_CHECK(inputs.size() == 11);

  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t oc = static_cast<index_t>(inputs[3]);
  const index_t kh = static_cast<index_t>(inputs[4]);
  const index_t kw = static_cast<index_t>(inputs[5]);
  const int sh = static_cast<int>(inputs[6]);
  const int sw = static_cast<int>(inputs[7]);
  const int dh = static_cast<int>(inputs[8]);
  const int dw = static_cast<int>(inputs[9]);
  const int padding_type = static_cast<int>(inputs[10]);

  MACE_CHECK(kh == kw && sh == sw);

  const index_t input_size = ih * iw * ic;
  const index_t filter_size = oc * ic * kh * kw;
  if (input_size <= 0 || filter_size <= 0) {
    return 0;
  }

  const std::vector<index_t> input_shape = {1, ih, iw, ic};
  const std::vector<index_t> kernel_shape = {oc, ic, kh, kw};
  const int dilations[2] = {dh, dw};
  const int strides[2] = {sh, sw};
  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  CalcNHWCPaddingAndOutputSize(
      input_shape.data(), kernel_shape.data(), dilations, strides,
      static_cast<Padding>(padding_type), output_shape.data(), paddings.data());
  const index_t oh = output_shape[1];
  const index_t ow = output_shape[2];

  const uint32_t default_kwg_size = 1024;
  
  index_t ow_blk_size;
  int owb, icb, ocb;
  int num_warps;
  uint32_t gws[3];
  std::vector<uint32_t> lws;
  GetOpenCLWidthBlockSize(kh, &ow_blk_size);
  CalcOpenCLBlockCount(ow, ic, oc, kh, &owb, &icb, &ocb);

  auto opencl_runtime = context->device()->gpu_runtime()->opencl_runtime();
  gws[0] = ocb; gws[1] = owb; gws[2] = oh;
  switch (kh) {
    case 1:
      CalcOpenCLConv2dK1x1LWS(opencl_runtime, gws, default_kwg_size,
                              ow_blk_size, lws);
      break;
    case 3:
      CalcOpenCLConv2dK3x3LWS(opencl_runtime, gws, default_kwg_size,
                              ow_blk_size, lws);
      break;
    default:
      CalcOpenCLConv2dGeneralLWS(
          opencl_runtime, gws, default_kwg_size, ow_blk_size, kh, lws);
  }
  
  OpenCLUtil::CalcWarpNumber(opencl_runtime, gws, lws.data(), &num_warps);

  std::vector<std::pair<std::string, double>> features;
  features.emplace_back("OH", oh);
  features.emplace_back("OWB", owb);
  features.emplace_back("ICB", icb);
  features.emplace_back("OCB", ocb);
  features.emplace_back("KS", kh);
  features.emplace_back("S", sh);
  
  const double t_warp = rf_->Predict(features);

  const double lat_op = num_warps * t_warp;

#if 0
  LOG(INFO) << "oh " << oh
            << ", owb " << owb
            << ", icb " << icb
            << ", ocb " << ocb
            << ", ks " << kh
            << ", s " << sh
            << ", num_warps " << num_warps
            << ", t_warp " << t_warp;
  IoUtil::Pause();
#endif

  return lat_op;
}

MaceStatus Conv2dCpuRFLatencyPredictor::Init() {
  cpu_direct_predictor_.reset(new Conv2dCpuDirectRFLatencyPredictor());
  cpu_gemm_predictor_.reset(new Conv2dCpuGemmRFLatencyPredictor());
  cpu_winograd_predictor_.reset(new Conv2dCpuWinogradRFLatencyPredictor());

  cpu_direct_predictor_->Init();
  cpu_gemm_predictor_->Init();
  cpu_winograd_predictor_->Init(cpu_gemm_predictor_);

  return MaceStatus::MACE_SUCCESS;
}

double Conv2dCpuRFLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t oc = static_cast<index_t>(inputs[3]);
  const index_t kh = static_cast<index_t>(inputs[4]);

  const Conv2dImplType impl_type = GetConv2dCpuImplType(ic, oc, kh);
  switch (impl_type) {
    case CONV2D_DIRECT:
      return cpu_direct_predictor_->Predict(context, inputs);
    case CONV2D_GEMM:
      return cpu_gemm_predictor_->Predict(context, inputs);
    case CONV2D_WINOGRAD:
      return cpu_winograd_predictor_->Predict(context, inputs);
  }

  return 0;
}

MaceStatus Conv2dGpuRFLatencyPredictor::Init() {
  gpu_direct_predictor_.reset(new Conv2dGpuDirectRFLatencyPredictor());
  gpu_direct_predictor_->Init();
  return MaceStatus::MACE_SUCCESS;
}

double Conv2dGpuRFLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {

  const Conv2dImplType impl_type = GetConv2dGpuImplType();
  switch (impl_type) {
    case CONV2D_DIRECT:
      return gpu_direct_predictor_->Predict(context, inputs);
    default:
      return 0;
  }

  return 0;
}

}  // namespace ops
}  // namespace mace

#endif
