
#include <cmath>
#include "mace/core/runtime/opencl/opencl_util.h"
#include "mace/ops/conv_2d_latency_predict.h"

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/conv_2d.h"
#endif

namespace mace {
namespace ops {

MaceStatus Conv2dCpuDirectLRLatencyPredictor::Init(LPInitContext *context) {
  lr_models_ = std::vector<std::shared_ptr<utils::LinearRegressionModel>>(
                  model_count_);
  for (size_t i = 0; i < lr_models_.size(); i ++) {
    const std::vector<double> means = {0, 0, 0, 0};
    const std::vector<double> stds = {1, 1, 1, 1};
    const std::vector<double> coefs = {0, 0, 0, 0};
    const double inter = 0;
    lr_models_[i].reset(new utils::LinearRegressionModel(
        "CONV2D_CPU_DIRECT", means, stds, coefs, inter));
  }

  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  if (codl_config->is_loaded()) {
    const index_t cur_idx = context->cur_idx();
    context->set_cur_idx(cur_idx + model_count_);
    for (size_t i = 0; i < lr_models_.size(); i ++) {
      const std::string path = MakeString(
          codl_config->predict_model_path(), "/",
          codl_config->predict_model_filenames()[cur_idx + i]);
      lr_models_[i]->BuildFromJson(path.c_str());
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

double Conv2dCpuDirectLRLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_UNUSED(context);
  MACE_CHECK(lr_models_.size() > 0);
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

  int kChannelPerItem = 4;
  if (kh == 3) {
    if (sh == 1) {
      kChannelPerItem = 2;
    } else if (sh == 2) {
      kChannelPerItem = 1;
    }
  }
  int tile_size;
  int max_thread_tiles;
  double multithread_coef = 1.0;
  utils::ThreadPool &thread_pool
      = context->device()->cpu_runtime()->thread_pool();
  thread_pool.Calculate2DTileCount(0, 1, 1, 0, oc, kChannelPerItem,
                                   nullptr, &tile_size, &max_thread_tiles);
#if 0
  if (thread_pool.thread_count() == 4) {
    multithread_coef = 1.83;
  }
#endif
#if 1
  VLOG(1) << "tile_size " << tile_size << ", max_thread_tiles " << max_thread_tiles;
#endif

  //const double flops_per_tile = tile_size * kChannelPerItem * flops / oc;

  const double flops_per_tile = flops / oc;
  max_thread_tiles = max_thread_tiles * tile_size * kChannelPerItem;

  const index_t k = kh;
  const int s = sh;

  std::vector<double> lr_inputs;
  lr_inputs.emplace_back(static_cast<double>(ih));
  lr_inputs.emplace_back(static_cast<double>(iw));
  lr_inputs.emplace_back(static_cast<double>(ic));
  lr_inputs.emplace_back(static_cast<double>(oc));

  int kernel_idx = -1;
  if (k == 3 && s == 1) {
    kernel_idx = 0;
  } else if (k == 3 && s == 2) {
    kernel_idx = 1;
  } else if (k == 5 && s == 1) {
    kernel_idx = 2;
  } else if (k == 5 && s == 2) {
    kernel_idx = 3;
  } else if (k == 7 && s == 1) {
    kernel_idx = 4;
  } else if (k == 7 && s == 2) {
    kernel_idx = 5;
  } else if (k == 9 && s == 1) {
    kernel_idx = 6;
  } else if (k == 9 && s == 2) {
    kernel_idx = 7;
  } else if (s == 1) {
    kernel_idx = 8;
  } else if (s == 2) {
    kernel_idx = 9;
  }

  MACE_CHECK(kernel_idx >= 0);

  const double t_flops = lr_models_[kernel_idx]->Predict(lr_inputs);

  const double lat_op = max_thread_tiles * flops_per_tile * t_flops * multithread_coef;

#if 1
  VLOG(1) << "flops " << flops
          << ", tile_size " << tile_size
          << ", max_thread_tiles " << max_thread_tiles
          << ", flops_per_tile " << flops_per_tile
          << ", t_flops " << t_flops;
#endif

  return lat_op;
}

MaceStatus Conv2dCpuWinogradLRLatencyPredictor::Init(LPInitContext *context) {
  const std::vector<double> means = {0, 0, 0};
  const std::vector<double> stds = {1, 1, 1};
  const std::vector<double> coefs = {0, 0, 0};
  const double inter = 0;
  pad_lr_.reset(new utils::LinearRegressionModel("WINOGRAD_PAD", means, stds, coefs, inter));
  trans_in_lr_.reset(new utils::LinearRegressionModel("WINOGRAD_TRANS_IN", means, stds, coefs, inter));
  comp_lr_.reset(new utils::LinearRegressionModel("WINOGRAD_COMP", means, stds, coefs, inter));
  trans_out_lr_.reset(new utils::LinearRegressionModel("WINOGRAD_TRANS_OUT", means, stds, coefs, inter));
  unpad_lr_.reset(new utils::LinearRegressionModel("WINOGRAD_UNPAD", means, stds, coefs, inter));

  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  if (codl_config->is_loaded()) {
    const index_t cur_idx = context->cur_idx();
    context->set_cur_idx(cur_idx + model_count_);
    std::string path;
    path = MakeString(codl_config->predict_model_path(), "/",
                      codl_config->predict_model_filenames()[cur_idx]);
    pad_lr_->BuildFromJson(path.c_str());
    path = MakeString(codl_config->predict_model_path(), "/",
                      codl_config->predict_model_filenames()[cur_idx + 1]);
    trans_in_lr_->BuildFromJson(path.c_str());
    path = MakeString(codl_config->predict_model_path(), "/",
                      codl_config->predict_model_filenames()[cur_idx + 2]);
    comp_lr_->BuildFromJson(path.c_str());
    path = MakeString(codl_config->predict_model_path(), "/",
                      codl_config->predict_model_filenames()[cur_idx + 3]);
    trans_out_lr_->BuildFromJson(path.c_str());
    path = MakeString(codl_config->predict_model_path(), "/",
                      codl_config->predict_model_filenames()[cur_idx + 4]);
    unpad_lr_->BuildFromJson(path.c_str());
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dCpuWinogradLRLatencyPredictor::Init(
    LPInitContext *context,
    std::shared_ptr<Conv2dCpuGemmLRLatencyPredictor> gemm_predictor) {
  set_gemm_predictor(gemm_predictor);
  return Init(context);
}

double Conv2dCpuWinogradLRLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_CHECK(inputs.size() == 11);
  //MACE_CHECK(gemm_predictor_ != nullptr);
  MACE_CHECK_NOTNULL(pad_lr_);
  MACE_CHECK_NOTNULL(trans_in_lr_);
  MACE_CHECK_NOTNULL(comp_lr_);
  MACE_CHECK_NOTNULL(trans_out_lr_);
  MACE_CHECK_NOTNULL(unpad_lr_);

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

  int tile_size;
  int max_thread_tiles;
  double multithread_coef = 1.0;
  utils::ThreadPool &thread_pool
      = context->device()->cpu_runtime()->thread_pool();
  thread_pool.Calculate2DTileCount(0, 1, 1, 0, ic, 1,
                                   nullptr, &tile_size, &max_thread_tiles);
  const int max_thread_ic = tile_size * max_thread_tiles;
  thread_pool.Calculate2DTileCount(0, 1, 1, 0, oc, 1,
                                   nullptr, &tile_size, &max_thread_tiles);
  const int max_thread_oc = tile_size * max_thread_tiles;
#if 0
  if (thread_pool.thread_count() == 4) {
    multithread_coef = 1.83;
  }
#endif

  std::vector<std::pair<std::string, double>> pad_features;
  std::vector<std::pair<std::string, double>> trans_in_features;
  std::vector<std::pair<std::string, double>> trans_out_features;
  std::vector<std::pair<std::string, double>> unpad_features;
  pad_features.emplace_back("IH", ih);
  pad_features.emplace_back("IW", iw);
  pad_features.emplace_back("IC", ic);
  trans_in_features.emplace_back("PIH", pad_ih);
  trans_in_features.emplace_back("PIW", pad_iw);
  trans_in_features.emplace_back("OT", out_tile_size);
  trans_out_features.emplace_back("POH", pad_oh);
  trans_out_features.emplace_back("POW", pad_ow);
  trans_out_features.emplace_back("OT", out_tile_size);
  unpad_features.emplace_back("OH", oh);
  unpad_features.emplace_back("OW", ow);
  unpad_features.emplace_back("OC", oc);
  const double t_pad = pad_lr_->Predict(pad_features);
  const double t_trans_in = trans_in_lr_->Predict(trans_in_features);
  const double t_trans_out = trans_out_lr_->Predict(trans_out_features);
  const double t_unpad = unpad_lr_->Predict(unpad_features);

  const double lat_pad = t_pad * ih * iw * ic;
  const double lat_trans_in
      = t_trans_in * max_thread_ic * total_out_tile_count;
  const double lat_trans_out
      = t_trans_out * max_thread_oc * total_out_tile_count;
  const double lat_unpad = t_unpad * oh * ow * oc;

  const index_t m = oc;
  const index_t k = ic;
  const index_t n = total_out_tile_count;
  int mb, kb, nb;
  arm::fp32::Gemm::CalcBlockCount(m, n, k, &mb, &nb, &kb);
  thread_pool.Calculate1DTileCount(0, mb, 1, &tile_size, &max_thread_tiles);

  const int max_thread_mb = tile_size * max_thread_tiles;
  const int num_gemm_blocks = max_thread_mb * nb * kb;
  std::vector<std::pair<std::string, double>> comp_features;
  comp_features.emplace_back("M", m);
  comp_features.emplace_back("K", k);
  comp_features.emplace_back("N", n);
  const double t_gemm =  comp_lr_->Predict(comp_features);
  const double t_comp = t_gemm * num_gemm_blocks;
  const double lat_gemm = t_comp * (out_tile_size + 2) * (out_tile_size + 2) * multithread_coef;

  const double lat_op
      = lat_pad + lat_trans_in + lat_gemm + lat_trans_out + lat_unpad;

#if 1
  VLOG(1) << "pad_ih " << pad_ih << ", pad_iw " << pad_iw
          << ", ot " << out_tile_size << ", t_trans_in " << t_trans_in;
  VLOG(1) << "max_thread_ic " << max_thread_ic
          << ", total_out_tile_count " << total_out_tile_count
          << ", lat_trans_in " << lat_trans_in;
#endif

#if 1
  VLOG(1) << "pad_oh " << pad_oh << ", pad_ow " << pad_ow
          << ", ot " << out_tile_size << ", t_trans_out " << t_trans_out;
  VLOG(1) << "max_thread_oc " << max_thread_oc
          << ", total_out_tile_count " << total_out_tile_count
          << ", lat_trans_out " << lat_trans_out;
#endif

#if 1
  VLOG(1) << "m " << m << ", k " << k << ", n " << n
          << ", max_thread_mb " << max_thread_mb
          << ", kb " << kb
          << ", nb " << nb
          << ", num_gemm_blocks " << num_gemm_blocks
          << ", t_gemm " << t_gemm;
  VLOG(1) << "out_tile_size " << out_tile_size << ", t_comp " << t_comp;
#endif

#if 1
  VLOG(1) << "lat_pad " << lat_pad
          << ", lat_trans_in " << lat_trans_in
          << ", lat_gemm " << lat_gemm
          << ", lat_trans_out " << lat_trans_out
          << ", lat_unpad " << lat_unpad;
#endif
  
  return lat_op;
}

MaceStatus Conv2dGpuDirectLRLatencyPredictor::Init(LPInitContext *context) {
  lr_models_ = std::vector<std::shared_ptr<utils::LinearRegressionModel>>(
                  model_count_);
  for (size_t i = 0; i < lr_models_.size(); i ++) {
    const std::vector<double> means = {0, 0, 0, 0};
    const std::vector<double> stds = {1, 1, 1, 1};
    const std::vector<double> coefs = {0, 0, 0, 0};
    const double inter = 0;
    lr_models_[i].reset(new utils::LinearRegressionModel(
        "CONV2D_GPU_DIRECT", means, stds, coefs, inter));
  }

  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  if (codl_config->is_loaded()) {
    const index_t cur_idx = context->cur_idx();
    context->set_cur_idx(cur_idx + model_count_);
    for (size_t i = 0; i < lr_models_.size(); i ++) {
      const std::string path = MakeString(
          codl_config->predict_model_path(), "/",
          codl_config->predict_model_filenames()[cur_idx + i]);
      lr_models_[i]->BuildFromJson(path.c_str());
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

double Conv2dGpuDirectLRLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  MACE_UNUSED(context);
  MACE_CHECK(lr_models_.size() > 0);
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
  
  index_t ow_blk_size;
  int owb, icb, ocb;
  int num_warps;
  uint32_t gws[3];
  std::vector<uint32_t> lws;
  GetOpenCLWidthBlockSize(kh, &ow_blk_size);
  CalcOpenCLBlockCount(ow, ic, oc, kh, &owb, &icb, &ocb);

  auto opencl_runtime = context->device()->gpu_runtime()->opencl_runtime();
  const uint32_t kwg_size = opencl_runtime->kwg_size();
  MACE_CHECK(kwg_size > 0);
  gws[0] = ocb; gws[1] = owb; gws[2] = oh;
  switch (kh) {
    case 1:
      CalcOpenCLConv2dK1x1LWS(opencl_runtime, gws, kwg_size, ow_blk_size, lws);
      break;
    case 3:
      CalcOpenCLConv2dK3x3LWS(opencl_runtime, gws, kwg_size, ow_blk_size, lws);
      break;
    default:
      CalcOpenCLConv2dGeneralLWS(opencl_runtime, gws, kwg_size,
                                 ow_blk_size, kh, lws);
  }
  
  OpenCLUtil::CalcWarpNumber(opencl_runtime, gws, lws.data(), &num_warps);

  std::vector<std::pair<std::string, double>> features;
  features.emplace_back("OH", oh);
  features.emplace_back("OWB", owb);
  features.emplace_back("ICB", icb);
  features.emplace_back("OCB", ocb);

  int kernel_idx = -1;
  if (kh == 1 && sh == 1) {
    kernel_idx = 0;
  } else if (kh == 1 && sh == 2) {
    kernel_idx = 1;
  } else if (kh == 3 && sh == 1) {
    kernel_idx = 2;
  } else if (kh == 3 && sh == 2) {
    kernel_idx = 3;
  } else if (kh == 5 && sh == 1) {
    kernel_idx = 4;
  } else if (kh == 5 && sh == 2) {
    kernel_idx = 5;
  } else if (kh == 7 && sh == 1) {
    kernel_idx = 6;
  } else if (kh == 7 && sh == 2) {
    kernel_idx = 7;
  } else if (kh == 9 && sh == 1) {
    kernel_idx = 8;
  } else if (kh == 9 && sh == 2) {
    kernel_idx = 9;
  }

  MACE_CHECK(kernel_idx >= 0);
  const double t_warp_icb = lr_models_[kernel_idx]->Predict(features);
  const double t_warp = icb * t_warp_icb;
  const double lat_op = num_warps * t_warp;

#if 1
  VLOG(1) << "oh " << oh << ", owb " << owb << ", icb " << icb << ", ocb " << ocb
          << ", ks " << kh << ", s " << sh << ", num_warps " << num_warps
          << ", t_warp " << t_warp;
#endif

  return lat_op;
}

MaceStatus Conv2dCpuLRLatencyPredictor::Init(LPInitContext *context) {
  cpu_direct_predictor_.reset(new Conv2dCpuDirectLRLatencyPredictor());
  cpu_gemm_predictor_.reset(new Conv2dCpuGemmLRLatencyPredictor());
  cpu_winograd_predictor_.reset(new Conv2dCpuWinogradLRLatencyPredictor());

  cpu_direct_predictor_->Init(context);
  cpu_gemm_predictor_->Init(context);
  cpu_winograd_predictor_->Init(context);

  return MaceStatus::MACE_SUCCESS;
}

double Conv2dCpuLRLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<double> &inputs) {
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t oc = static_cast<index_t>(inputs[3]);
  const index_t kh = static_cast<index_t>(inputs[4]);
  const index_t sh = static_cast<index_t>(inputs[6]);

  const Conv2dImplType impl_type = GetConv2dCpuImplType(ic, oc, kh, sh);
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

MaceStatus Conv2dGpuLRLatencyPredictor::Init(LPInitContext *context) {
  gpu_direct_predictor_.reset(new Conv2dGpuDirectLRLatencyPredictor());
  gpu_direct_predictor_->Init(context);
  return MaceStatus::MACE_SUCCESS;
}

double Conv2dGpuLRLatencyPredictor::Predict(
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

MaceStatus Conv2dFLOPsLatencyPredictor::Init(LPInitContext *context) {
#if 0
  const std::vector<double> means = {0};
  const std::vector<double> stds = {1};
  const std::vector<double> coefs = {0};
#endif
  const std::vector<double> means = {0, 0};
  const std::vector<double> stds = {1, 1};
  const std::vector<double> coefs = {0, 0};
  const double inter = 0;
  lr_.reset(new utils::LinearRegressionModel("CONV2D_FLOPS", means, stds, coefs, inter));

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

double Conv2dFLOPsLatencyPredictor::Predict(
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
  
#if 0
  // Calculate FLOPs.
  const double flops = MaceFLOPsStatistics::Compute(
      "Conv2D", kernel_shape, output_shape);

  std::vector<double> lr_inputs;
  lr_inputs.emplace_back(static_cast<double>(flops));
#endif

  // Calculate features.
  const double f0 = ih * iw * ic;
  const double f1 = pow(((double) kh / sh), 2.0) * oc;

  std::vector<double> lr_inputs;
  lr_inputs.emplace_back(static_cast<double>(f0));
  lr_inputs.emplace_back(static_cast<double>(f1));

  // Predict OP latency.
  const double lat_op = lr_->Predict(lr_inputs);

#if 1
  VLOG(1) << "f0 " << f0 << ", f1 " << f1;
#endif

  return lat_op;
}

}  // namespace ops
}  // namespace mace
