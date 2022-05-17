
#include "mace/utils/math.h"
#include "test/codl_run/nn_model_builder.h"

#define APPEND_CONV2D(...) \
  params.emplace_back(new CodlConv2dChainParam( \
      __VA_ARGS__, common_param_)); op_idx ++;

#define APPEND_POOLING(...) \
  params.emplace_back(new CodlPoolingChainParam( \
      __VA_ARGS__, common_param_)); op_idx ++;

#define APPEND_FULLY_CONNECTED(...) \
  params.emplace_back(new CodlFullyConnectedChainParam( \
      __VA_ARGS__, common_param_)); op_idx ++;

#define APPEND_MATMUL(...) \
  params.emplace_back(new CodlMatMulChainParam( \
      __VA_ARGS__, common_param_)); op_idx ++;

void Yolov2Builder::Build(
    const int chain_idx,
    const std::vector<int> chain_lengths,
    const std::vector<int> pdims,
    const std::vector<float> pratioes,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  MACE_CHECK(chain_idx < static_cast<int>(op_chain_count_));
  MACE_CHECK(pdims.size() == op_count_);
  MACE_CHECK(pratioes.size() == op_count_);
  int op_idx = 0;
  double size_factor = 1.0;
  double ch_factor = 1.0;
  int in_size = (416 + 0);
  APPEND_CONV2D(in_size+2, in_size+2, 3, 32, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(in_size*size_factor, in_size*size_factor, 32, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 2;
  in_size = RoundUpDiv(in_size, 2);
  APPEND_CONV2D(in_size+2, in_size+2, 32, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(in_size*size_factor, in_size*size_factor, 64, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 4;
  in_size = RoundUpDiv(in_size, 2);
  APPEND_CONV2D(in_size+2, in_size+2, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size*size_factor, in_size*size_factor, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(in_size*size_factor, in_size*size_factor, 128, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 8;
  in_size = RoundUpDiv(in_size, 2);
  APPEND_CONV2D(in_size+2, in_size+2, 128, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size*size_factor, in_size*size_factor, 256, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 128, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(in_size*size_factor, in_size*size_factor, 256, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 13;
  in_size = RoundUpDiv(in_size, 2);
  APPEND_CONV2D(in_size+2, in_size+2, 256, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size*size_factor, in_size*size_factor, 512, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 256, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size*size_factor, in_size*size_factor, 512, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 256, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(in_size*size_factor, in_size*size_factor, 512, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 19;
  in_size = RoundUpDiv(in_size, 2);
  APPEND_CONV2D(in_size+2, in_size+2, 512 * ch_factor, 1024, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 1024, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 512 * ch_factor, 1024, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 1024, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 512 * ch_factor, 1024, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 1024 * ch_factor, 1024, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 1024 * ch_factor, 1024, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 1024 * ch_factor, 1536, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 125 * ch_factor, 1024, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);

  if (chain_lengths.size() > 0) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    if (chain_idx > -1 && chain_idx < static_cast<int>(chain_lengths.size())) {
      LOG(INFO) << "===== Build YOLO-v2 Info =====";
      LOG(INFO) << "chain_idx " << chain_idx
                << ", chain_lengths " << chain_lengths[chain_idx];
      size_t start_op_idx = 0;
      for (int i = 0; i < chain_idx; i ++) {
        start_op_idx += chain_lengths[i];
      }
      for (int i = 0; i < chain_lengths[chain_idx]; i ++) {
        out_params.emplace_back(params[start_op_idx + i]);
      }
    }
    params = out_params;
  }
}

void Yolov2Builder::Build(
    const int chain_idx,
    const int chain_param_hint,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  std::vector<int> chain_lengths;
  std::vector<int> pdims;
  std::vector<float> pratioes;
  if (chain_param_hint == 0) {
    // CPU.
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.0);
  } else if (chain_param_hint == 1) {
    // GPU.
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 1.0);
  } else if (chain_param_hint == 2) {
    // CPU+GPU+H.
    
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 3) {
    // CPU+GPU+H.
    pdims = std::vector<int>(op_count_, 4);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 4) {
    // CPU+GPU+H, profiling.
    chain_lengths = {8, 4, 6, 2, 2, 1, 1, 1, 1, 1};

    pdims = std::vector<int>(op_count_, 1);

    // h_pratioes = {1.0,1.0,0.7,1.0,0.6,1.0,0.7,1.0,0.5,1.0,0.5,1.0,0.6,1.0,0.5,1.0,0.5,1.0,0.4,1.0,0.5,1.0,0.5,0.4,0.5,0.5,1.0};
    
    // const std::vector<float> h_oc_pratioes = {1.0, 1.0,
    //                                           0.7, 1.0,
    //                                           0.6, 1.0, 0.7, 1.0,
    //                                           0.5, 1.0, 0.5, 1.0,
    //                                           0.6, 1.0, 0.6, 1.0, 0.5, 1.0,
    //                                           0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5,
    //                                           0.5, 1.0};
    
    // const std::vector<float> h_oc_chain_pratioes = {
    //     1.0, 1.0,
    //     0.7, 0.7,
    //     0.7, 0.7, 0.7, 0.7,
    //     0.5, 0.5, 0.5, 0.5,
    //     0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    //     0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5,
    //     0.5, 1.0};
    
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 5) {
    // CPU+GPU+OC, profiling.
    pdims = std::vector<int>(op_count_, 4);
    // oc_pratioes = {1,1,1,1,0.8,1,0.8,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,0.5,0.5,0.5,1};
    // h_oc_pratioes = {1.0, 1.0,
    //                  0.7, 1.0,
    //                  0.6, 1.0, 0.7, 1.0,
    //                  0.5, 1.0, 0.5, 1.0,
    //                  0.6, 1.0, 0.6, 1.0, 0.5, 1.0,
    //                  0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5,
    //                  0.5, 1.0};
    const std::vector<float> h_oc_chain_pratioes = {
        1.0, 1.0,
        0.7, 0.7,
        0.7, 0.7, 0.7, 0.7,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5,
        0.5, 1.0};
    pratioes = h_oc_chain_pratioes;
  } else if (chain_param_hint == 6) {
    // CPU+GPU+H, prediction.
    pdims = std::vector<int>(op_count_, 1);
    pratioes = {1,1,0.5,1,0.4,1,0.4,1,0.3,1,0.3,1,0.3,1,0.3,1,0.3,1,0.3,1,0.3,1,0.3,0.3,0.3,0.3,0.3};
  } else if (chain_param_hint == 7) {
    // CPU+GPU+OC, prediction.
    pdims = std::vector<int>(op_count_, 4);
    pratioes = {1,0,0.7,0,0.5,1,0.5,0,0.4,0.8,0.4,0,0.4,0.5,0.4,0.5,0.4,0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
  } else if (chain_param_hint == 10) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
      pdims = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 4, 1,
               4, 1, 4, 4, 4, 4, 1};
      //pratioes = std::vector<float>(op_count_, 0.5);
      pratioes = {1.0, 1.0,
                  0.7, 1.0,
                  0.6, 1.0, 0.7, 1.0,
                  0.5, 1.0, 0.5, 1.0,
                  0.6, 1.0, 0.6, 1.0, 0.5, 1.0,
                  0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5,
                  0.5, 1.0};
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
      pdims = {1,1,1,1,1,1,1,1,4,1,
               4,1,1,1,4,1,1,1,4,1,
               4,1,4,4,4,4,1};
      pratioes = {1,1,0.7,1,0.7,1,0.6,1,0.5,1,
                  0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,
                  0.5,1,0.5,0.5,0.5,0.5,1};
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
      pdims = {1,1,1,1,1,1,1,1,4,1,
               4,1,1,1,1,1,1,1,1,1,
               4,1,1,4,4,4,1};
      pratioes = {1,1,0.7,1,0.6,1,0.6,1,0.6,1,
                  0.6,1,0.5,1,0.5,1,0.5,1,0.5,1,
                  0.5,1,0.5,0.5,0.5,0.5,1};
    } else if (!codl_config->soc_name().compare("Kirin990")) {
      pdims = std::vector<int>(op_count_, 1);
      pratioes = {0.5,1.0,1.0,1.0,1.0,0.5,0.4,1.0,0.3,0.5,
                  0.3,1.0,0.3,0.4,0.3,0.4,0.3,1.0,0.3,0.4,
                  0.3,0.4,0.3,0.3,0.3,0.3,0.3};
    }
  } else if (chain_param_hint == 11) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
      chain_lengths = {2, 6, 3, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      //const std::vector<int> h_pdims = std::vector<int>(op_count_, 1);
      const std::vector<int> h_oc_pdims = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           4, 1, 4, 1, 4, 4, 4, 4, 1};
      pdims = h_oc_pdims;
      pratioes = {1.0, 1.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                  0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0};
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
      chain_lengths = {2,5,5,2,2,4,2,1,1,1,1,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,4,4,4,4,1};
      pratioes = {0.8,0.8,0.7,0.7,0.7,0.7,0.7,0.5,0.5,0.5,
                  0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.4,0.4,0.4,
                  0.4,0.4,0.5,0.5,0.5,0.5,1};
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
      chain_lengths = {8,4,4,2,2,2,2,1,1,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1};
      pratioes = {0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.4,0.5,0.5,0.5,0.5,0.4,0.4,0.5,0.5,1};
    } else if (!codl_config->soc_name().compare("Kirin990")) {
      chain_lengths = std::vector<int>(op_count_, 1);
      pdims = std::vector<int>(op_count_, 1);
      pratioes = {0.5,1.0,1.0,1.0,1.0,0.5,0.4,1.0,0.3,0.5,
                  0.3,1.0,0.3,0.4,0.3,0.4,0.3,1.0,0.3,0.4,
                  0.3,0.4,0.3,0.3,0.3,0.3,0.3};
    }
  } else if (chain_param_hint == 12) {
    chain_lengths = {4};
    pdims = {1, 1, 1, 1};
    pratioes = {0.7, 0.7, 0.7, 0.7};
  } else if (chain_param_hint == 20) {
    // CPU+GPU+H+OC, prediction.
    pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4};
    pratioes = {1,1,0.5,1,0.4,1,0.4,1,0.3,1,0.3,1,0.3,1,0.3,1,0.3,1,0.3,1,0.3,1,0.3,0.3,0.3,0.3,0.5};
  } else if (chain_param_hint == 21) {
    // CPU+GPU+H+OC+Chain, prediction.
    chain_lengths = {2,3,2,4,3,2,2,2,2,5};
    pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    pratioes = {0.9,0.9,0.5,0.5,0.5,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2};
  }

  Build(chain_idx, chain_lengths, pdims, pratioes, params);
}

void PosenetBuilder::Build(
    const int chain_idx,
    const std::vector<int> chain_lengths,
    const std::vector<int> pdims,
    const std::vector<float> pratioes,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  int op_idx = 0;
  double size_factor = 1.0;
  double size_factor2 = 1.0;
  double ch_factor = 1.0;
  APPEND_CONV2D(258, 258, 3, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(258, 258, 64, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(256 * size_factor, 256 * size_factor, 64, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 3;
  APPEND_CONV2D(130, 130, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(130, 130, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(128 * size_factor, 128 * size_factor, 128, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 6;
  APPEND_CONV2D(66, 66, 128, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(66, 66, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(66, 66, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(66, 66, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(64 * size_factor, 64 * size_factor, 256, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 11;
  APPEND_CONV2D(34, 34, 256, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(34, 34, 512, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(34, 34, 512, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(34, 34, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(34, 34, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(34, 34, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(34, 34, 256, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(32 * size_factor, 32 * size_factor, 128, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(32 * size_factor, 32 * size_factor, 512, 21, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 19;
  APPEND_CONV2D(38, 38, 149, 128, 7, 7, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(38, 38, 128, 128, 7, 7, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(38 / size_factor2, 38 / size_factor2, 128 * ch_factor, 128 * ch_factor, 7, 7, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(38 / size_factor2, 38 / size_factor2, 128 * ch_factor, 128 * ch_factor, 7, 7, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(38 / size_factor2, 38 / size_factor2, 128 * ch_factor, 128 * ch_factor, 7, 7, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(32 * size_factor, 32 * size_factor, 128 * ch_factor, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(32 * size_factor, 32 * size_factor, 128, 21, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 26;
  APPEND_CONV2D(38, 38, 149, 128, 7, 7, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(38, 38, 128, 128, 7, 7, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(38 / size_factor2, 38 / size_factor2, 128 * ch_factor, 128 * ch_factor, 7, 7, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(38 / size_factor2, 38 / size_factor2, 128 * ch_factor, 128 * ch_factor, 7, 7, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(38 / size_factor2, 38 / size_factor2, 128 * ch_factor, 128 * ch_factor, 7, 7, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(32 * size_factor, 32 * size_factor, 128, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(32 * size_factor, 32 * size_factor, 128, 21, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  if (chain_lengths.size() > 0) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    if (chain_idx > -1 && chain_idx < static_cast<int>(chain_lengths.size())) {
      LOG(INFO) << "===== Build PoseNet Info =====";
      LOG(INFO) << "chain_idx " << chain_idx
                << ", chain_lengths " << chain_lengths[chain_idx];
      size_t start_op_idx = 0;
      for (int i = 0; i < chain_idx; i ++) {
        start_op_idx += chain_lengths[i];
      }
      for (int i = 0; i < chain_lengths[chain_idx]; i ++) {
        out_params.emplace_back(params[start_op_idx + i]);
      }
    }
    params = out_params;
  }
}

void PosenetBuilder::Build(
    const int chain_idx,
    const int chain_param_hint,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  std::vector<int> chain_lengths;
  std::vector<int> pdims;
  std::vector<float> pratioes;
  if (chain_param_hint == 0) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.0);
  } else if (chain_param_hint == 1) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 1.0);
  } else if (chain_param_hint == 2) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 3) {
    pdims = std::vector<int>(op_count_, 4);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 4) {
    chain_lengths = {6, 5, 3, 2, 4, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3};
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 5) {
    pdims = std::vector<int>(op_count_, 4);
    //pratioes = {1,0.6,1,0.6,0.5,1,0.5,0.4,0.4,0.4,1,0.6,0.5,0.5,0.6,0.6,0.6,0.6,1,1,0.6,0.7,0.6,0.7,0.7,1,1,0.6,0.7,0.7,0.6,0.7,1,1};
    pratioes = {1,0.6,1,0.6,0.5,1,0.5,0.4,0.4,0.4,1,0.5,0.5,0.5,0.6,0.6,0.6,1,1,1,0.6,0.7,0.8,0.7,0.6,1,1,0.6,0.7,0.7,0.7,0.8,1,1};
  } else if (chain_param_hint == 6) {
    pdims = std::vector<int>(op_count_, 4);
    //pratioes = {1,1,1,0.7,0.6,1,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.6,0.6,0.6,1,1,1,0.7,0.8,0.7,0.8,0.7,1,1,0.7,0.7,0.7,0.7,0.8,1,1};
    pratioes = {1,0.6,1,0.6,0.5,1,0.5,0.4,0.4,0.4,1,0.5,0.5,0.5,0.6,0.6,0.6,1,1,1,0.6,0.7,0.8,0.7,0.6,1,1,0.6,0.7,0.7,0.7,0.8,1,1};
  } else if (chain_param_hint == 10) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
      pdims = {1, 1, 1,
               1, 1, 1,
               1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1};
      pratioes = {1.0, 0.6, 1.0,
                  0.6, 0.5, 1.0,
                  0.5, 0.4, 0.4, 0.4, 1.0,
                  0.6, 0.5, 0.6, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0,
                  0.6, 0.6, 0.6, 0.6, 0.7, 1.0, 1.0,
                  0.6, 0.6, 0.6, 0.6, 0.7, 1.0, 1.0};

      pdims = {1,1,1,1,1,1,1,1,1,1,1,4,4,4,1,1,1,1,1,1,1,1,4,1,1,1,1,1,4,1,1,4,1,1};
      pratioes = {1,0.6,1,0.6,0.5,1,0.5,0.4,0.4,0.4,1,0.5,0.5,0.5,0.6,0.6,0.6,1,1,1,0.6,0.7,0.8,0.7,0.6,1,1,0.6,0.7,0.7,0.7,0.8,1,1};

      pdims = {1,1,1,1,1,1,1,1,1,1,1,4,1,1,4,1,4,1,1,1,1,4,4,1,1,1,1,4,4,4,4,1,1,1};
      pratioes = std::vector<float>(op_count_, 0.5);
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
      pdims = {1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
      pratioes = {1,0.6,1,0.6,0.5,1,0.5,0.4,0.4,0.4,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,0.6,0.7,0.6,0.7,0.7,1,1,0.7,0.7,0.7,0.7,0.7,1,1};
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
      pdims = {1,1,1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
      pratioes = {1,0.6,1,0.6,0.5,1,0.6,0.5,0.5,0.5,1,0.5,0.5,0.6,0.6,0.6,0.6,0.7,1,1,0.7,0.7,0.7,0.7,0.7,1,1,0.7,0.7,0.7,0.7,0.7,1,1};
    } else if (!codl_config->soc_name().compare("Kirin990")) {
      pdims = std::vector<int>(op_count_, 1);
      pratioes = {0.7,0.5,1.0,1.0,0.4,1.0,0.4,0.3,0.3,0.3,
                  1.0,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.5,0.5,
                  0.4,0.4,0.4,0.4,0.4,0.7,0.6,0.4,0.4,0.4,
                  0.4,0.4,0.7,0.7};
    }
  } else if (chain_param_hint == 11) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
      chain_lengths = {2, 1, 2, 1, 3, 2, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1};
      pdims = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1};
      pratioes = {0.6, 0.6, 1.0, 0.5, 0.5, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
                  0.6, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0,
                  0.7, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0};

      chain_lengths = {2,1,2,1,2,3,3,2,4,1,2,1,3,1,1,2,3};
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1};
      pratioes = {0.6,0.6,1,0.5,0.5,1,0.4,0.4,0.4,0.4,0.4,0.5,0.5,0.5,0.5,0.5,0.6,0.6,0.6,0.6,0.6,0.7,0.7,0.7,0.7,0.7,0.7,0.6,0.7,0.7,0.7,0.6,0.6,0.6};
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
      chain_lengths = {5,1,2,1,2,1,1,2,5,1,1,1,4,1,2,1,3};
      pdims = {1,1,1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
      pratioes = {0.6,0.6,0.6,0.6,0.6,1,0.4,0.4,0.4,0.4,0.4,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.6,0.7,0.6,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7};
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
      chain_lengths = {2,2,1,3,3,2,3,4,2,3,1,3,2,3};
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
      pratioes = {0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.4,0.4,0.4,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.7,0.7,0.7,1,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7};
    } else if (!codl_config->soc_name().compare("Kirin990")) {
      chain_lengths = std::vector<int>(op_count_, 1);
      pdims = std::vector<int>(op_count_, 1);
      pratioes = {0.7,0.5,1.0,1.0,0.4,1.0,0.4,0.3,0.3,0.3,
                  1.0,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.5,0.5,
                  0.4,0.4,0.4,0.4,0.4,0.7,0.6,0.4,0.4,0.4,
                  0.4,0.4,0.7,0.7};
    }
  }
  
  Build(chain_idx, chain_lengths, pdims, pratioes, params);
}

void AlexnetBuilder::Build(
    const int chain_idx,
    const std::vector<int> chain_lengths,
    const std::vector<int> pdims,
    const std::vector<float> pratioes,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  int op_idx = 0;
  APPEND_CONV2D(224, 224, 3, 96, 11, 11, 4, 4, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(59, 59, 96, 256, 5, 5, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(54, 54, 256, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(29, 29, 256, 384, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(26, 26, 384, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(15, 15, 384, 384, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(15, 15, 384, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(12, 12, 256, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_FULLY_CONNECTED(7, 7, 256, 4096, 4, pratioes[op_idx]);
  APPEND_FULLY_CONNECTED(1, 1, 4096, 4096, 4, pratioes[op_idx]);
  APPEND_FULLY_CONNECTED(1, 1, 4096, 1000, 4, pratioes[op_idx]);

  if (chain_lengths.size() > 0) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    if (chain_idx > -1 && chain_idx < static_cast<int>(chain_lengths.size())) {
      LOG(INFO) << "===== Build AlexNet Info =====";
      LOG(INFO) << "chain_idx " << chain_idx
                << ", chain_lengths " << chain_lengths[chain_idx];
      size_t start_op_idx = 0;
      for (int i = 0; i < chain_idx; i ++) {
        start_op_idx += chain_lengths[i];
      }
      for (int i = 0; i < chain_lengths[chain_idx]; i ++) {
        out_params.emplace_back(params[start_op_idx + i]);
      }
    }
    params = out_params;
  }
}

void AlexnetBuilder::Build(
    const int chain_idx,
    const int chain_param_hint,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  std::vector<int> chain_lengths;
  std::vector<int> pdims;
  std::vector<float> pratioes;
  if (chain_param_hint == 0) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.0);
  } else if (chain_param_hint == 1) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 1.0);
  } else if (chain_param_hint == 2) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 10) {
    pdims = {1, 1, 1,
             1, 1,
             1, 1, 1};
    pratioes = {1.0, 0.8, 1.0,
                0.6, 1.0,
                1.0, 1.0, 1.0};
  } else if (chain_param_hint == 11) {
    chain_lengths = {3, 3, 1, 1};
    pdims = {1, 1, 1,
             1, 1,
             1, 1, 1};
    pratioes = {0.8, 0.8, 0.8,
                0.4, 0.4,
                0.4, 1.0, 1.0};
  }
  
  Build(chain_idx, chain_lengths, pdims, pratioes, params);
}

void Vgg16Builder::Build(
    const int chain_idx,
    const std::vector<int> chain_lengths,
    const std::vector<int> pdims,
    const std::vector<float> pratioes,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  int op_idx = 0;
  double size_factor = 1.0;
  double ch_factor = 1.0;
  APPEND_CONV2D(226, 226, 3, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(226, 226, 64, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(224 * size_factor, 224 * size_factor, 64, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 3;
  APPEND_CONV2D(114, 114, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(114, 114, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(112 * size_factor, 112 * size_factor, 128, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 6;
  APPEND_CONV2D(58, 58, 128, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(58, 58, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(58, 58, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(56 * size_factor, 56 * size_factor, 256, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 10;
  APPEND_CONV2D(30, 30, 256, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(30, 30, 512, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(30, 30, 512, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(28 * size_factor, 28 * size_factor, 512, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 14;
  APPEND_CONV2D(16, 16, 512 * ch_factor, 512 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(16, 16, 512 * ch_factor, 512 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(16, 16, 512 * ch_factor, 512 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(14 * size_factor, 14 * size_factor, 512, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 18;
  APPEND_FULLY_CONNECTED(7, 7, 512, 4096, 4, pratioes[op_idx]);
  APPEND_FULLY_CONNECTED(1, 1, 4096, 4096, 4, pratioes[op_idx]);
  APPEND_FULLY_CONNECTED(1, 1, 4096, 1000, 4, pratioes[op_idx]);
  
  if (chain_lengths.size() > 0) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    if (chain_idx > -1 && chain_idx < static_cast<int>(chain_lengths.size())) {
      LOG(INFO) << "===== Build VGG-16 Info =====";
      LOG(INFO) << "chain_idx " << chain_idx
                << ", chain_lengths " << chain_lengths[chain_idx];
      size_t start_op_idx = 0;
      for (int i = 0; i < chain_idx; i ++) {
        start_op_idx += chain_lengths[i];
      }
      for (int i = 0; i < chain_lengths[chain_idx]; i ++) {
        out_params.emplace_back(params[start_op_idx + i]);
      }
    }
    params = out_params;
  }
}

void Vgg16Builder::Build(
    const int chain_idx,
    const int chain_param_hint,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  std::vector<int> chain_lengths;
  std::vector<int> pdims;
  std::vector<float> pratioes;
  if (chain_param_hint == 0) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.0);
  } else if (chain_param_hint == 1) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 1.0);
  } else if (chain_param_hint == 2) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 3) {
    pdims = std::vector<int>(op_count_, 4);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 4) {
    chain_lengths = {6, 4, 4, 1, 1, 2, 1, 1, 1, 1};
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
    //pratioes = {1,0.6,1,0.7,0.5,1,0.5,0.5,0.5,1,0.5,0.4,0.4,1,0.6,0.7,0.6,0.9,0.5,0.9,0.4};
    //pratioes = {1,0.6,1,0.7,0.5,1,0.6,0.5,0.5,1,0.5,0.5,0.5,1,0.5,0.5,0.5,1,0.6,0.2,0.5};
  } else if (chain_param_hint == 5) {
    pdims = std::vector<int>(op_count_, 4);
    //pratioes = {1,0.8,1,0.7,0.6,1,0.6,0.5,0.5,1,0.5,0.5,0.5,1,0.5,0.5,0.5,1,0.6,0.2,0.4};
    pratioes = {1,0.6,1,0.7,0.5,1,0.6,0.5,0.5,1,0.5,0.5,0.5,1,0.5,0.5,0.5,1,0.6,0.2,0.5};
  } else if (chain_param_hint == 6) {
    pdims = std::vector<int>(op_count_, 4);
    pratioes = {};
  } else if (chain_param_hint == 7) {
    pdims = std::vector<int>(op_count_, 4);
    pratioes = {};
  } else if (chain_param_hint == 10) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
      pdims = {1, 1, 1,
               1, 1, 1,
               1, 1, 1, 1,
               1, 1, 1, 1,
               4, 4, 4, 1};
      pratioes = {1.0, 0.6, 1.0,
                  0.7, 0.5, 1.0,
                  0.5, 0.5, 0.5, 1.0,
                  0.5, 0.5, 0.5, 1.0,
                  0.6, 0.5, 0.5, 1.0};

      pdims = {1,1,1,1,1,1,4,1,1,1,1,4,4,1,4,4,4,1,4,4,4};
      pratioes = {1,0.6,1,0.7,0.5,1,0.6,0.5,0.5,1,0.5,0.5,0.5,1,0.5,0.5,0.5,1,0.6,0.2,0.5};

      pdims = {1,1,1,1,1,4,1,4,4,1,1,4,4,1,4,4,4,1,4,4,4};
      pratioes = std::vector<float>(op_count_, 0.5);
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
      pdims = {1,1,1,1,1,1,1,1,1,1,4,1,4,1,4,4,4,1,4,4,4};
      pratioes = {1,0.6,1,0.7,0.5,1,0.5,0.5,0.5,1,0.5,0.4,0.5,1,0.5,0.5,0.5,1,0.5,0.2,0.1};
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
      pdims = {1,1,1,1,1,1,4,1,1,1,4,4,1,1,4,4,4,1,4,4,4};
      pratioes = {1,0.6,1,0.6,0.5,1,0.6,0.5,0.5,1,0.5,0.5,0.5,1,0.5,0.5,0.5,1,0.5,0.2,0.1};
    } else if (!codl_config->soc_name().compare("Kirin990")) {
      pdims = std::vector<int>(op_count_, 1);
      pratioes = {0.4,0.4,1.0,0.4,0.3,1.0,0.3,0.3,0.3,1.0,
                  0.3,0.3,0.3,1.0,0.3,0.3,1.0,1.0,0.2,0.1,
                  0.2};
    }
  } else if (chain_param_hint == 11) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
      chain_lengths = {2, 3, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1};
      pdims = {1, 1, 1,
               1, 1, 1,
               1, 1, 1, 1,
               1, 1, 1, 1,
               4, 4, 4, 1};
      pratioes = {0.6, 0.6, 0.5,
                  0.5, 0.5, 1.0,
                  0.5, 0.3, 0.3, 1.0,
                  0.4, 0.4, 0.5, 1.0,
                  0.6, 0.5, 0.5, 1.0};

      chain_lengths = {1,2,5,2,2,1,1,1,1,1,1,1,1,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,4,1,4,4,4,1,4,4,4};
      pratioes = {1,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.4,0.5,1,0.5,0.5,0.5,1.0,0.6,0.2,0.5};
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
      chain_lengths = {3,5,2,2,2,1,1,1,1,1,1,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,1,4,4,4};
      pratioes = {0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.4,0.3,0.3,0.5,0.5,0.5,1,0.5,0.2,0.1};
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
      chain_lengths = {2,2,1,4,3,2,1,1,1,1,1,1,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,1,4,4,4};
      pratioes = {0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,0.5,0.2,0.1};
    } else if (!codl_config->soc_name().compare("Kirin990")) {
      chain_lengths = std::vector<int>(op_count_, 1);
      pdims = std::vector<int>(op_count_, 1);
      pratioes = {0.4,0.4,1.0,0.4,0.3,1.0,0.3,0.3,0.3,1.0,
                  0.3,0.3,0.3,1.0,0.3,0.3,1.0,1.0,0.2,0.1,
                  0.2};
    }
  } else if (chain_param_hint == 20) {

  }
  
  Build(chain_idx, chain_lengths, pdims, pratioes, params);
}

void FastStyleTransferBuilder::Build(
    const int chain_idx,
    const std::vector<int> chain_lengths,
    const std::vector<int> pdims,
    const std::vector<float> pratioes,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  int op_idx = 0;
  APPEND_CONV2D(480+8, 640+8, 3, 32, 9, 9, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(480+1, 640+1, 32, 64, 3, 3, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(240+1, 320+1, 64, 128, 3, 3, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(120+2, 160+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(120+2, 160+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(120+2, 160+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(120+2, 160+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(120+2, 160+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(120+2, 160+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(120+2, 160+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(120+2, 160+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(120+2, 160+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(120+2, 160+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(480+8, 640+8, 32, 3, 9, 9, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  if (chain_lengths.size() > 0) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    if (chain_idx > -1 && chain_idx < static_cast<int>(chain_lengths.size())) {
      LOG(INFO) << "===== Build Fast Style Transfer Info =====";
      LOG(INFO) << "chain_idx " << chain_idx
                << ", chain_lengths " << chain_lengths[chain_idx];
      size_t start_op_idx = 0;
      for (int i = 0; i < chain_idx; i ++) {
        start_op_idx += chain_lengths[i];
      }
      for (int i = 0; i < chain_lengths[chain_idx]; i ++) {
        out_params.emplace_back(params[start_op_idx + i]);
      }
    }
    params = out_params;
  }
}

void FastStyleTransferBuilder::Build(
    const int chain_idx,
    const int chain_param_hint,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  std::vector<int> chain_lengths;
  std::vector<int> pdims;
  std::vector<float> pratioes;
  if (chain_param_hint == 0) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.0);
  } else if (chain_param_hint == 1) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 1.0);
  } else if (chain_param_hint == 2) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 3) {
    pdims = std::vector<int>(op_count_, 4);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 4) {
    chain_lengths = {2, 1, 3, 3, 4, 1};
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 5) {
    pdims = std::vector<int>(op_count_, 4);
    pratioes = {};
  } else if (chain_param_hint == 10) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
      pdims = {1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1};
      pratioes = {0.6, 0.9, 0.9,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                  1.0};

      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,4};
      pratioes = {0.6,0.9,0.9,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.8};

      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,4};
      pratioes = std::vector<float>(op_count_, 0.5);
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,1};
      pratioes = {0.6,0.9,0.9,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.9};
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,4};
      pratioes = {0.7,0.9,0.9,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.8};
    } else if (!codl_config->soc_name().compare("Kirin990")) {
      pdims = std::vector<int>(op_count_, 1);
      pratioes = {0.4,0.8,0.8,0.4,0.3,0.3,0.3,0.3,0.3,0.3,
                  0.3,0.3,0.3,0.4};
    }
  } else if (chain_param_hint == 11) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
      chain_lengths = {1,1,1,1,1,2,1,1,2,2,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,4};
      pratioes = {0.6,0.9,0.9,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1};
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
      chain_lengths = {1,2,2,1,6,1,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,1};
      pratioes = {0.6,0.9,0.9,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.9};
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
      chain_lengths = {1,2,3,1,1,1,1,1,1,1,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,1,1,1,4};
      pratioes = {0.7,0.9,0.9,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1};
    } else if (!codl_config->soc_name().compare("Kirin990")) {
      chain_lengths = std::vector<int>(op_count_, 1);
      pdims = std::vector<int>(op_count_, 1);
      pratioes = {0.4,0.8,0.8,0.4,0.3,0.3,0.3,0.3,0.3,0.3,
                  0.3,0.3,0.3,0.4};
    }
  }
  
  Build(chain_idx, chain_lengths, pdims, pratioes, params);
}

void RetinaFaceBuilder::Build(
    const int chain_idx,
    const std::vector<int> chain_lengths,
    const std::vector<int> pdims,
    const std::vector<float> pratioes,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  int op_idx = 0;
  double size_factor = 1.0;
  //double size_factor2 = 3.0;
  double ch_factor = 1.0;
  // Resnet-v1-50 as the backbone.
  // chain 0, op count 2
  APPEND_CONV2D(640+6, 640+6, 3, 64, 7, 7, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(320, 320, 64, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);

  // chain 1
  //APPEND_CONV2D(160, 160, 64, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 5
  APPEND_CONV2D(160, 160, 64, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(160+2, 160+2, 64, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(160, 160, 64, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 8
  APPEND_CONV2D(160, 160, 256, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(160+2, 160+2, 64, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(160, 160, 64, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  // chain 2
  //APPEND_POOLING(160, 160, 256, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 11
  APPEND_CONV2D(160, 160, 256, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(160+1, 160+1, 64, 64, 3, 3, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80, 80, 64, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  // chain 3
  //APPEND_CONV2D(80, 80, 256, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 14
  APPEND_CONV2D(80, 80, 256, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80+2, 80+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80, 80, 128, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 17
  APPEND_CONV2D(80, 80, 512, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80+2, 80+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80, 80, 128, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 20
  APPEND_CONV2D(80, 80, 512, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80+2, 80+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80, 80, 128, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  // chain 4
  //APPEND_POOLING(80, 80, 512, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 23
  APPEND_CONV2D(80, 80, 512, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80+1, 80+1, 128, 128, 3, 3, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40, 40, 128, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 5
  //APPEND_CONV2D(40, 40, 512, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 26
  APPEND_CONV2D(40, 40, 512, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40 / size_factor + 2, 40 / size_factor + 2, 256 * ch_factor, 256 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40, 40, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 29
  APPEND_CONV2D(40, 40, 1024, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40 / size_factor + 2, 40 / size_factor + 2, 256 * ch_factor, 256 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40, 40, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 32
  APPEND_CONV2D(40, 40, 1024, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40 / size_factor + 2, 40 / size_factor + 2, 256 * ch_factor, 256 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40, 40, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 35
  APPEND_CONV2D(40, 40, 1024, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40 / size_factor + 2, 40 / size_factor + 2, 256 * ch_factor, 256 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40, 40, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 38
  APPEND_CONV2D(40, 40, 1024, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40 / size_factor + 2, 40 / size_factor + 2, 256 * ch_factor, 256 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40, 40, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  // chain 6
  //APPEND_POOLING(40, 40, 1024, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 41
  APPEND_CONV2D(40, 40, 1024, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40 / size_factor + 2, 40 / size_factor + 2, 256 * ch_factor, 256 * ch_factor, 3, 3, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20, 20, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 7
  //APPEND_CONV2D(20, 20, 1024, 2048, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  // chain 0, op count 44
  APPEND_CONV2D(20, 20, 1024, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20 / size_factor + 2, 20 / size_factor + 2, 512 * ch_factor, 512 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20, 20, 512, 2048, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 47
  APPEND_CONV2D(20, 20, 2048, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20 / size_factor + 2, 20, 512 * ch_factor, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20, 20, 512, 2048, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 50
  APPEND_CONV2D(20, 20, 2048, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20 / size_factor + 2, 20 / size_factor + 2, 512 * ch_factor, 512 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20, 20, 512, 2048, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 51
  APPEND_CONV2D(20, 20, 2048, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);



  APPEND_CONV2D(40+2, 40+2, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40, 40, 512, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  APPEND_CONV2D(80+2, 80+2, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80, 80, 256, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  APPEND_CONV2D(160+2, 160+2, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(160, 160, 256, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  APPEND_CONV2D(160+2, 160+2, 256, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(160+2, 160+2, 256, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(160+2, 160+2, 256, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(160+2, 160+2, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(160+2, 160+2, 256, 12, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);

  APPEND_CONV2D(80+2, 80+2, 256, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80+2, 80+2, 256, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80+2, 80+2, 256, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80+2, 80+2, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80+2, 80+2, 256, 12, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);

  APPEND_CONV2D(40+2, 40+2, 256 * ch_factor, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40+2, 40+2, 256 * ch_factor, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40+2, 40+2, 256 * ch_factor, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40+2, 40+2, 256 * ch_factor, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40+2, 40+2, 256 * ch_factor, 12, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);

  APPEND_CONV2D(20+2, 20+2, 256 * ch_factor, 128 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20+2, 20+2, 256 * ch_factor, 64 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20+2, 20+2, 256 * ch_factor, 64 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20+2, 20+2, 256 * ch_factor, 256 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20+2, 20+2, 256 * ch_factor, 12 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);

  APPEND_CONV2D(10+2, 10+2, 256 * ch_factor, 128 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(10+2, 10+2, 256 * ch_factor, 64 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(10+2, 10+2, 256 * ch_factor, 64 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(10+2, 10+2, 256 * ch_factor, 256 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(10+2, 10+2, 256 * ch_factor, 12 * ch_factor, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);

  APPEND_CONV2D(160+2, 160+2, 256, 6, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(80+2, 80+2, 256, 6, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(40+2, 40+2, 256, 6, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(20+2, 20+2, 256, 6, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(10+2, 10+2, 256, 6, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);

  if (chain_lengths.size() > 0) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    if (chain_idx > -1 && chain_idx < static_cast<int>(chain_lengths.size())) {
      LOG(INFO) << "===== Build Retina Face Info =====";
      LOG(INFO) << "chain_idx " << chain_idx
                << ", chain_lengths " << chain_lengths[chain_idx];
      size_t start_op_idx = 0;
      for (int i = 0; i < chain_idx; i ++) {
        start_op_idx += chain_lengths[i];
      }
      for (int i = 0; i < chain_lengths[chain_idx]; i ++) {
        out_params.emplace_back(params[start_op_idx + i]);
      }
    }
    params = out_params;
  }
}

void RetinaFaceBuilder::Build(
    const int chain_idx,
    const int chain_param_hint,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  std::vector<int> chain_lengths;
  std::vector<int> pdims;
  std::vector<float> pratioes;
  if (chain_param_hint == 0) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.0);
  } else if (chain_param_hint == 1) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 1.0);
  } else if (chain_param_hint == 2) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 3) {
    pdims = std::vector<int>(op_count_, 4);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 10) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
      pdims = {1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,4,1,
               1,1,1,4,1,1,1,1,4,1,
               1,1,1,1,1,1,1,1,4,1,
               1,1,4,1};
      pratioes = {0.7,1,1,1,0.6,1,1,0.7,1,1,1,1,1,0.7,1,0.6,1,1,0.6,1,1,0.6,1,1,
                  1,1,1,0.7,1,0.6,1,1,0.6,1,1,0.6,1,1,0.6,1,1,0.6,1,1,1,1,1,0.7,
                  1,0.5,1,1,0.5,1,1,0.5,1,1,0.6,1,0.5,1,0.3,0.7,0.5,0.6,0.6,0.3,
                  0.8,0.5,0.6,0.6,0.5,0.8,0.7,1,1,0.6,0.8,1,1,1,1,1,1,1,1,1,0.8,
                  1,1,1,0.9,1};

      pdims = {1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,4,1,1,4,1,1,
               4,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,4,1,1,1,
               1,4,1,1,4,1,1};
      pratioes = {0.7,1,1,0.6,1,1,0.7,1,1,1,
                  1,1,0.6,1,1,0.6,1,1,0.5,1,
                  1,1,1,1,0.5,1,1,0.5,1,1,
                  0.5,1,1,0.6,1,1,0.6,1,1,1,
                  1,1,0.5,1,1,0.6,1,1,0.6,1,
                  1,0.5,1,0.5,1,0.3,0.7,0.5,0.6,0.6,
                  0.3,0.9,0.5,0.7,0.6,0.5,1,0.7,1,1,
                  0.6,0.8,1,1,1,1,0.8,1,1,1,
                  1,0.9,1,1,0.9,1,1};
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
      pdims = {1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,4,1,1,4,1,1,
               1,1,1,4,1,1,4,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,4,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1};
      pratioes = {0.7,1,1,0.7,1,1,0.6,1,1,1,
                  1,1,0.6,1,1,0.5,1,1,0.6,1,
                  1,1,1,1,0.5,0.6,1,0.5,0.7,0.7,
                  0.5,0.6,0.7,0.5,1,0.7,0.5,1,1,1,
                  1,0.6,0.4,0.6,0.7,0.4,1,0.7,0.4,0.6,
                  0.7,0.5,1,0.4,0.7,0.3,0.7,0.5,0.5,0.5,
                  0.3,0.9,0.5,0.6,0.6,0.5,1,0.6,0.7,0.8,
                  0.6,1,1,1,1,1,1,1,1,1,
                  1,1,0.9,1,1,1,1};
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
      pdims = {1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,4,1,1,
               4,1,1,4,1,1,4,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               4,1,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1};
      pratioes = {0.7,1,1,0.6,1,1,0.7,1,1,1,
                  1,1,0.6,1,0.7,0.6,1,1,0.6,1,
                  1,1,1,1,0.5,1,1,0.6,1,0.7,
                  0.6,1,0.7,0.6,1,0.7,0.6,0.7,1,1,
                  1,1,0.5,1,1,0.6,1,1,0.5,0.7,
                  1,0.5,1,0.4,0.7,0.4,0.8,0.5,0.6,0.6,
                  0.4,0.9,0.5,0.6,0.7,0.4,1,1,1,1,
                  0.6,1,1,1,1,1,1,1,1,1,
                  1,0.9,0.8,1,1,1,1};
    } else if (!codl_config->soc_name().compare("Kirin990")) {
      pdims = std::vector<int>(op_count_, 1);
      pratioes = {0.4,1.0,0.6,0.5,0.6,0.6,0.4,0.6,0.6,0.7,
                  0.5,0.5,0.3,0.5,0.5,0.4,0.5,0.5,0.3,0.5,
                  0.5,0.6,0.4,0.4,0.3,0.4,0.5,0.3,0.4,0.5,
                  0.3,0.4,0.4,0.3,0.4,0.4,0.3,0.4,0.4,0.6,
                  0.4,0.4,0.3,0.4,0.4,0.3,0.4,0.4,0.3,0.4,
                  0.4,1.0,0.4,1.0,0.5,0.3,0.6,0.3,0.4,0.4,
                  0.3,0.4,0.3,0.4,0.4,0.3,0.4,0.3,0.3,0.3,
                  0.3,0.4,0.4,0.3,0.3,0.3,0.2,0.1,0.1,0.1,
                  0.2,0.1,1.0,0.1,0.2,0.1,1.0};
    }
  } else if (chain_param_hint == 11) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
      chain_lengths = {18,21,10,2,1,1,1,1,1,1,1,1,1,  // 60
                       1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,4,1,1,1,
               1,4,1,1,4,1,1};
      pratioes = {0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,
                  0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.6,0.6,
                  0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,
                  0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,1,
                  1,1,1,1,1,1,1,1,1,0.7,
                  0.7,0.5,1,0.5,1,0.3,0.7,0.5,0.6,0.6,
                  0.3,0.9,0.5,0.7,0.6,0.5,1,0.7,1,1,
                  0.6,0.8,1,1,1,1,0.8,1,1,1,
                  1,0.9,1,1,0.9,1,1};
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
      chain_lengths = {18,11,10,3,6,3,1,1,1,1,1,1,1,1,1,  // 60
                       1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,4,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1};
      pratioes = {0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,
                  0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.6,0.6,
                  0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,
                  0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,1,
                  1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
                  0.5,0.5,1,0.4,0.7,0.3,0.7,0.5,0.5,0.5,
                  0.3,0.9,0.5,0.6,0.6,0.5,1,0.6,0.7,0.8,
                  0.6,1,1,1,1,1,1,1,1,1,
                  1,1,0.9,1,1,1,1};
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
      chain_lengths = {1,2,6,3,7,2,1,1,6,6,4,3,3,3,3,1,1,1,1,1,1,1,1,1,  // 60
                       1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1};
      pdims = {1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,4,1,1,1,
               1,1,1,1,1,1,1,1,1,1,
               1,4,1,1,1,1,1};
      pratioes = {0.7,0.8,0.8,0.7,0.7,0.7,0.7,0.7,0.7,1,
                  1,1,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.7,
                  0.7,1,1,0.6,0.6,0.6,0.6,0.6,0.6,0.6,
                  0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.8,
                  0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,
                  0.6,0.5,1,0.4,0.7,0.3,0.7,0.5,0.5,0.5,
                  0.3,0.9,0.5,0.6,0.6,0.5,1,0.6,0.7,0.8,
                  0.6,1,1,1,1,1,1,1,1,1,
                  1,1,0.9,1,1,1,1};
    } else if (!codl_config->soc_name().compare("Kirin990")) {
      chain_lengths = std::vector<int>(op_count_, 1);
      pdims = std::vector<int>(op_count_, 1);
      pratioes = {0.4,1.0,0.6,0.5,0.6,0.6,0.4,0.6,0.6,0.7,
                  0.5,0.5,0.3,0.5,0.5,0.4,0.5,0.5,0.3,0.5,
                  0.5,0.6,0.4,0.4,0.3,0.4,0.5,0.3,0.4,0.5,
                  0.3,0.4,0.4,0.3,0.4,0.4,0.3,0.4,0.4,0.6,
                  0.4,0.4,0.3,0.4,0.4,0.3,0.4,0.4,0.3,0.4,
                  0.4,1.0,0.4,1.0,0.5,0.3,0.6,0.3,0.4,0.4,
                  0.3,0.4,0.3,0.4,0.4,0.3,0.4,0.3,0.3,0.3,
                  0.3,0.4,0.4,0.3,0.3,0.3,0.2,0.1,0.1,0.1,
                  0.2,0.1,1.0,0.1,0.2,0.1,1.0};
    }
  }
  
  Build(chain_idx, chain_lengths, pdims, pratioes, params);
}

void MobileNetV1Builder::Build(
    const int chain_idx,
    const std::vector<int> chain_lengths,
    const std::vector<int> pdims,
    const std::vector<float> pratioes,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  int op_idx = 0;
  APPEND_CONV2D(224+2, 224+2, 3, 32, 3, 3, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(112, 112, 32, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(56, 56, 64, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(56, 56, 128, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(28, 28, 128, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(28, 28, 256, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(14, 14, 256, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(14, 14, 512, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(14, 14, 512, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(14, 14, 512, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(14, 14, 512, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(14, 14, 512, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(7, 7, 512, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(7, 7, 1024, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_POOLING(7, 7, 1024, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(1, 1, 1024, 1001, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  if (chain_lengths.size() > 0) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    if (chain_idx > -1 && chain_idx < static_cast<int>(chain_lengths.size())) {
      LOG(INFO) << "===== Build MobileNetV1 Info =====";
      LOG(INFO) << "chain_idx " << chain_idx
                << ", chain_lengths " << chain_lengths[chain_idx];
      size_t start_op_idx = 0;
      for (int i = 0; i < chain_idx; i ++) {
        start_op_idx += chain_lengths[i];
      }
      for (int i = 0; i < chain_lengths[chain_idx]; i ++) {
        out_params.emplace_back(params[start_op_idx + i]);
      }
    }
    params = out_params;
  }
}

void MobileNetV1Builder::Build(
    const int chain_idx,
    const int chain_param_hint,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  std::vector<int> chain_lengths;
  std::vector<int> pdims;
  std::vector<float> pratioes;
  if (chain_param_hint == 0) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.0);
  } else if (chain_param_hint == 1) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 1.0);
  } else if (chain_param_hint == 2) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 3) {
    pdims = std::vector<int>(op_count_, 4);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 10) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
    } else if (!codl_config->soc_name().compare("Kirin990")) {
    }
  } else if (chain_param_hint == 11) {
    mace::utils::CodlConfig *codl_config = mace::utils::GetGlobalCodlConfig();
    if (!codl_config->soc_name().compare("Snapdragon855")) {
    } else if (!codl_config->soc_name().compare("Snapdragon865")) {
    } else if (!codl_config->soc_name().compare("Snapdragon888")) {
    }
  }
  
  Build(chain_idx, chain_lengths, pdims, pratioes, params);
}

void Resnet50v1Builder::Build(
    const int chain_idx,
    const std::vector<int> chain_lengths,
    const std::vector<int> pdims,
    const std::vector<float> pratioes,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  int op_idx = 0;
  int in_size = 640;
  // chain 0, op count 2
  APPEND_CONV2D(in_size+6, in_size+6, 3, 64, 7, 7, 2, 2, pdims[op_idx], pratioes[op_idx]);
  in_size /= 2;
  APPEND_POOLING(in_size, in_size, 64, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);

  in_size /= 2;
  // chain 1
  //APPEND_CONV2D(in_size, in_size, 64, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 5
  APPEND_CONV2D(in_size, in_size, 64, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 64, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 64, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 8
  APPEND_CONV2D(in_size, in_size, 256, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 64, 64, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 64, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  // chain 2
  //APPEND_POOLING(in_size, in_size, 256, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 11
  APPEND_CONV2D(in_size, in_size, 256, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+1, in_size+1, 64, 64, 3, 3, 2, 2, pdims[op_idx], pratioes[op_idx]);
  in_size /= 2;
  APPEND_CONV2D(in_size, in_size, 64, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  // chain 3
  //APPEND_CONV2D(in_size, in_size, 256, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 14
  APPEND_CONV2D(in_size, in_size, 256, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 128, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 17
  APPEND_CONV2D(in_size, in_size, 512, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 128, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 20
  APPEND_CONV2D(in_size, in_size, 512, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 128, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 128, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  // chain 4
  //APPEND_POOLING(in_size, in_size, 512, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 23
  APPEND_CONV2D(in_size, in_size, 512, 128, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+1, in_size+1, 128, 128, 3, 3, 2, 2, pdims[op_idx], pratioes[op_idx]);
  in_size /= 2;
  APPEND_CONV2D(in_size, in_size, 128, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 5
  //APPEND_CONV2D(in_size, in_size, 512, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 26
  APPEND_CONV2D(in_size, in_size, 512, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 29
  APPEND_CONV2D(in_size, in_size, 1024, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 32
  APPEND_CONV2D(in_size, in_size, 1024, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 35
  APPEND_CONV2D(in_size, in_size, 1024, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 38
  APPEND_CONV2D(in_size, in_size, 1024, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 256, 256, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  // chain 6
  //APPEND_POOLING(in_size, in_size, 1024, 2, 2, 2, 2, 2, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 41
  APPEND_CONV2D(in_size, in_size, 1024, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+1, in_size+1, 256, 256, 3, 3, 2, 2, pdims[op_idx], pratioes[op_idx]);
  in_size /= 2;
  APPEND_CONV2D(in_size, in_size, 256, 1024, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 7
  //APPEND_CONV2D(in_size, in_size, 1024, in_size48, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  // chain 0, op count 44
  APPEND_CONV2D(in_size, in_size, 1024, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 512, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 512, 2048, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 47
  APPEND_CONV2D(in_size, in_size, 2048, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 512, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 512, 2048, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 50
  APPEND_CONV2D(in_size, in_size, 2048, 512, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size+2, in_size+2, 512, 512, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(in_size, in_size, 512, 2048, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  // chain 0, op count 51
  APPEND_CONV2D(in_size, in_size, 2048, 256, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);

  if (chain_lengths.size() > 0) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    if (chain_idx > -1 && chain_idx < static_cast<int>(chain_lengths.size())) {
      LOG(INFO) << "===== Build Retina Face Info =====";
      LOG(INFO) << "chain_idx " << chain_idx
                << ", chain_lengths " << chain_lengths[chain_idx];
      size_t start_op_idx = 0;
      for (int i = 0; i < chain_idx; i ++) {
        start_op_idx += chain_lengths[i];
      }
      for (int i = 0; i < chain_lengths[chain_idx]; i ++) {
        out_params.emplace_back(params[start_op_idx + i]);
      }
    }
    params = out_params;
  }
}

void Resnet50v1Builder::Build(
    const int chain_idx,
    const int chain_param_hint,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  std::vector<int> chain_lengths;
  std::vector<int> pdims;
  std::vector<float> pratioes;
  if (chain_param_hint == 0) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.0);
  } else if (chain_param_hint == 1) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 1.0);
  } else if (chain_param_hint == 2) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 3) {
    pdims = std::vector<int>(op_count_, 4);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 4) {
    chain_lengths = {1,3,2,3,8,2,1,1,2,2,2,1,4,1,1,2,1,3,2,1,1,1,2,1,2,1};
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 10) {
    pdims = {0};
    pratioes = {0};
  } else if (chain_param_hint == 11) {
    chain_lengths = {0};
    pdims = {0};
    pratioes = {0};
  }
  
  Build(chain_idx, chain_lengths, pdims, pratioes, params);
}

void BertBuilder::Build(
    const int chain_idx,
    const std::vector<int> chain_lengths,
    const std::vector<int> pdims,
    const std::vector<float> pratioes,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  int op_idx = 0;
  APPEND_MATMUL(1, 256, config_.H, 2, false, false, pdims[op_idx], pratioes[op_idx]);
  APPEND_MATMUL(1, 256, config_.H, config_.H, false, false, pdims[op_idx], pratioes[op_idx]);
  APPEND_MATMUL(config_.A, 256, 256, 64, false, false, pdims[op_idx], pratioes[op_idx]);
  APPEND_MATMUL(config_.A, 256, 64, 256, false, true, pdims[op_idx], pratioes[op_idx]);
  APPEND_MATMUL(1, 256, config_.H * 4, config_.H, false, false, pdims[op_idx], pratioes[op_idx]);
  APPEND_MATMUL(1, 256, config_.H, config_.H * 4, false, false, pdims[op_idx], pratioes[op_idx]);
  // NOTE(fucheng): These sizes are not in BERT model, just for evaluation.
  APPEND_MATMUL(1, 256 * 4, config_.H, config_.H * 4, false, false, pdims[op_idx], pratioes[op_idx]);
  APPEND_MATMUL(1, 256 * 4, config_.H * 4, config_.H * 4, false, false, pdims[op_idx], pratioes[op_idx]);
  APPEND_MATMUL(1, config_.H * 4, config_.H * 4, config_.H * 4, false, false, pdims[op_idx], pratioes[op_idx]);

  if (chain_lengths.size() > 0) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    if (chain_idx > -1 && chain_idx < static_cast<int>(chain_lengths.size())) {
      LOG(INFO) << "===== Build BERT Info =====";
      LOG(INFO) << "chain_idx " << chain_idx
                << ", chain_lengths " << chain_lengths[chain_idx];
      size_t start_op_idx = 0;
      for (int i = 0; i < chain_idx; i ++) {
        start_op_idx += chain_lengths[i];
      }
      for (int i = 0; i < chain_lengths[chain_idx]; i ++) {
        out_params.emplace_back(params[start_op_idx + i]);
      }
    }
    params = out_params;
  }
}

void BertBuilder::Build(
    const int chain_idx,
    const int chain_param_hint,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  std::vector<int> chain_lengths;
  std::vector<int> pdims;
  std::vector<float> pratioes;
  if (chain_param_hint == 0) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.0);
  } else if (chain_param_hint == 1) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 1.0);
  } else if (chain_param_hint == 2) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 10) {
    pdims = {0};
    pratioes = {0};
  } else if (chain_param_hint == 11) {
    chain_lengths = {0};
    pdims = {0};
    pratioes = {0};
  }
  
  Build(chain_idx, chain_lengths, pdims, pratioes, params);
}

void OpChainNetBuilder::Build(
    const int chain_idx,
    const std::vector<int> chain_lengths,
    const std::vector<int> pdims,
    const std::vector<float> pratioes,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  int op_idx = 0;
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 10;
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  // op_idx = 20;
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdims[op_idx], pratioes[op_idx]);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdims[op_idx], pratioes[op_idx]);
  
  if (chain_lengths.size() > 0) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    if (chain_idx > -1 && chain_idx < static_cast<int>(chain_lengths.size())) {
      LOG(INFO) << "===== Build VGG-16 Info =====";
      LOG(INFO) << "chain_idx " << chain_idx
                << ", chain_lengths " << chain_lengths[chain_idx];
      size_t start_op_idx = 0;
      for (int i = 0; i < chain_idx; i ++) {
        start_op_idx += chain_lengths[i];
      }
      for (int i = 0; i < chain_lengths[chain_idx]; i ++) {
        out_params.emplace_back(params[start_op_idx + i]);
      }
    }
    params = out_params;
  }
}

void OpChainNetBuilder::Build(
    const int chain_idx,
    const int chain_param_hint,
    std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  std::vector<int> chain_lengths;
  std::vector<int> pdims;
  std::vector<float> pratioes;
  if (chain_param_hint == 2) {
    pdims = std::vector<int>(op_count_, 1);
    pratioes = std::vector<float>(op_count_, 0.5);
  } else if (chain_param_hint == 10) {
  } else if (chain_param_hint == 11) {
  }
  
  Build(chain_idx, chain_lengths, pdims, pratioes, params);
}

#undef APPEND_MATMUL
#undef APPEND_POOLING
#undef APPEND_CONV2D
