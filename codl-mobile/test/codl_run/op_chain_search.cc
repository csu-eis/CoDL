
#include "test/codl_run/op_chain_search.h"
#include "test/codl_run/op_chain_executor.h"
#include "test/codl_run/op_chain_helper.h"
#include "test/codl_run/op_chain_latency_predict.h"

namespace mace {

constexpr int kStartRoundIdx = 10;
constexpr int kRounds = 10;
constexpr int kMaxSize = 65535;
constexpr double kMaxDouble = kMaxSize;
constexpr double kMinDouble = -1;

std::string LatencyAcquirementToString(const LatencyAcquirement acq) {
  switch (acq) {
    case LA_PROFILING: return "Profiling";
    case LA_PREDICTION: return "Prediction";
    default: return "None";
  }
}

template<typename T>
void SetArrayValue(T *arr, const size_t arr_size, const T v) {
  for (size_t i = 0; i < arr_size; i ++) {
    arr[i] = v;
  }
}

int FreeOpChain(CodlOpTaskChain &op_chain) {
  op_chain.Destroy(DESTROY_TYPE_SOFT);
  return 0;
}

int FreeOpChains(std::vector<CodlOpTaskChain> &op_chains) {
  for (auto &chain : op_chains) {
    chain.Destroy(DESTROY_TYPE_SOFT);
  }
  return 0;
}

int ChainSearch::OptimalPartitionProfileOrPredict(
    const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
    const LatencyAcquirement acq,
    const LPBackend lp_backend,
    const bool profile_data_transform,
    const bool profile_compute,
    const int pdim_hint,
    const int pratio_hint,
    const int debug_level,
    std::vector<std::shared_ptr<CodlOpChainParam>> &out_op_params) {
  if (op_params.size() == 0) {
    return 0;
  }

  LOG(INFO) << "Latency acquirement: " << LatencyAcquirementToString(acq);

  OpChainLatencyPredictor chain_predictor;
  if (acq == LA_PREDICTION) {
    LPInitContext lp_init_context;
    lp_init_context.set_backend(lp_backend);
    chain_predictor.Init(&lp_init_context);
  }

  out_op_params = op_params;
  
  // Profiling to find optimal partition dimension and ratio.
  std::vector<int> dim_candidates = {1, 4};
  if (pdim_hint == 1) {
    dim_candidates = {1};
  } else if (pdim_hint == 4) {
    dim_candidates = {4};
  }
  if (op_params[0]->gpu_mtype() == MemoryType::GPU_BUFFER) {
    dim_candidates = {4};
  }
  
  std::vector<float> ratio_candidates;
  if (pratio_hint == 0) {
    ratio_candidates = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  } else if (pratio_hint == 1) {
    ratio_candidates = {0.5};
  } else if (pratio_hint == 2) {
    const float epsi = 0.3;
    const int epsi_count = epsi * 10;
    ratio_candidates.push_back(0.5);
    for (int i = 1; i <= epsi_count; i ++) {
      ratio_candidates.push_back(0.5 + ((float) i / 10));
      ratio_candidates.push_back(0.5 - ((float) i / 10));
    }
  }

  if (debug_level >= 2) {
    LOG(INFO) << "ratio_candidates " << VectorToString<float>(ratio_candidates);
  }

  if (debug_level >= 1) {
    LOG(INFO) << "===== Partition Profiling Info =====";
  }

  // Add an OP for warming up.
  {
    CodlOpTaskChain chain;
    chain.Append(out_op_params[0].get());
    FreeOpChain(chain);
    chain.Clear();
  }

  int64_t t0;
  int predict_times = 0;
  double predict_duration = 0;
  for (size_t i = 0; i < op_params.size(); i ++) {
    size_t opt_d_idx = 1;
    size_t opt_r_idx = 0;
    double min_lat = kMaxDouble;
    const CodlOpType op_type = op_params[i]->op_type();
    for (size_t d_idx = 0; d_idx < dim_candidates.size(); d_idx ++) {
      for (size_t r_idx = 0; r_idx < ratio_candidates.size(); r_idx ++) {
        const int dim = dim_candidates[d_idx];
        const float ratio = ratio_candidates[r_idx];
        out_op_params[i]->set_part_dim(dim);
        out_op_params[i]->set_part_ratio(ratio);
        
        CodlOpTaskChain chain;
        chain.Append(out_op_params[i].get());
        
        double lat = 0;
        bool profile_flags = !profile_data_transform && profile_compute;
        if (acq == LA_PROFILING) {
          chain.SerialRun(kStartRoundIdx, kRounds, profile_flags, &lat);
        } else if (acq == LA_PREDICTION) {
          t0 = NowMicros();
          chain_predictor.Predict(chain, debug_level, &lat);
          predict_duration += (NowMicros() - t0) / 1000.0;
          predict_times ++;
        }
        
        if (lat < min_lat) {
          opt_d_idx = d_idx;
          opt_r_idx = r_idx;
          min_lat = lat;
        }

        if (debug_level >= 1) {
          LOG(INFO) << "op " << i << ", type " << CodlOpTypeToString(op_type)
                    << ", d " << dim << ", r " << ratio << ", lat " << lat << " ms"
                    << ", opt_d " << dim_candidates[opt_d_idx]
                    << ", opt_r " << ratio_candidates[opt_r_idx]
                    << ", min_lat " << min_lat << " ms";
        }

        FreeOpChain(chain);
        chain.Clear();
      }
      
      out_op_params[i]->set_part_dim(dim_candidates[opt_d_idx]);
      out_op_params[i]->set_part_ratio(ratio_candidates[opt_r_idx]);
    }
  }

  std::vector<int> opt_dims;
  std::vector<float> opt_ratioes;
  for (size_t i = 0; i < out_op_params.size(); i ++) {
    if (dim_candidates.size() == 2 &&
        out_op_params[i]->part_ratio() == 1.0 &&
        out_op_params[i]->part_dim() == 4) {
      out_op_params[i]->set_part_dim(1);
    }
    const CodlOpType op_type = out_op_params[i]->op_type();
    const int opt_dim = out_op_params[i]->part_dim();
    const float opt_ratio = out_op_params[i]->part_ratio();
    if (debug_level >= 1) {
      LOG(INFO) << "op " << i << ", type " << CodlOpTypeToString(op_type)
                << ", opt_dim " << opt_dim << ", opt_ratio " << opt_ratio;
    }
    opt_dims.push_back(opt_dim);
    opt_ratioes.push_back(opt_ratio);
  }
  if (debug_level >= 1) {
    LOG(INFO) << "opt_dims " << VectorToString<int>(opt_dims);
    LOG(INFO) << "opt_ratioes " << VectorToString<float>(opt_ratioes);
  }

  LOG(INFO) << "Build partition plan, op count " << op_params.size()
            << ", prediction times " << predict_times
            << ", time cost " << predict_duration << " ms";

  return 0;
}

int ChainSearch::Serial(
    const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
    std::vector<CodlOpTaskChain> &op_chains) {
  for (size_t i = 0; i < op_params.size(); i ++) {
    CodlOpTaskChain chain;
    chain.Append(op_params[i].get());
    op_chains.push_back(chain);
    chain.Clear();
  }
  
  return 0;
}

int ChainSearch::Heuristic(
    const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
    std::vector<CodlOpTaskChain> &op_chains) {
  for (size_t i = 0; i < op_params.size(); i ++) {
    CodlOpTaskChain chain;
    chain.Append(op_params[i].get());

    for (size_t j = i + 1; j < op_params.size(); j ++) {
      if (op_params[j]->part_dim() == 1 &&
          op_params[j]->part_ratio() == op_params[i]->part_ratio()) {
        chain.Append(op_params[j].get());
        i ++;
      } else {
        break;
      }
    }

    op_chains.push_back(chain);
  }

  return 0;
}

enum GreedyBaseline {
  BASELINE_SERIAL = 0,
  BASELINE_HEURISTIC = 1
};

std::string GreedyBaselineToString(const GreedyBaseline baseline) {
  switch (baseline) {
    case BASELINE_SERIAL: return "Serial";
    case BASELINE_HEURISTIC: return "Heuristic";
    default: return "Unknown";
  }
}

constexpr int kMaxChainLength = 30;

int ChainSearch::Greedy(
    const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
    const LatencyAcquirement acq,
    const int baseline_idx,
    const int pratio_hint,
    const int debug_level,
    std::vector<CodlOpTaskChain> &op_chains) {
  MACE_CHECK(op_params.size() < 100);

  LOG(INFO) << "Latency acquirement: " << LatencyAcquirementToString(acq);

  OpChainLatencyPredictor chain_predictor;
  if (acq == LA_PREDICTION) {
    LPInitContext lp_init_context;
    chain_predictor.Init(&lp_init_context);
  }

  std::vector<float> ratio_candidates;
  if (pratio_hint == 0) {
    ratio_candidates = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  } else if (pratio_hint == 1) {
    ratio_candidates = {0.5};
  } else if (pratio_hint == 2) {
    const float epsi = 0.3;
    const int epsi_count = epsi * 10;
    ratio_candidates.push_back(0.5);
    for (int i = 1; i <= epsi_count; i ++) {
      ratio_candidates.push_back(0.5 + ((float) i / 10));
      ratio_candidates.push_back(0.5 - ((float) i / 10));
    }
  }

  if (debug_level >= 2) {
    LOG(INFO) << "ratio_candidates " << VectorToString<float>(ratio_candidates);
  }

  std::vector<std::shared_ptr<CodlOpChainParam>> mutable_op_params;
  for (auto &op_param : op_params) {
    mutable_op_params.emplace_back(op_param->Copy());
  }

  const GreedyBaseline baseline = static_cast<GreedyBaseline>(baseline_idx);
  LOG(INFO) << "Greedy baseline: " << GreedyBaselineToString(baseline);
  
  int64_t t0;
  int predict_times = 0;
  int last_chain_identity = -1;
  double predict_duration = 0;
  for (size_t i = 0; i < mutable_op_params.size(); i ++) {
    size_t max_j = i;
    size_t opt_r_idx = 0;
    size_t max_chain_size = 0;
    double min_lat_chain = kMaxDouble;
    double max_lat_gain = kMinDouble;
    double lat_chain, lat_chain_prev, lat_baseline;
    double lat_gain, lat_gain_prev;
    double max_part_lat_gain[op_params.size()];
    CodlOpTaskChain chain;
    CodlOpTaskChain opt_chain;
    CodlOpTaskChain base_chain;
    std::vector<std::shared_ptr<CodlOpChainParam>> serial_op_params;
    std::vector<std::shared_ptr<CodlOpChainParam>> heuristic_op_params;
    std::vector<CodlOpTaskChain> base_op_chains;
    SetArrayValue<double>(max_part_lat_gain, op_params.size(), kMinDouble);
    for (size_t r_idx = 0; r_idx < ratio_candidates.size(); r_idx ++) {
      size_t j = i;
      mutable_op_params[i]->set_part_dim(1);
      mutable_op_params[i]->set_part_ratio(ratio_candidates[r_idx]);
      chain.Append(mutable_op_params[i].get());
      
      if (baseline == BASELINE_SERIAL) {
        //base_chain.Append(op_params[i].get());
        serial_op_params.emplace_back(op_params[i]);
      } else if (baseline == BASELINE_HEURISTIC) {
        heuristic_op_params.emplace_back(op_params[i]);
      }

      for (size_t k = i + 1; k < mutable_op_params.size(); k ++) {
        // Set partition ratio of OP k as OP i.
        std::shared_ptr<CodlOpChainParam> new_op_param(
            mutable_op_params[k]->Copy());
        //new_op_param->set_part_dim(mutable_op_params[i]->part_dim());
        new_op_param->set_part_dim(1);
        new_op_param->set_part_ratio(mutable_op_params[i]->part_ratio());
        
        chain.Append(new_op_param.get());
        chain.Prepare();
        if (acq == LA_PROFILING) {
          chain.Run(kStartRoundIdx, kRounds, false, &lat_chain);
        } else if (acq == LA_PREDICTION) {
          t0 = NowMicros();
          chain_predictor.Predict(chain, 0, &lat_chain);
          predict_duration += (NowMicros() - t0) / 1000.0;
          predict_times += chain.size();
        }

        if (baseline == BASELINE_SERIAL) {
          //base_chain.Append(op_params[k].get());
          serial_op_params.emplace_back(op_params[k]);
        } else if (baseline == BASELINE_HEURISTIC) {
          heuristic_op_params.emplace_back(op_params[k]);
        }

        const int cur_chain_identity = ((i * 100) + j * 100) + k;
        if (cur_chain_identity != last_chain_identity) {
          last_chain_identity = cur_chain_identity;
          FreeOpChains(base_op_chains);
          base_op_chains.clear();
          std::string baseline_tag;
          if (baseline == BASELINE_SERIAL) {
            baseline_tag = "greedy search - serial";
            //base_chain.Prepare();
            //base_chain.SerialRun(kStartRoundIdx,
            //                     kRounds,
            //                     false,
            //                     &lat_baseline);
            ChainSearch::Serial(serial_op_params, base_op_chains);
          } else if (baseline == BASELINE_HEURISTIC) {
            baseline_tag = "greedy search - heuristic";
            ChainSearch::Heuristic(heuristic_op_params, base_op_chains);
          }

          //ChainSearch::PrintChainInfo(base_op_chains);
          if (acq == LA_PROFILING) {
            CodlOpTaskChainExecutor::Execute(base_op_chains,
                                             kRounds,
                                             baseline_tag,
                                             0,
                                             &lat_baseline);
          } else if (acq == LA_PREDICTION) {
            t0 = NowMicros();
            chain_predictor.Predict(base_op_chains, 0, &lat_baseline);
            predict_duration += (NowMicros() - t0) / 1000.0;
            predict_times += chain.size();
          }
        }

        if (baseline == BASELINE_HEURISTIC) {
          if (CodlOpTaskChainHelper::Equal(&chain, base_op_chains)) {
            lat_chain = lat_baseline;
          }
        }

        lat_gain = lat_baseline - lat_chain;

        if (debug_level >= 1) {
          LOG(INFO) << "===== Greedy Search Info ====";
          LOG(INFO) << "ratio " << ratio_candidates[r_idx];
          LOG(INFO) << "i " << i << ", j " << j << ", k " << k;
          LOG(INFO) << "chain_size " << chain.size()
                    << ", base_chain_size " << base_chain.size()
                    << ", serial_size " << serial_op_params.size()
                    << ", heuristic_size " << heuristic_op_params.size();
          LOG(INFO) << "lat_chain " << lat_chain << " ms"
                    << ", lat_baseline " << lat_baseline << " ms"
                    << ", lat_gain " << lat_gain << " ms";
          LOG(INFO) << "opt_ratio " << ratio_candidates[opt_r_idx]
                    << ", max_chain_size " << max_chain_size
                    << ", min_lat_chain " << min_lat_chain << " ms"
                    << ", max_lat_gain " << max_lat_gain << " ms";
        }

        if (lat_gain >= 0 &&
            lat_gain >= max_part_lat_gain[chain.size() - 1] &&
            chain.size() < kMaxChainLength) {
          j ++;
          lat_chain_prev = lat_chain;
          lat_gain_prev = lat_gain;
          if (lat_gain > max_lat_gain) {
            max_lat_gain = lat_gain;

            max_j = j;
            opt_r_idx = r_idx;
            max_chain_size = chain.size();
            min_lat_chain = lat_chain;
            //opt_chain.CopyFrom(&chain);
          }
          max_part_lat_gain[chain.size() - 1] = lat_gain;
        } else {
          const bool fast_stop = true;
          if (fast_stop) {
            chain.RemoveLast();
            lat_gain = lat_gain_prev;
            lat_chain = lat_chain_prev;
            break;
          }
        }
      }

      bool is_updated = false;
      if (chain.size() > max_chain_size && lat_gain > max_lat_gain) {
        is_updated = true;
      } else if (chain.size() == max_chain_size) {
        if (lat_chain < min_lat_chain && lat_gain > max_lat_gain) {
          is_updated = true;
        }
      }

      if (is_updated) {
        max_j = j;
        opt_r_idx = r_idx;
        max_chain_size = chain.size();
        min_lat_chain = lat_chain;
        //opt_chain.CopyFrom(&chain);
      }
      
      FreeOpChain(chain);
      chain.Clear();
      base_chain.Clear();
      serial_op_params.clear();
      heuristic_op_params.clear();
    }

    opt_chain.Clear();
    if (max_chain_size == 1) {
      opt_chain.Append(op_params[i].get());
    } else if (max_chain_size > 1) {
      for (size_t l = i; l < (max_j + 1); l ++) {
        mutable_op_params[l]->set_part_dim(1);
        mutable_op_params[l]->set_part_ratio(ratio_candidates[opt_r_idx]);
        opt_chain.Append(mutable_op_params[l].get());
      }
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    i = max_j;
    op_chains.push_back(opt_chain);
  }
  
  LOG(INFO) << "Greedy search, prediction times " << predict_times
            << ", time cost " << predict_duration << " ms";

  return 0;
}

int ChainSearch::BuildChainCases(
    const int op_count,
    std::vector<std::set<std::vector<int>>> &chain_case_sets) {
  chain_case_sets = std::vector<std::set<std::vector<int>>>(op_count);
  for (int i = 0; i < op_count; i ++) {
    chain_case_sets[i] = std::set<std::vector<int>>();
  }
  
  std::vector<int> init_chain_case = {op_count};
  chain_case_sets[op_count - 1].insert(init_chain_case);
  LOG(INFO) << "n " << (op_count - 1);
  LOG(INFO) << "init_chain_case " << VectorToString<int>(init_chain_case);

  for (int n = op_count - 2; n >= 0; n --) {
    LOG(INFO) << "n " << n;
    for (auto &chain_case : chain_case_sets[n + 1]) {
      LOG(INFO) << "last_chain_case " << VectorToString<int>(chain_case);
      bool split_flag = true;
      std::vector<std::vector<int>> tmp_cases(1);
      for (auto &len : chain_case) {
        if (len > 1 && split_flag) {
          std::vector<int> tmp_case = tmp_cases[0];
          tmp_cases.clear();
          for (int k = 1; k < len; k ++) {
            tmp_case.push_back(len - k);
            tmp_case.push_back(k);
            tmp_cases.push_back(tmp_case);
            tmp_case.pop_back();
            tmp_case.pop_back();
          }
          split_flag = false;
        } else {
          for (auto &tmp_case : tmp_cases) {
            tmp_case.push_back(len);
          }
        }
      }
      
      for (auto &tmp_case : tmp_cases) {
        LOG(INFO) << "tmp_case " << VectorToString<int>(tmp_case);
        chain_case_sets[n].insert(tmp_case);
      }
    }
  }

  return 0;
}

int CalcTotalCaseCount(
    const std::vector<std::set<std::vector<int>>> &chain_case_sets) {
  int total_case_count = 0;
  for (auto &case_set : chain_case_sets) {
    for (auto &chain_case : case_set) {
      int count = 1;
      for (auto &chain_len : chain_case) {
        count *= 11;
        if (chain_len == 1) {
          count *= 2;
        }
      }
      total_case_count += count;
    }
  }
  return total_case_count;
}

constexpr int kInitDim = 1;
constexpr int kFinalDim = 5;
constexpr int kNextDim[kFinalDim] = {1, 4, 0, 0, 5};

int IncreaseChainPdim(const std::vector<int> &chain_len,
                      std::vector<int> &chain_pdim) {
  const size_t num_chains = chain_pdim.size();
  for (size_t i = 0; i < num_chains; i ++) {
    if (chain_len[i] == 1) {
      if (chain_pdim[i] != kFinalDim) {
        chain_pdim[i] = kNextDim[chain_pdim[i]];
        if (chain_pdim[i] != kFinalDim) {
          break;
        } else {
          chain_pdim[i] = 1;
        }
      }
    }
  }

  return 0;
}

bool IsLastChainDim(const std::vector<int> &chain_len,
                    std::vector<int> &chain_pdim) {
  MACE_UNUSED(chain_len);
  return chain_pdim[chain_pdim.size() - 1] > kInitDim;
}

bool IsLastChainPratio(const std::vector<float> &chain_pratio) {
  return chain_pratio.back() > 0.05;
}

int IncreaseChainPratio(std::vector<float> &chain_pratio) {
  chain_pratio[0] += 0.1;
  for (size_t i = 0; i < chain_pratio.size(); i ++) {
    if (chain_pratio[i] > 1.0 && chain_pratio[i] < 1.05) {
      chain_pratio[i] = 1.0;
    }
    if (chain_pratio[i] > 1.05) {
      if ((i + 1) < chain_pratio.size()) {
        chain_pratio[i + 1] += 0.1;
      }
      chain_pratio[i] = 0.0;
    }
  }

  return 0;
}

int BuildOpChains(
    const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
    const std::vector<int> &chain_case,
    const std::vector<float> &chain_pratio,
    std::vector<CodlOpTaskChain> &op_chains) {
  size_t k = 0;
  for (size_t i = 0; i < chain_case.size(); i ++) {
    auto &num_op = chain_case[i];
    CodlOpTaskChain chain;
    for (int j = 0; j < num_op; j ++) {
      std::shared_ptr<CodlOpChainParam> mutable_op_param(op_params[k]->Copy());
      mutable_op_param->set_part_ratio(chain_pratio[i]);
      chain.Append(mutable_op_param.get());
      k ++;
    }
    op_chains.push_back(chain);
    chain.Clear();
  }

  return 0;
}

int ChainSearch::Full(
    const std::vector<std::shared_ptr<CodlOpChainParam>> &op_params,
    std::vector<CodlOpTaskChain> &op_chains) {
  MACE_UNUSED(op_chains);
  const int op_count = static_cast<int>(op_params.size());
  MACE_CHECK(op_count > 0);
  std::vector<std::set<std::vector<int>>> chain_case_sets;
  ChainSearch::BuildChainCases(op_count, chain_case_sets);
  const int total_case_count = CalcTotalCaseCount(chain_case_sets);

  LPInitContext lp_init_context;
  OpChainLatencyPredictor lat_predictor;
  lat_predictor.Init(&lp_init_context);

  int cur_case_count = 1;
  double lat = 0;
  double min_lat = kMaxDouble;
  std::vector<int> opt_chain_case;
  std::vector<float> opt_chain_pratio;
  //std::vector<CodlOpTaskChain> opt_op_chains;
  const size_t num_sets = chain_case_sets.size();
  for (int i = (num_sets - 1); i >= 0; i --) {
    auto &chain_case_set = chain_case_sets[i];
    for (auto &chain_case : chain_case_set) {
      std::vector<int> chain_len;
      for (auto &num_op : chain_case) {
        chain_len.push_back(num_op);
      }
      chain_len.push_back(1);
      std::vector<int> chain_pdim(chain_case.size() + 1, kInitDim);
      while (!IsLastChainDim(chain_len, chain_pdim)) {
        std::vector<float> chain_pratio(chain_case.size() + 1, 0.0);
        while (!IsLastChainPratio(chain_pratio)) {
          std::vector<CodlOpTaskChain> tmp_op_chains;
          BuildOpChains(op_params,
                        chain_case,
                        chain_pratio,
                        tmp_op_chains);

#if 1
          // Execute OP chains to evaluate latency.
          const bool debug = false;
          const int rounds = 10;
          CodlOpTaskChainExecutor::Execute(tmp_op_chains,
                                           rounds,
                                           "full search",
                                           debug,
                                           &lat);
#endif

#if 0
          // Predict OP chains latency.
          lat_predictor.Predict(tmp_op_chains, &lat);
#endif
          if (lat < min_lat) {
            min_lat = lat;
            opt_chain_case = chain_case;
            opt_chain_pratio = chain_pratio;
            //opt_op_chains = tmp_op_chains;
          }
          LOG(INFO) << "===== Full Search Info ====";
          LOG(INFO) << "case_count " << cur_case_count << "/" << total_case_count
                    << ", chain_case " << VectorToString<int>(chain_case)
                    << ", chain_pdim " << VectorToString<int>(chain_pdim)
                    << ", chain_pratio " << VectorToString<float>(chain_pratio)
                    << ", lat " << lat << " ms"
                    << ", opt_chain_case " << VectorToString<int>(opt_chain_case)
                    << ", opt_chain_case " << VectorToString<float>(opt_chain_pratio)
                    << ", min_lat " << min_lat << " ms";

          FreeOpChains(tmp_op_chains);
          tmp_op_chains.clear();

          IncreaseChainPratio(chain_pratio);

          cur_case_count ++;
        }
        IncreaseChainPdim(chain_len, chain_pdim);
      }
    }
  }

  //op_chains = opt_op_chains;

  return 0;
}

int ChainSearch::PrintChainInfo(
    const std::vector<CodlOpTaskChain> &op_chains) {
  LOG(INFO) << "===== Chain Info =====";
  std::vector<size_t> chain_lens;
  std::vector<int> op_dims;
  std::vector<float> op_ratioes;
  for (size_t i = 0; i < op_chains.size(); i ++) {
    LOG(INFO) << "Chain: id " << i << ", size " << op_chains[i].size()
              << ", dim " << op_chains[i].dim()
              << ", ratio " << op_chains[i].ratio();
    chain_lens.push_back(op_chains[i].size());
    for (size_t j = 0; j < op_chains[i].size(); j ++) {
      op_dims.push_back(op_chains[i].dim());
      op_ratioes.push_back(op_chains[i].ratio());
    }
  }

  LOG(INFO) << "chain_lens " << VectorToString<size_t>(chain_lens);
  LOG(INFO) << "op_dims " << VectorToString<int>(op_dims);
  LOG(INFO) << "op_ratioes " << VectorToString<float>(op_ratioes);

  return 0;
}

int ChainSearch::PrintChainCases(
    const std::vector<std::set<std::vector<int>>> &chain_case_sets) {
  int n = 0;
  for (auto &chain_case_set : chain_case_sets) {
    for (auto &chain_case : chain_case_set) {
      LOG(INFO) << VectorToString<int>(chain_case);
      n ++;
    }
  }
  LOG(INFO) << "Number of cases: " << n;
  
  return 0;
}

}  // namespace mace
