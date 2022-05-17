
#include "test/codl_run/op_chain_executor.h"
#include "test/codl_run/op_chain_search.h"

namespace mace {

double CalcOpComputeDuration(const std::vector<double> &stat) {
  MACE_CHECK(stat.size() == 5);
  return fmax(stat[2], stat[3]);
}

double CalcOpDuration(const std::vector<double> &stat) {
  MACE_CHECK(stat.size() == 5);
  return stat[1] + fmax(stat[2], stat[3]) + stat[4];
}

double CalcOpIdlePercent(const std::vector<double> &stat) {
  MACE_CHECK(stat.size() == 5);
  if (stat[2] < 0.0001 || stat[3] < 0.0001) {
    return 0.0;
  }
  const double total_time = CalcOpComputeDuration(stat);
  const double idle_pect = (stat[2] - stat[3]) / total_time;
  return idle_pect;
}

std::string OpDurationToString(const std::vector<double> &stat) {
  MACE_CHECK(stat.size() == 5);
  std::stringstream stream;
  stream << "dms " << (stat[1] + stat[4]) << " ms"
         << ", cpu " << stat[2] << " ms, gpu " << stat[3] << " ms"
         << ", comp " << CalcOpComputeDuration(stat) << " ms"
         << ", idle " << CalcOpIdlePercent(stat) << " %%";
  return stream.str();
}

int CodlOpTaskChainExecutor::Execute(
    std::vector<CodlOpTaskChain> &op_chains,
    const int rounds,
    const std::string &tag,
    const int debug_level,
    std::vector<double> &out_lats) {
  if (debug_level >= 1) {
    ChainSearch::PrintChainInfo(op_chains);
  }

  for (auto &chain : op_chains) {
    if (!chain.is_ready()) {
      chain.Prepare();
    }
  }

  const int warmup_rounds = 0;
  //const int inner_rounds = 20;
  double avg_lat = 0, avg_dms_lat = 0, avg_comp_lat = 0;
  for (int i = 0; i < (warmup_rounds + rounds); i ++) {
#if 0
    if (debug_level >= 2) {
      LOG(INFO) << "Round " << i;
    }
#endif
    int64_t t0 = NowMicros();
    double calc_duration = 0;
    double calc_compute_duration = 0;
    double calc_idle_pect = 0;
    for (size_t j = 0; j < op_chains.size(); j ++) {
      DurationCollector<double> dura_collector;
      CodlOpTaskChain &chain = op_chains[j];
      chain.Run(&dura_collector);
      
      std::vector<double> stat = dura_collector.StatSum();
      calc_duration += CalcOpDuration(stat);
      calc_compute_duration += CalcOpComputeDuration(stat);
      calc_idle_pect = CalcOpIdlePercent(stat);

#if 1
      if (debug_level >= 2) {
        //LOG(INFO) << "chain_idx " << j << ", " << VectorToString<double>(stat);
        LOG(INFO) << "chain_idx " << j << ", " << OpDurationToString(stat);
      }
#endif
    }
    const double duration = (NowMicros() - t0) / 1000.0;
    const double calc_dt_map_sync_duration
        = (calc_duration - calc_compute_duration);

    if (i >= warmup_rounds) {
      avg_lat += calc_duration;
      avg_dms_lat += calc_dt_map_sync_duration;
      avg_comp_lat += calc_compute_duration;
    }

#if 1
    if (debug_level >= 2) {
      LOG(INFO) << "Example: " << tag
                << ", round " << i
                << ", calc_duration " << calc_duration << " ms"
                << ", calc_dms_duration " << calc_dt_map_sync_duration << " ms"
                << ", calc_comp_duration " << calc_compute_duration << " ms"
                << ", duration " << duration << " ms"
                << ", calc_idle_pect " << calc_idle_pect << " %%";
    }
#endif
  }

  avg_lat = avg_lat / rounds;
  avg_dms_lat = avg_dms_lat / rounds;
  avg_comp_lat = avg_comp_lat / rounds;
  out_lats.push_back(avg_lat);
  out_lats.push_back(avg_dms_lat);
  out_lats.push_back(avg_comp_lat);

  if (debug_level >= 1) {
    LOG(INFO) << "Example: " << tag
              << ", avg_lat " << avg_lat << " ms"
              << ", avg_dms_lat " << avg_dms_lat << " ms"
              << ", avg_comp_lat " << avg_comp_lat << " ms";
  }

  return 0;
}

}  // namespace mace
