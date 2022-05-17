
#include "mace/core/future.h"

namespace mace {

void SetFutureDefaultWaitFn(StatsFuture *future) {
  if (future != nullptr) {
    future->wait_fn = [](CallStats * stats) {
      if (stats != nullptr) {
        stats->start_micros = NowMicros();
        stats->end_micros = stats->start_micros;
      }
    };
  }
}

void MergeMultipleFutureWaitFn(
    const std::vector<StatsFuture> &org_futures,
    StatsFuture *dst_future) {
  if (dst_future != nullptr) {
    dst_future->wait_fn = [org_futures](CallStats *stats) {
      if (stats != nullptr) {
        stats->start_micros = INT64_MAX;
        stats->end_micros = 0;
        for (auto &org_future : org_futures) {
          CallStats tmp_stats;
          if (org_future.wait_fn != nullptr) {
            org_future.wait_fn(&tmp_stats);
#if 0
            LOG(INFO) << "tmp_start_micros " << tmp_stats.start_micros
                      << ", tmp_end_micros " << tmp_stats.end_micros
                      << ", tmp_duration " << (tmp_stats.end_micros - tmp_stats.start_micros) / 1000.0 << " ms";
#endif
            stats->start_micros = std::min(stats->start_micros,
                                           tmp_stats.start_micros);
            stats->end_micros += tmp_stats.end_micros - tmp_stats.start_micros;
          }
        }
        stats->end_micros += stats->start_micros;
#if 0
        LOG(INFO) << "start_micros " << stats->start_micros
                  << ", end_micros " << stats->end_micros
                  << ", duration " << (stats->end_micros - stats->start_micros) / 1000.0 << " ms";
#endif
      } else {
        for (auto &org_future : org_futures) {
          if (org_future.wait_fn != nullptr) {
            org_future.wait_fn(nullptr);
          }
        }
      }
    };
  }
}

}  // namespace mace
