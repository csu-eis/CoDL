
#include "test/codl_run/op_chain_helper.h"

namespace mace {

bool CodlOpTaskChainHelper::Equal(
    const CodlOpTaskChain *src_chain,
    const std::vector<CodlOpTaskChain> &dst_chains) {
  if (dst_chains.size() < 1) {
    return false;
  }
#if 0
  LOG(INFO) << "===== Chain Helper Info =====";
  LOG(INFO) << "src_dim " << static_cast<int>(src_chain->dim())
            << ", dst_dim " << static_cast<int>(dst_chains[0].dim())
            << ", src_ratio " << src_chain->ratio()
            << ", dst_ratio " << dst_chains[0].ratio();
#endif
  if (src_chain->size() != dst_chains.size()) {
    return false;
  }

  for (size_t i = 0; i < src_chain->size(); i ++) {
    if (src_chain->dim() != dst_chains[i].dim() ||
        src_chain->ratio() != dst_chains[i].ratio()) {
      return false;
    }
  }
  
  return true;
}

index_t CodlOpTaskChainHelper::CalcTotalMemoryUsage(
    const std::vector<CodlOpTaskChain> &op_chains,
    const int debug_level,
    index_t *max_cpu_input_raw_size_ptr,
    index_t *max_cpu_output_raw_size_ptr,
    index_t *const_raw_size_ptr) {
  index_t max_cpu_input_raw_size = -1;
  index_t max_cpu_output_raw_size = -1;
  index_t total_gpu_input_raw_size = 0;
  index_t total_gpu_output_raw_size = 0;
  index_t total_weight_raw_size = 0;
  for (auto &chain : op_chains) {
    const index_t input_raw_size = chain.max_cpu_input_raw_size();
    if (input_raw_size > max_cpu_input_raw_size) {
      max_cpu_input_raw_size = input_raw_size;
    }
    const index_t output_raw_size = chain.max_cpu_output_raw_size();
    if (output_raw_size > max_cpu_output_raw_size) {
      max_cpu_output_raw_size = output_raw_size;
    }

    const index_t gpu_input_raw_size = chain.gpu_input_raw_size();
    const index_t gpu_output_raw_size = chain.gpu_output_raw_size();

    total_gpu_input_raw_size += gpu_input_raw_size;
    total_gpu_output_raw_size += gpu_output_raw_size;

    const index_t weight_raw_size = chain.weight_raw_size();
    const index_t cpu_weight_raw_size = chain.cpu_weight_raw_size();
    total_weight_raw_size += weight_raw_size;
    if (chain.ratio() > 0 && chain.ratio() < 1) {
      total_weight_raw_size += cpu_weight_raw_size;
    }

    if (debug_level >= 3) {
      LOG(INFO) << "cpu_input_raw_size " << input_raw_size
                << ", cpu_weight_raw_size " << cpu_weight_raw_size
                << ", cpu_output_raw_size " << output_raw_size;
      LOG(INFO) << "gpu_input_raw_size " << gpu_input_raw_size
                << ", weight_raw_size " << weight_raw_size
                << ", gpu_output_raw_size " << gpu_output_raw_size;
      LOG(INFO) << "max_cpu_input_raw_size " << max_cpu_input_raw_size
                << ", max_cpu_output_raw_size " << max_cpu_output_raw_size;
      LOG(INFO) << "total_gpu_input_raw_size " << total_gpu_input_raw_size
                << ", total_gpu_output_raw_size " << total_gpu_output_raw_size
                << ", total_weight_raw_size " << total_weight_raw_size;
    }
  }

  if (max_cpu_input_raw_size_ptr != nullptr &&
      max_cpu_input_raw_size > *max_cpu_input_raw_size_ptr) {
    *max_cpu_input_raw_size_ptr = max_cpu_input_raw_size;
  }

  if (max_cpu_output_raw_size_ptr != nullptr &&
      max_cpu_output_raw_size > *max_cpu_output_raw_size_ptr) {
    *max_cpu_output_raw_size_ptr = max_cpu_output_raw_size;
  }

  if (const_raw_size_ptr != nullptr) {
    *const_raw_size_ptr += (total_gpu_input_raw_size + total_gpu_output_raw_size
                              + total_weight_raw_size);
  }

  return max_cpu_input_raw_size + max_cpu_output_raw_size
      + total_gpu_input_raw_size + total_gpu_output_raw_size
      + total_weight_raw_size;
}

}  // namespace mace
