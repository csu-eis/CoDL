
#include <regex>
#include <string>
#include <fstream>
#include "mace/utils/soc_devfreq.h"

namespace mace {
namespace utils {

SocDevfreq *SocDevfreq::Create(const SocName soc_name) {
  SocDevfreq *soc_devfreq = new SocDevfreq();
  const std::string cpufreq_path("/sys/devices/system/cpu/cpufreq");
  std::string gpufreq_path;
  switch (soc_name) {
    case SOC_SNAPDRAGON_855:
    case SOC_SNAPDRAGON_865:
      soc_devfreq->cpu_available_freq_path().push_back(
          cpufreq_path + "/policy4/scaling_available_frequencies");
      soc_devfreq->cpu_available_freq_path().push_back(
          cpufreq_path + "/policy7/scaling_available_frequencies");
      soc_devfreq->cpu_cur_freq_path().push_back(
          cpufreq_path + "/policy4/scaling_cur_freq");
      soc_devfreq->cpu_cur_freq_path().push_back(
          cpufreq_path + "/policy7/scaling_cur_freq");
      soc_devfreq->cpu_max_freq_path().push_back(
          cpufreq_path + "/policy4/scaling_max_freq");
      soc_devfreq->cpu_max_freq_path().push_back(
          cpufreq_path + "/policy7/scaling_max_freq");
      gpufreq_path = std::string("/sys/class/kgsl/kgsl-3d0");
      soc_devfreq->set_gpu_available_freq_path(gpufreq_path + "/gpu_available_frequencies");
      soc_devfreq->set_gpu_cur_freq_path(gpufreq_path + "/gpuclk");
      soc_devfreq->set_gpu_max_freq_path(gpufreq_path + "/max_gpuclk");
      // For non-root user.
      //soc_devfreq->set_gpu_available_freq_path(
      //    std::string(DEFAULT_MACE_WORKSPACE
      //        "/devfreq/gpufreq/gpu_available_frequencies"));
      break;
    case SOC_KIRIN_960:
      soc_devfreq->cpu_available_freq_path().push_back(
          cpufreq_path + "/policy4/scaling_available_frequencies");
      soc_devfreq->cpu_available_freq_path().push_back(
          cpufreq_path + "/policy4/scaling_available_frequencies");
      soc_devfreq->cpu_cur_freq_path().push_back(
          cpufreq_path + "/policy4/scaling_cur_freq");
      soc_devfreq->cpu_cur_freq_path().push_back(
          cpufreq_path + "/policy4/scaling_cur_freq");
      soc_devfreq->cpu_max_freq_path().push_back(
          cpufreq_path + "/policy4/scaling_max_freq");
      soc_devfreq->cpu_max_freq_path().push_back(
          cpufreq_path + "/policy4/scaling_max_freq");
      gpufreq_path = std::string("/sys/devices/platform/e82c0000.mali/devfreq/gpufreq");
      soc_devfreq->set_gpu_available_freq_path(gpufreq_path + "/available_frequencies");
      soc_devfreq->set_gpu_cur_freq_path(gpufreq_path + "/cur_freq");
      soc_devfreq->set_gpu_max_freq_path(gpufreq_path + "/max_freq");
      // For non-root user.
      //soc_devfreq->set_gpu_available_freq_path(
      //    std::string(DEFAULT_MACE_WORKSPACE
      //        "/devfreq/gpufreq/gpu_available_frequencies"));
      break;
    default:
      LOG(ERROR) << "Unsupported SoC name";
      MACE_NOT_IMPLEMENTED;
  }

  return soc_devfreq;
}

SocDevfreq *SocDevfreq::CreateLocal() {
  //return SocDevfreq::Create(SocUtils::ReadSocNameFromFile());
  return SocDevfreq::Create(SocUtils::ReadSocNameFromCodlConfig());
}

std::vector<int> SocDevfreq::GetFreqList(const std::string &path) const {
  std::ifstream f(path);
  std::vector<int> freq_list;
  if (f.is_open()) {
    std::string line;
    if (std::getline(f, line)) {
      std::regex ws_re("\\s+");
      std::vector<std::string> v(std::sregex_token_iterator(line.begin(),
                                                            line.end(),
                                                            ws_re,
                                                            -1),
                                 std::sregex_token_iterator());
      for (auto iter = v.begin(); iter != v.end(); ++iter) {
        const int freq = strtol((*iter).c_str(), nullptr, 0);
        freq_list.push_back(freq);
      }
    }
    f.close();
  } else {
    LOG(WARNING) << "Failed to read frequency file: " << path;
  }

  return freq_list;
}

int SocDevfreq::GetFreqInList(const std::string &path,
                              const int idx,
                              const bool reverse) const {
  const std::vector<int> freq_list = GetFreqList(path);
  const int target_idx = reverse ? (freq_list.size() - 1) - idx : idx;
  return freq_list[target_idx];
}

int SocDevfreq::GetFreq(const std::string &path) const {
  int freq = 0;
  std::ifstream f(path);
  if (f.is_open()) {
    std::string line;
    if (std::getline(f, line)) {
      freq = strtol(line.c_str(), nullptr, 0);
    }
    f.close();
  } else {
    LOG(WARNING) << "Failed to read frequency file: " << path;
  }

  return freq;
}

const std::vector<int> SocDevfreq::cpu_freq(
    const std::vector<int> freq_idx) const {
  const size_t num_clusters = cpu_available_freq_path_.size();
  std::vector<int> cpu_freq;
  for (size_t i = 0; i < num_clusters; ++i) {
    cpu_freq.emplace_back(
        GetFreqInList(cpu_available_freq_path_[i], freq_idx[i], true));
  }

  return cpu_freq;
}

const std::vector<int> SocDevfreq::cpu_cur_freq() const {
  const size_t num_clusters = cpu_cur_freq_path_.size();
  std::vector<int> cpu_freq;
  for (size_t i = 0; i < num_clusters; ++i) {
    cpu_freq.emplace_back(GetFreq(cpu_cur_freq_path_[i]));
  }

  return cpu_freq;
}

const std::vector<int> SocDevfreq::cpu_max_freq() const {
  const size_t num_clusters = cpu_max_freq_path_.size();
  std::vector<int> cpu_freq;
  for (size_t i = 0; i < num_clusters; ++i) {
    cpu_freq.emplace_back(GetFreq(cpu_max_freq_path_[i]));
  }

  return cpu_freq;
}

float SocDevfreq::cpu_capability() const {
  const std::vector<int> cur_freqs = cpu_cur_freq();
  const std::vector<int> max_freqs = cpu_max_freq();
  const size_t num_clusters = cur_freqs.size();
  float capability = 0;
  for (size_t i = 0; i < num_clusters; ++i) {
    capability += (100 * ((float) cur_freqs[i]) / max_freqs[i]);
  }
  return capability / num_clusters;
}

int SocDevfreq::gpu_freq(const int freq_idx) const {
  return GetFreqInList(gpu_available_freq_path_, freq_idx);
}

int SocDevfreq::gpu_cur_freq() const {
  return GetFreq(gpu_cur_freq_path_);
}

int SocDevfreq::gpu_max_freq() const {
  return GetFreq(gpu_max_freq_path_);
}

float SocDevfreq::gpu_capability() const {
  const int cur_freq = gpu_cur_freq();
  const int max_freq = gpu_max_freq();
  return ((float) cur_freq * 100) / max_freq;
}

SocDevfreq *GetSocDevfreq() {
  static SocDevfreq *devfreq = SocDevfreq::CreateLocal();
  return devfreq;
}

}  // namespace utils
}  // namespace mace
