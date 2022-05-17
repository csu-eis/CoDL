
#ifndef MACE_UTILS_DEVFREQ_TOOL_H_
#define MACE_UTILS_DEVFREQ_TOOL_H_

#include <vector>
#include "mace/public/mace.h"
#include "mace/utils/soc_utils.h"

namespace mace {
namespace utils {

class SocDevfreq {
public:
  static SocDevfreq *Create(const SocName soc_name);
  static SocDevfreq *CreateLocal();

  SocDevfreq() : capability_(0) {}

  inline std::vector<std::string> &cpu_available_freq_path() {
    return cpu_available_freq_path_;
  }

  inline std::vector<std::string> &cpu_cur_freq_path() {
    return cpu_cur_freq_path_;
  }

  inline std::vector<std::string> &cpu_max_freq_path() {
    return cpu_max_freq_path_;
  }

  inline std::string &gpu_available_freq_path() {
    return gpu_available_freq_path_;
  }

  inline std::string &gpu_cur_freq_path() {
    return gpu_cur_freq_path_;
  }

  inline std::string &gpu_max_freq_path() {
    return gpu_max_freq_path_;
  }

  inline void set_gpu_available_freq_path(const std::string path) {
    gpu_available_freq_path_ = path;
  }

  inline void set_gpu_cur_freq_path(const std::string path) {
    gpu_cur_freq_path_ = path;
  }

  inline void set_gpu_max_freq_path(const std::string path) {
    gpu_max_freq_path_ = path;
  }

  const std::vector<int> cpu_freq(const std::vector<int> freq_idx) const;

  const std::vector<int> cpu_cur_freq() const;

  const std::vector<int> cpu_max_freq() const;

  float cpu_capability() const;

  int gpu_freq(const int freq_idx) const;

  int gpu_cur_freq() const;
  
  int gpu_max_freq() const;

  float gpu_capability() const;

  float capability() const {
    return capability_;
  }

  void set_capability(float c) {
    capability_ = c;
  }

private:
  std::vector<int> GetFreqList(const std::string &path) const;

  int GetFreqInList(const std::string &path,
                    const int idx,
                    const bool reverse = false) const;

  int GetFreq(const std::string &path) const;

  float capability_;
  std::vector<std::string> cpu_available_freq_path_;
  std::vector<std::string> cpu_cur_freq_path_;
  std::vector<std::string> cpu_max_freq_path_;
  std::string gpu_available_freq_path_;
  std::string gpu_cur_freq_path_;
  std::string gpu_max_freq_path_;
};

SocDevfreq *GetSocDevfreq();

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_DEVFREQ_TOOL_H_
