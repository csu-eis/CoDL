
#ifndef MACE_UTILS_THERMAL_ZONE_TOOL_H_
#define MACE_UTILS_THERMAL_ZONE_TOOL_H_

#include <vector>
#include "mace/public/mace.h"
#include "mace/utils/soc_utils.h"

namespace mace {
namespace utils {

class SocThermalZone {
public:
  SocThermalZone() {}
  ~SocThermalZone() {}

  static SocThermalZone *Create(const SocName soc_name);
  static SocThermalZone *CreateLocal();

  MaceStatus SetCPULowestCoolingStatus();
  MaceStatus SetCPULowestCoolingStatus(const size_t dev_id);
  MaceStatus SetCPUHighestCoolingStatus();
  MaceStatus SetCPUHighestCoolingStatus(const size_t dev_id);

  inline void set_cpu_dev_count(const size_t count) {
    cpu_dev_count_ = count;
  }

  inline void set_gpu_dev_count(const size_t count) {
    gpu_dev_count_ = count;
  }

  inline void set_cpu_tz_id(const int i) {
    cpu_tz_id_ = i;
  }

  inline void set_gpu_tz_id(const int i) {
    gpu_tz_id_ = i;
  }

  inline std::vector<int> &cpu_tz_dev_id() {
    return cpu_tz_dev_id_;
  }

  inline std::vector<int> &gpu_tz_dev_id() {
    return gpu_tz_dev_id_;
  }

  inline std::vector<int> &cpu_max_cooling_status() {
    return cpu_max_cooling_status_;
  }

  inline std::vector<int> &cpu_min_cooling_status() {
    return cpu_min_cooling_status_;
  }

  inline std::vector<int> &gpu_max_cooling_status() {
    return gpu_max_cooling_status_;
  }

  inline std::vector<int> &gpu_min_cooling_status() {
    return gpu_min_cooling_status_;
  }

  const std::vector<int> cpu_cur_cooling_status() const;

  int gpu_cur_cooling_status() const;

private:
  int GetCoolingStatus(const int tz_id, const int dev_id) const;
  MaceStatus SetCPUCoolingStatus(const int dev_id, const int status);
  MaceStatus SetCoolingStatus(const int tz_id,
                              const int dev_id,
                              const int status);

  size_t cpu_dev_count_;
  size_t gpu_dev_count_;
  int cpu_tz_id_;
  int gpu_tz_id_;
  std::vector<int> cpu_tz_dev_id_;
  std::vector<int> gpu_tz_dev_id_;
  std::vector<int> cpu_max_cooling_status_;
  std::vector<int> cpu_min_cooling_status_;
  std::vector<int> gpu_max_cooling_status_;
  std::vector<int> gpu_min_cooling_status_;
};

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_THERMAL_ZONE_TOOL_H_
