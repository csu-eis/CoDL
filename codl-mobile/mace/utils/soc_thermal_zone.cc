
#include <regex>
#include <string>
#include <fstream>
#include "mace/utils/soc_thermal_zone.h"

namespace mace {
namespace utils {

SocThermalZone *SocThermalZone::Create(const SocName soc_name) {
  SocThermalZone *soc_thermal_zone = new SocThermalZone();
  switch (soc_name) {
    case SOC_SNAPDRAGON_855:
      soc_thermal_zone->set_cpu_dev_count(2);
      soc_thermal_zone->set_gpu_dev_count(1);
      soc_thermal_zone->set_cpu_tz_id(35);
      soc_thermal_zone->set_gpu_tz_id(32);
      soc_thermal_zone->cpu_max_cooling_status().push_back(16);
      soc_thermal_zone->cpu_max_cooling_status().push_back(19);
      soc_thermal_zone->cpu_min_cooling_status().push_back(0);
      soc_thermal_zone->cpu_min_cooling_status().push_back(0);
      soc_thermal_zone->gpu_max_cooling_status().push_back(4);
      soc_thermal_zone->gpu_min_cooling_status().push_back(0);
      break;
    case SOC_SNAPDRAGON_865:
      soc_thermal_zone->set_cpu_dev_count(2);
      soc_thermal_zone->set_gpu_dev_count(1);
      soc_thermal_zone->set_cpu_tz_id(28);
      soc_thermal_zone->set_gpu_tz_id(25);
      soc_thermal_zone->cpu_max_cooling_status().push_back(17);
      soc_thermal_zone->cpu_max_cooling_status().push_back(19);
      soc_thermal_zone->cpu_min_cooling_status().push_back(0);
      soc_thermal_zone->cpu_min_cooling_status().push_back(0);
      soc_thermal_zone->gpu_max_cooling_status().push_back(5);
      soc_thermal_zone->gpu_min_cooling_status().push_back(0);
      break;
    case SOC_KIRIN_960:
      soc_thermal_zone->set_cpu_dev_count(1);
      soc_thermal_zone->set_gpu_dev_count(1);
      soc_thermal_zone->set_cpu_tz_id(0);
      soc_thermal_zone->set_gpu_tz_id(0);
      soc_thermal_zone->gpu_tz_dev_id().push_back(0);
      soc_thermal_zone->cpu_tz_dev_id().push_back(2);
      soc_thermal_zone->cpu_max_cooling_status().push_back(4);
      soc_thermal_zone->cpu_min_cooling_status().push_back(0);
      soc_thermal_zone->gpu_max_cooling_status().push_back(5);
      soc_thermal_zone->gpu_min_cooling_status().push_back(0);
      break;
    default:
      LOG(ERROR) << "Error: Unsupported soc name";
      MACE_NOT_IMPLEMENTED;
  }
  
  return soc_thermal_zone;
}

SocThermalZone *SocThermalZone::CreateLocal() {
  return SocThermalZone::Create(SocUtils::ReadSocNameFromFile());
}

MaceStatus SocThermalZone::SetCoolingStatus(const int tz_id,
                                            const int dev_id,
                                            const int status) {
  std::string cooling_status_conf = MakeString(
      "/sys/class/thermal/thermal_zone",
      tz_id,
      "/cdev",
      dev_id,
      "/cur_state");
  std::ofstream f(cooling_status_conf);
  if (!f.is_open()) {
    LOG(ERROR) << "failed to open " << cooling_status_conf;
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
  
  f << status;

#if 0
  LOG(INFO) << "Set cooling status, tz " << tz_id
            << ", dev " << dev_id << ", status " << status;
#endif
  
  if (f.bad()) {
    LOG(ERROR) << "failed to write " << cooling_status_conf;
  }
  
  f.close();
  
  return MaceStatus::MACE_SUCCESS;
}

int SocThermalZone::GetCoolingStatus(const int tz_id,
                                     const int dev_id) const {
  std::string cooling_status_conf = MakeString(
      "/sys/class/thermal/thermal_zone",
      tz_id,
      "/cdev",
      dev_id,
      "/cur_state");
  std::ifstream f(cooling_status_conf);
  int status = 0;
  if (f.is_open()) {
    std::string line;
    if (std::getline(f, line)) {
      status = strtol(line.c_str(), nullptr, 0);
    }
    f.close();
  } else {
    LOG(WARNING) << "Failed to read cooling state file: " << cooling_status_conf;
  }

  return status;
}

MaceStatus SocThermalZone::SetCPUCoolingStatus(const int dev_id,
                                               const int status) {
  return SetCoolingStatus(cpu_tz_id_, dev_id, status);
}

MaceStatus SocThermalZone::SetCPULowestCoolingStatus() {
  for (size_t i = 0; i < cpu_dev_count_; i ++) {
    if (SetCPUCoolingStatus(i, cpu_min_cooling_status_[i])
        != MaceStatus::MACE_SUCCESS) {
        LOG(ERROR) << "Set cooling status failed";
        return MaceStatus::MACE_RUNTIME_ERROR;
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SocThermalZone::SetCPULowestCoolingStatus(const size_t dev_id) {
  MACE_CHECK(dev_id < cpu_dev_count_);
  return SetCPUCoolingStatus(dev_id, cpu_min_cooling_status_[dev_id]);
}

MaceStatus SocThermalZone::SetCPUHighestCoolingStatus() {
  for (size_t i = 0; i < cpu_dev_count_; i ++) {
    if (SetCPUCoolingStatus(i, cpu_max_cooling_status_[i])
        != MaceStatus::MACE_SUCCESS) {
        LOG(ERROR) << "Set cooling status failed";
        return MaceStatus::MACE_RUNTIME_ERROR;
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SocThermalZone::SetCPUHighestCoolingStatus(const size_t dev_id) {
  MACE_CHECK(dev_id < cpu_dev_count_);
  return SetCPUCoolingStatus(dev_id, cpu_max_cooling_status_[dev_id]);
}

const std::vector<int> SocThermalZone::cpu_cur_cooling_status() const {
  std::vector<int> cooling_status_list;

  if (cpu_tz_dev_id_.size() == 0) {
    for (size_t i = 0; i < cpu_dev_count_; i ++) {
      cooling_status_list.emplace_back(GetCoolingStatus(cpu_tz_id_, i));
    }
  } else {
    MACE_CHECK(cpu_dev_count_ == cpu_tz_dev_id_.size(), "Size shoule be same");
    for (auto it = cpu_tz_dev_id_.begin(); it != cpu_tz_dev_id_.end(); ++it) {
      cooling_status_list.emplace_back(GetCoolingStatus(cpu_tz_id_, *it));
    }
  }
  
  return cooling_status_list;
}

int SocThermalZone::gpu_cur_cooling_status() const {
  if (gpu_tz_dev_id_.size() == 0) {
    return GetCoolingStatus(gpu_tz_id_, 0);
  } else {
    MACE_CHECK(gpu_dev_count_ == gpu_tz_dev_id_.size(), "Size should be same");
    return GetCoolingStatus(gpu_tz_id_, gpu_tz_dev_id_[0]);
  }
}

}  // namespace utils
}  // namespace mace
