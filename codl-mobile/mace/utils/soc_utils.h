
#ifndef MACE_UTILS_SOC_UTILS_H_
#define MACE_UTILS_SOC_UTILS_H_

#include <fstream>
#include "mace/utils/logging.h"
#include "mace/utils/codl_config.h"

#define DEFAULT_MACE_WORKSPACE "/storage/emulated/0/mace_workspace"

namespace mace {
namespace utils {

// NOTE(fucheng): Other SoC requires to be extended here.
enum SocName {
  SOC_NONE = 0,
  SOC_SNAPDRAGON_855 = 10,
  SOC_SNAPDRAGON_865 = 11,
  SOC_KIRIN_960 = 20,
};

static const std::map<std::string, SocName> kSocNameMap = {
  {"Snapdragon865",SOC_SNAPDRAGON_865},
  {"Snapdragon855", SOC_SNAPDRAGON_855},
  {"Kirin960", SOC_KIRIN_960}
};

class SocUtils {
public:
  static SocName ReadSocNameFromFile() {
    std::string soc_name_conf = std::string(
        DEFAULT_MACE_WORKSPACE "/devinfo/soc_name");
    std::ifstream f(soc_name_conf);
    SocName soc_name = SOC_NONE;
    if (f.is_open()) {
      std::string soc_name_str;
      if (std::getline(f, soc_name_str)) {
        auto it = kSocNameMap.find(soc_name_str);
        if (it != kSocNameMap.end()) {
          soc_name = it->second;
        }
      }
      f.close();
    } else {
      LOG(WARNING) << "Failed to read soc name: " << soc_name_conf;
      MACE_NOT_IMPLEMENTED;
    }
    
    return soc_name;
  }

  static SocName ReadSocNameFromCodlConfig() {
    CodlConfig *config = GetGlobalCodlConfig();
    auto it = kSocNameMap.find(config->soc_name());
    if (it != kSocNameMap.end()) {
      return it->second;
    } else {
      return SOC_NONE;
    }
  }
};

#undef DEFAULT_MACE_WORKSPACE

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_SOC_UTILS_H_
