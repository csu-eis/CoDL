
#ifndef MACE_UTILS_CODL_CONFIG_H_
#define MACE_UTILS_CODL_CONFIG_H_

#include <cstdint>
#include <string>
#include <vector>

//#define CODL_DEFAULT_CONFIG_FILEPATH "/storage/emulated/0/codl_workspace/codl_config.json"

namespace mace {
namespace utils {

class CodlConfig {
 public:
  CodlConfig() {}

  inline std::string soc_name() const {
    return soc_name_;
  }

  inline uint32_t gpu_warp_size() const {
    return gpu_warp_size_;
  }

  inline uint32_t kwg_size() const {
    return kwg_size_;
  }

  inline std::string predict_model_path() const {
    return predict_model_path_;
  }

  inline std::vector<std::string> predict_model_filenames() const {
    return predict_model_filenames_;
  }

  inline bool is_loaded() {
    return config_filepath_.size() > 0;
  }

  void Load();

  void Show();

 private:
  std::string config_filepath_;
  std::string soc_name_;
  uint32_t gpu_warp_size_;
  uint32_t kwg_size_;
  std::string predict_model_path_;
  std::vector<std::string> predict_model_filenames_;
};

CodlConfig *GetGlobalCodlConfig();

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_CODL_CONFIG_H_
