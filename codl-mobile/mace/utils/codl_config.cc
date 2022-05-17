
#include <fstream>
#include "mace/utils/logging.h"
#include "mace/utils/codl_config.h"
#include "third_party/cppjson/cpp_json.h"

constexpr const char *kCodlConfigPath = "CODL_CONFIG_PATH";

namespace mace {
namespace utils {

inline std::string GetCodlConfigPathFromEnv() {
  std::string path;
  GetEnv(kCodlConfigPath, &path);
  return path;
}

void CodlConfig::Load() {
  if (is_loaded()) {
    return;
  }
  
  // Load config path from env.
  config_filepath_ = GetCodlConfigPathFromEnv();
  if (config_filepath_.size() == 0) {
    return;
  }

  // Load json.
  std::ifstream in(config_filepath_);
  MACE_CHECK(!in.fail(),
             "Open CoDL config file failed, of which path is ",
             config_filepath_);
  auto configs = json::parse(in).as_object();
  
  // Soc name.
  soc_name_ = configs["SocName"].as_string();

  // GPU warp size.
  gpu_warp_size_ = json::to_number<uint32_t>(configs["GPUWarpSize"]);

  // GPU max workgroup size.
  kwg_size_ = json::to_number<uint32_t>(configs["MaxWorkGroupSize"]);

  // Predict model path.
  predict_model_path_ = configs["PredictModelPath"].as_string();

  // Predict model file names.
  auto file_arr = configs["PredictModelFiles"].as_array();
  for (auto iter = file_arr.begin(); iter != file_arr.end(); ++iter) {
    const std::string filename = (*iter).as_string();
    predict_model_filenames_.emplace_back(filename);
  }

  Show();
}

void CodlConfig::Show() {
  LOG(INFO) << "Config file path: " << config_filepath_;
  LOG(INFO) << "Soc name: " << soc_name_;
  LOG(INFO) << "GPU warp size: " << gpu_warp_size_;
  LOG(INFO) << "Max work group size: " << kwg_size_;
  LOG(INFO) << "Predict model path: " << predict_model_path_;
  for (auto iter = predict_model_filenames_.begin();
      iter != predict_model_filenames_.end(); ++iter) {
    LOG(INFO) << "Predict model file name: " << *iter;
  }
}

CodlConfig *GetGlobalCodlConfig() {
  static CodlConfig config;
  if (!config.is_loaded()) {
    config.Load();
  }
  return &config;
}

}  // namespace utils
}  // namespace mace
