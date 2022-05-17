
#include <fstream>
#include "mace/utils/logging.h"
#include "mace/utils/linear_regression_model.h"
#include "third_party/cppjson/cpp_json.h"

namespace mace {
namespace utils {

template<typename T>
static void JsonArrayToVector(const json::array arr,
                              std::vector<T> &vec) {
  vec.clear();
  for (auto iter = arr.begin(); iter != arr.end(); ++iter) {
    const double v = json::to_number<T>(*iter);
    vec.push_back(v);
  }
}

void LinearRegressionModel::BuildFromJson(const char *json_filepath) {
  std::ifstream in(json_filepath);
  //MACE_CHECK(!in.fail(), "Open json file failed");
  if (in.fail()) {
    const std::string json_filepath_str = json_filepath;
    LOG(WARNING) << "Open json file failed, path " << json_filepath_str;
    return;
  }

  LOG(INFO) << "Build LR model named " << name_ << " from " << json_filepath;

  auto lr_params = json::parse(in).as_object();

  if (lr_params.find("means") != lr_params.end()) {
    auto mean_arr = lr_params["means"].as_array();
    JsonArrayToVector<double>(mean_arr, means_);
  }

  if (lr_params.find("stds") != lr_params.end()) {
    auto std_arr = lr_params["stds"].as_array();
    JsonArrayToVector<double>(std_arr, stds_);
  }
  
  auto coef_arr = lr_params["coefs"].as_array();
  JsonArrayToVector<double>(coef_arr, coefs_);
#if 0
  coefs_.clear();
  for (auto iter = coef_arr.begin(); iter != coef_arr.end(); ++iter) {
    const double coef = json::to_number<double>(*iter);
    coefs_.push_back(coef);
  }
#endif

  inter_ = json::to_number<double>(lr_params["inter"]);

  LOG(INFO) << "Loaded means " << VectorToString<double>(means_)
            << ", stds " << VectorToString<double>(stds_)
            << ", coefs " << VectorToString<double>(coefs_)
            << ", inter " << inter_;
}

double LinearRegressionModel::Predict(
    const std::vector<double> &features) const {
  MACE_CHECK(coefs_.size() == features.size(),
             ", model ", name_,
             ", coefs size ", coefs_.size(),
             " not equal to feature size ", features.size());

  double result = 0;
  for (size_t i = 0; i < coefs_.size(); i ++) {
    result += (features[i] - means_[i]) / stds_[i] * coefs_[i];
  }

  return result + inter_;
}

void LinearRegressionModel::DebugPrint() const {
  LOG(INFO) << "name " << name_
            << ", coefs " << VectorToString<double>(coefs_)
            << ", inter " << inter_;
}

}  // namespace utils
}  // namespace mace
