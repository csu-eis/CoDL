
#ifndef MACE_UTILS_LINEAR_REGRESSION_MODEL_H_
#define MACE_UTILS_LINEAR_REGRESSION_MODEL_H_

#include <string>
#include <vector>
#include "mace/utils/logging.h"
#include "mace/utils/predict_model.h"

namespace mace {
namespace utils {

class LinearRegressionModel : public PredictModel {
public:
  LinearRegressionModel(const std::string name)
      : name_(name) {}

  LinearRegressionModel(const std::string name,
                        const std::vector<double> means,
                        const std::vector<double> stds,
                        const std::vector<double> coefs,
                        const double inter)
      : name_(name), means_(means), stds_(stds), coefs_(coefs), inter_(inter) {}

  void BuildFromJson(const char *json_filepath) override;

  double Predict() const override {
    MACE_NOT_IMPLEMENTED;
    return 0;
  }

  double Predict(const std::vector<double> &features) const override;

  double Predict(const std::vector<std::pair<std::string, double>> &features)
      const override {
    MACE_UNUSED(features);
    std::vector<double> new_features;
    for (auto &feature : features) {
      new_features.push_back(feature.second);
    }
    return Predict(new_features);
  }

  void DebugPrint() const override;

private:
  std::string name_;
  std::vector<double> means_;
  std::vector<double> stds_;
  std::vector<double> coefs_;
  double inter_;
};

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_LINEAR_REGRESSION_MODEL_H_
