
#ifndef MACE_UTILS_CONST_MODEL_H_
#define MACE_UTILS_CONST_MODEL_H_

#include <string>
#include "mace/utils/predict_model.h"

namespace mace {
namespace utils {

class ConstModel : public PredictModel {
public:
  ConstModel(const std::string name, const double v)
      : name_(name),
        value_(v) {}

  void BuildFromJson(const char *filepath) override {
    MACE_UNUSED(filepath);
    MACE_NOT_IMPLEMENTED;
  }

  double Predict() const override {
    return value_;
  }

  double Predict(const std::vector<double> &features) const override {
    MACE_UNUSED(features);
    return value_;
  }

  double Predict(const std::vector<std::pair<std::string, double>> &features)
      const override {
    MACE_UNUSED(features);
    return value_;
  }

  void DebugPrint() const override {
    LOG(INFO) << "name " << name_ << ", value " << value_;
  }

private:
  std::string name_;
  double value_;
};

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_CONST_MODEL_H_
