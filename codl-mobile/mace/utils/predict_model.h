
#ifndef MACE_UTILS_PREDICT_MODEL_H_
#define MACE_UTILS_PREDICT_MODEL_H_

#include <string>
#include <vector>

namespace mace {
namespace utils {

class PredictModel {
public:
  virtual ~PredictModel() {}

  virtual void BuildFromJson(const char *filepath) = 0;

  virtual double Predict() const = 0;

  virtual double Predict(const std::vector<double> &features) const = 0;

  virtual double Predict(
      const std::vector<std::pair<std::string, double>> &features) const = 0;

  virtual void DebugPrint() const = 0;
};

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_PREDICT_MODEL_H_
