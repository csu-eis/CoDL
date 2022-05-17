
#ifndef MACE_CORE_PART_RATE_PREDICTOR_H_
#define MACE_CORE_PART_RATE_PREDICTOR_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "mace/public/mace.h"
#include "mace/core/op_context.h"
#include "mace/utils/logging.h"

namespace mace {

class PartitionRatioPredictor {
 public:
  PartitionRatioPredictor() : model_start_idx_(0) {}

  PartitionRatioPredictor(const index_t model_start_idx)
      : model_start_idx_(model_start_idx) {}

  virtual ~PartitionRatioPredictor() = default;
  
  virtual MaceStatus Init() = 0;
  
  virtual void Predict(OpContext *context,
                       std::vector<double> &inputs,
                       std::vector<double> &outputs) = 0;

  virtual void PredictChain(OpContext *context,
                            std::vector<std::vector<double>> &inputs,
                            std::vector<double> &outputs) = 0;
 protected:
  index_t model_start_idx_;
};

struct PartRatioPredictorRegistrationInfo {
 public:
  typedef std::function<std::unique_ptr<PartitionRatioPredictor>()> PartRatioPredictorCreator;

  void Register(const std::string &key, PartRatioPredictorCreator creator);

  std::unordered_map<std::string, PartRatioPredictorCreator> creators;
};

class PartRatioPredictorRegistryBase {
 public:
  PartRatioPredictorRegistryBase() = default;
  virtual ~PartRatioPredictorRegistryBase() = default;

  MaceStatus Register(const std::string &op_type,
                      PartRatioPredictorRegistrationInfo::PartRatioPredictorCreator creator);

  std::unique_ptr<PartitionRatioPredictor> CreatePartRatioPredictor(
      const std::string op_type) const;

  template<class DerivedType>
  static std::unique_ptr<PartitionRatioPredictor> DefaultCreator() {
    return std::unique_ptr<PartitionRatioPredictor>(new DerivedType());
  }

 private:
  std::unordered_map<
      std::string, std::unique_ptr<PartRatioPredictorRegistrationInfo>> registry_;
  MACE_DISABLE_COPY_AND_ASSIGN(PartRatioPredictorRegistryBase);
};

}  // namespace mace

#endif  // MACE_CORE_PART_RATE_PREDICTOR_H_
