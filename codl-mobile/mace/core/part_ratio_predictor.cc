
#include "mace/core/part_ratio_predictor.h"

namespace mace {

void PartRatioPredictorRegistrationInfo::Register(
    const std::string &key,
    PartRatioPredictorRegistrationInfo::PartRatioPredictorCreator creator) {
  VLOG(3) << "Registering PR predictor: " << key;
  MACE_CHECK(creators.count(key) == 0, "Key is already registered: ", key);
  creators[key] = creator;
}

MaceStatus PartRatioPredictorRegistryBase::Register(
    const std::string &op_type,
    PartRatioPredictorRegistrationInfo::PartRatioPredictorCreator creator) {
  if (registry_.count(op_type) == 0) {
    registry_[op_type] = std::unique_ptr<PartRatioPredictorRegistrationInfo>(
        new PartRatioPredictorRegistrationInfo());
  }

  const std::string op_key = op_type;
  registry_.at(op_type)->Register(op_key, creator);
  return MaceStatus::MACE_SUCCESS;
}

std::unique_ptr<PartitionRatioPredictor>
PartRatioPredictorRegistryBase::CreatePartRatioPredictor(const std::string op_type) const {
  const std::string key = op_type;
  return registry_.at(op_type)->creators.at(key)();
}

}  // namespace mace
