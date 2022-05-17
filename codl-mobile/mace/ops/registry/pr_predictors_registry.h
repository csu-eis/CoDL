
#ifndef MACE_OPS_REGISTRY_PR_PREDICTORS_REGISTRY_H_
#define MACE_OPS_REGISTRY_PR_PREDICTORS_REGISTRY_H_

#include "mace/core/part_ratio_predictor.h"

namespace mace {

class PartRatioPredictorRegistry : public PartRatioPredictorRegistryBase {
public:
  PartRatioPredictorRegistry();
  ~PartRatioPredictorRegistry() = default;
};

}  // namespace mace

#endif  // MACE_OPS_REGISTRY_PR_PREDICTORS_REGISTRY_H_
