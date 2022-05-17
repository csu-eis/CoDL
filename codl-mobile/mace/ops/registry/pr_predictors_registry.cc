
#include "mace/ops/registry/pr_predictors_registry.h"

namespace mace {

namespace ops {
extern void RegisterConv2dPartRatioPredictor(
    PartRatioPredictorRegistryBase *pr_registry);
}  // namespace ops

PartRatioPredictorRegistry::PartRatioPredictorRegistry()
    : PartRatioPredictorRegistryBase() {
  ops::RegisterConv2dPartRatioPredictor(this);

  LOG(INFO) << "Register all partition ratio predictors success";
}

}  // namespace mace
