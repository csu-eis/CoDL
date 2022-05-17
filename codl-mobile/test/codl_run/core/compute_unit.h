
#ifndef TEST_CODL_RUN_CORE_COMPUTE_UNIT_H_
#define TEST_CODL_RUN_CORE_COMPUTE_UNIT_H_

namespace mace {

enum ComputeUnitType {
  COMPUTE_UNIT_TYPE_CPU = 0,
  COMPUTE_UNIT_TYPE_GPU = 1
};

enum ComputeUnitHint {
  COMPUTE_UNIT_HINT_DEFAULT = 0,
  COMPUTE_UNIT_HINT_CPU     = 1,
  COMPUTE_UNIT_HINT_GPU     = 2
};

inline bool IsComputeUnitOn(ComputeUnitType compute_unit,
                            ComputeUnitHint hint) {
  if (hint == COMPUTE_UNIT_HINT_DEFAULT) {
    return true;
  }

  switch (compute_unit) {
    case COMPUTE_UNIT_TYPE_CPU:
      return hint == COMPUTE_UNIT_HINT_CPU;
    case COMPUTE_UNIT_TYPE_GPU:
      return hint == COMPUTE_UNIT_HINT_GPU;
    default:
      return false;
  }
}

}  // namespace mace

#endif  // TEST_CODL_RUN_CORE_COMPUTE_UNIT_H_
