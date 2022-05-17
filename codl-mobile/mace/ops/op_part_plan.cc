
#include "mace/ops/op_part_plan.h"

namespace mace {
namespace ops {

index_t PartPlanUtils::RoundUpDiv(const index_t v, const index_t f) {
  return (v + f - 1) / f;
}

index_t PartPlanUtils::RoundUp(const index_t v, const index_t f) {
  return RoundUpDiv(v, f) * f;
}

index_t PartPlanUtils::MultipleAndAlign(
    const index_t len,
    const float ratio,
    const int align) {
  const float scaled_len = len * ratio;
  index_t count = scaled_len / align;
  if (scaled_len > 0 && (scaled_len != count * align)) {
    count += 1;
  }
  // Minimal count is 1.
  //if (count == 0) {
  //  count = 1;
  //}
  return count * align;
}

index_t PartPlanUtils::MultipleAndRoundUp(
    const double len,
    const double ratio,
    const index_t base) {
  const double partial_len = len * ratio;
  const index_t v0 = static_cast<index_t>(partial_len * 10.0);
  const index_t v1 = static_cast<index_t>(base * 10.0);
  const index_t v2 = (v0 % v1 > 0) ? (v0 / v1 + 1) : (v0 / v1);
  return v2 * base;
}

}  // namespace ops
}  // namespace mace
