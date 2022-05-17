
#ifndef MACE_OPS_COMMON_GRU_TYPE_H_
#define MACE_OPS_COMMON_GRU_TYPE_H_

#include <string>

namespace mace {
namespace ops {

enum GruDirection {
  GD_FORWARD = 0,
  GD_REVERSE = 1,
  GD_BIDIRECTIONAL = 2
};

inline std::string GruDirectionToString(const GruDirection direction) {
  switch (direction) {
    case GD_FORWARD:
      return "FORWARD";
    case GD_REVERSE:
      return "REVERSE";
    case GD_BIDIRECTIONAL:
      return "BIDIRECTIONAL";
    default:
      return "UNKNOWN";
  }
}

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_GRU_TYPE_H_
