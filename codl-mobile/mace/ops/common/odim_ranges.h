
#ifndef MACE_OPS_COMMON_ODIM_RANGES_H_
#define MACE_OPS_COMMON_ODIM_RANGES_H_

#include <vector>
#include "mace/utils/logging.h"
#include "mace/utils/string_util.h"

namespace mace {
namespace ops {

/**
 * Dimension information of tensor, especially output tensor.
 * The first dimension is shape, 4 in most cases.
 * The second dimension are:
 *   [0]: destination start
 *   [1]: destination end
 *   [2]: offset to source
 **/
typedef std::vector<std::vector<index_t>> OdimRanges;

inline void PrintOdimRanges(const OdimRanges &odim_ranges) {
  for (size_t i = 0; i < odim_ranges.size(); ++i) {
    if (odim_ranges[i].size() == 3) {
      LOG(INFO) << "dim0 " << i
                << ", values " << VectorToString<index_t>(odim_ranges[i]);
    }
  }
}

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_ODIM_RANGES_H_
