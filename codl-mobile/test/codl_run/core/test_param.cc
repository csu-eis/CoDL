
#include "mace/utils/logging.h"
#include "mace/utils/string_util.h"
#include "test/codl_run/core/test_param.h"

namespace mace {

std::vector<std::vector<index_t>> ArgumentUtils::ParseInputShapeString(
    const std::string &s) {
  size_t pos_semi = 0;
  size_t pos_start = 0, pos_end = 0;
  std::vector<std::vector<index_t>> output_shapes;

  while ((pos_semi = s.find(";", pos_start)) != std::string::npos) {
    pos_end = pos_semi;
    std::string shape_str = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + 1;
    
    std::vector<index_t> shape = ParseValueString<index_t>(shape_str);
    output_shapes.push_back(shape);
  }

  pos_end = s.length();
  std::string shape_str = s.substr(pos_start, pos_end - pos_start);
  std::vector<index_t> shape = ParseValueString<index_t>(shape_str);
  output_shapes.push_back(shape);

  return output_shapes;
}

int ArgumentUtils::ParseInputShapeStringTest() {
  
  std::string shape_str("1,112,112,3;64,3,1,1;1,1;");

  std::vector<std::vector<index_t>> shapes = ParseInputShapeString(shape_str);

  LOG(INFO) << "Input Shape: " << VectorToString<index_t>(shapes[0]);
  LOG(INFO) << "Filter Shape: " << VectorToString<index_t>(shapes[1]);
  LOG(INFO) << "Strides: " << VectorToString<index_t>(shapes[2]);

  return 0;
}

}  // namespace mace
