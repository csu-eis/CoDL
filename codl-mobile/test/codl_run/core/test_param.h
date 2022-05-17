
#ifndef TEST_CODL_RUN_CORE_TEST_PARAM_H_
#define TEST_CODL_RUN_CORE_TEST_PARAM_H_

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <string>

#include "mace/core/types.h"
#include "test/codl_run/core/compute_unit.h"

namespace mace {

typedef int64_t index_t;

class TestParam {
 public:
  bool is_debug_on;
  bool do_data_transform;
  bool do_compute;
  bool do_warmup;
  int test_program_idx;
  int cpu_affinity_policy;
  int num_threads;
  int total_rounds;
  MemoryType gpu_memory_type;
  DataType cpu_dtype;
  DataType gpu_dtype;
  int part_dim;
  float part_ratio;
  float delta_part_ratio;
  ComputeUnitHint compute_unit_hint;
  std::vector<index_t> input_shape;
};

class ArgumentUtils {
 public:
  virtual int Parse(int argc, char *argv[], TestParam *param_out) = 0;

  virtual int Check(const TestParam *param) = 0;

  virtual int Print(const TestParam *param) = 0;

  template <typename T>
  static std::vector<T> ParseValueString(const std::string &s) {
    size_t pos_start = 0, pos_end = 0;
    size_t pos_comma;
    std::vector<T> shape;
    
    while ((pos_comma = s.find(",", pos_start)) != std::string::npos) {
      pos_end = pos_comma;
      std::string num_str = s.substr(pos_start, pos_end - pos_start);
      pos_start = pos_end + 1;
      
      int num = ArgumentUtils::StringToInt(num_str.data());
      shape.push_back(num);
    }

    pos_end = s.length();
    std::string num_str = s.substr(pos_start, pos_end - pos_start);
    int num = ArgumentUtils::StringToInt(num_str.data());
    shape.push_back(num);

    return shape;
  }

  static std::vector<std::string> ParseTextString(const std::string &s) {
    size_t pos_start = 0, pos_end = 0;
    size_t pos_comma;
    std::vector<std::string> shape;
    
    while ((pos_comma = s.find(",", pos_start)) != std::string::npos) {
      pos_end = pos_comma;
      std::string str = s.substr(pos_start, pos_end - pos_start);
      pos_start = pos_end + 1;
      
      shape.push_back(str);
    }

    pos_end = s.length();
    std::string str = s.substr(pos_start, pos_end - pos_start);
    shape.push_back(str);

    return shape;
  }
  
  static std::vector<std::vector<index_t>>
      ParseInputShapeString(const std::string &s);

  static std::vector<std::string>
      ParseFeatureNameString(const std::string &s) {
    return ArgumentUtils::ParseTextString(s);
  }

  static std::vector<double>
      ParseFeatureValueString(const std::string &s) {
    return ArgumentUtils::ParseValueString<double>(s);
  }
  
  static int ParseInputShapeStringTest();

 protected:
  inline static int StringToInt(const char *s) {
    return atoi(s);
  }

  inline static float StringToFloat(const char *s) {
    return strtof(s, NULL);
  }
};

}  // namespace mace

#endif  // TEST_CODL_RUN_CORE_TEST_PARAM_H_
