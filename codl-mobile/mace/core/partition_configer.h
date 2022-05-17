
#ifndef MACE_CORE_PARTITION_CONFIGER_H_
#define MACE_CORE_PARTITION_CONFIGER_H_

#include <vector>
#include "mace/utils/logging.h"

#define CONFIG_PATH "/storage/emulated/0/mace_workspace/config"

namespace mace {

class PartitionConfiger {
public:
  PartitionConfiger();

  int dim();

  float ratio();

  inline int dim_at(const size_t idx) const {
    if (idx < dim_values_.size()) {
      return dim_values_[idx];
    } else {
      return 1;
    }
  }

  inline float ratio_at(const size_t idx) const {
    if (idx < ratio_values_.size()) {
      return ratio_values_[idx];
    } else {
      return 0.0f;
    }
  }

  inline int value_size() const {
    MACE_CHECK(dim_values_.size() == ratio_values_.size());
    return static_cast<int>(dim_values_.size());
  }

  bool is_file_open() const;

  void Load();

  void Reset();

  void Clear();

  void Update(const int *dim_arr, const float *ratio_arr);

  void Append(const int dim, const float ratio);

  void Save();

  std::string to_string() const;

private:
  std::string GetNetNameFromEnv();
  std::string MakeConfigFileName(const std::string &key_filename);
  int LoadDim();
  int LoadRatio();
  int SaveDim();
  int SaveRatio();

  const std::string dim_filename_ = "pdim_config";
  const std::string ratio_filename_ = "pratio_config";
  bool is_file_open_;
  std::vector<int> dim_values_;
  std::vector<float> ratio_values_;
  std::vector<int>::iterator dim_iter_;
  std::vector<float>::iterator ratio_iter_;
};

}  // namespace mace

#endif  // MACE_CORE_PARTITION_CONFIGER_H_
