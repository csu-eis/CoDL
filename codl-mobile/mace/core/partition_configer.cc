
#include <iterator>
#include <fstream>
#include <algorithm>

#include "mace/core/partition_configer.h"

namespace mace {

std::string PartitionConfiger::GetNetNameFromEnv() {
  std::string name;
  GetEnv("MACE_NET_NAME_HINT", &name);
  return name;
}

std::string PartitionConfiger::MakeConfigFileName(
    const std::string &key_filename) {
  std::string nn_name = GetNetNameFromEnv();
  std::string filename;
  if (!nn_name.empty()) {
    filename = MakeString(CONFIG_PATH, "/", key_filename, "_", nn_name, ".txt");
  } else {
    filename = MakeString(CONFIG_PATH, "/", key_filename, ".txt");
  }

  return filename;
}

PartitionConfiger::PartitionConfiger() : is_file_open_(false) {
  //Load();
}

int PartitionConfiger::LoadDim() {
  std::string in_file_name = MakeConfigFileName(dim_filename_);
  //LOG(INFO) << "PartitionConfiger: Try to read file " << in_file_name;
  std::ifstream in_file(in_file_name);
  //MACE_CHECK(in_file.is_open(), "Open dimension configure file failed");
  if (in_file.is_open()) {
    const std::istream_iterator<int> start(in_file), end;
    const std::vector<int> values(start, end);
    MACE_CHECK(values.size() > 0);
    dim_values_ = values;
    dim_iter_ = dim_values_.begin();
    in_file.close();

#ifdef CODL_ENABLE_DEBUG
    LOG(INFO) << "PartitionConfiger: Read file " << in_file_name;
#endif
    LOG(INFO) << "PartitionConfiger: Loaded pdim " << VectorToString<int>(dim_values_);

    return 1;
  } else {
    return 0;
  }
}

int PartitionConfiger::LoadRatio() {
  std::string in_file_name = MakeConfigFileName(ratio_filename_);
  //LOG(INFO) << "PartitionConfiger: Try to read file " << in_file_name;
  std::ifstream in_file(in_file_name);
  //MACE_CHECK(in_file.is_open(), "Open ratio configure file failed");
  if (in_file.is_open()) {
    const std::istream_iterator<float> start(in_file), end;
    const std::vector<float> values(start, end);
    MACE_CHECK(values.size() > 0);
    ratio_values_ = values;
    ratio_iter_ = ratio_values_.begin();
    in_file.close();

#ifdef CODL_ENABLE_DEBUG
    LOG(INFO) << "PartitionConfiger: Read file " << in_file_name;
#endif
    LOG(INFO) << "PartitionConfiger: Loaded pratio " << VectorToString<float>(ratio_values_);

    return 1;
  } else {
    return 0;
  }
}

void PartitionConfiger::Load() {
  if (!is_file_open_) {
    int ret = 0;
    ret = LoadDim();
    if (!ret) {
      return;
    }
    ret = LoadRatio();
    if (!ret) {
      return;
    }
    is_file_open_ = true;
  } else {
    is_file_open_ = false;
  }
}

void PartitionConfiger::Reset() {
  if (is_file_open_) {
    // Reset iterator.
    dim_iter_ = dim_values_.begin();
    ratio_iter_ = ratio_values_.begin();
  } //else {
  //  Load();
  //}
}

void PartitionConfiger::Clear() {
  dim_values_.clear();
  ratio_values_.clear();
}

void PartitionConfiger::Update(const int *dim_arr, const float *ratio_arr) {
  MACE_CHECK(dim_values_.size() > 0);
  if (dim_arr != nullptr) {
    for (size_t i = 0; i < dim_values_.size(); ++i) {
      dim_values_[i] = dim_arr[0];
    }
  }
  MACE_CHECK(ratio_values_.size() > 0);
  if (ratio_arr != nullptr) {
    for (size_t i = 0; i < ratio_values_.size(); ++i) {
      ratio_values_[i] = ratio_arr[0];
    }
  }
}

bool PartitionConfiger::is_file_open() const {
  return is_file_open_;
}

void PartitionConfiger::Append(const int dim, const float ratio) {
  dim_values_.push_back(dim);
  ratio_values_.push_back(ratio);
}

int PartitionConfiger::dim() {
  if (is_file_open_) {
    //MACE_CHECK(dim_iter_ != values_.end());
    if (dim_iter_ == dim_values_.end()) {
      LOG(WARNING) << "Iterator reach the end,"
                   << " perhaps something wrong in partition ratio configer.";
      MACE_CHECK(dim_iter_ != dim_values_.end());
      return 1;
    } else {
      const float v = *dim_iter_;
      dim_iter_ ++;
      return v;
    }
  } else {
    dim_values_.push_back(1);
    return 1;
  }
}

float PartitionConfiger::ratio() {
  if (is_file_open_) {
    //MACE_CHECK(ratio_iter_ != values_.end());
    if (ratio_iter_ == ratio_values_.end()) {
      LOG(WARNING) << "Iterator reach the end,"
                   << " perhaps something wrong in partition ratio configer.";
      MACE_CHECK(ratio_iter_ != ratio_values_.end());
      return 1.0f;
    } else {
      const float v = *ratio_iter_;
      ratio_iter_ ++;
      return v;
    }
  } else {
    ratio_values_.push_back(1.0f);
    return 1.0f;
  }
}

int PartitionConfiger::SaveDim() {
  if (dim_values_.size() > 0) {
    std::string out_file_name = MakeConfigFileName(dim_filename_);
    std::ofstream out_file(out_file_name);
    const std::ostream_iterator<int> out_iter(out_file, " ");
    std::copy(dim_values_.begin(), dim_values_.end(), out_iter);
    out_file.flush();
    out_file.close();

    LOG(INFO) << "PartitionConfiger: Write " << dim_values_.size() << " values";
#ifdef CODL_ENABLE_DEBUG
    LOG(INFO) << "PartitionConfiger: Write file " << out_file_name;
#endif

    return 1;
  } else {
    return 0;
  }
}

int PartitionConfiger::SaveRatio() {
  if (ratio_values_.size() > 0) {
    std::string out_file_name = MakeConfigFileName(ratio_filename_);
    std::ofstream out_file(out_file_name);
    const std::ostream_iterator<float> out_iter(out_file, " ");
    std::copy(ratio_values_.begin(), ratio_values_.end(), out_iter);
    out_file.flush();
    out_file.close();

    LOG(INFO) << "PartitionConfiger: Write " << ratio_values_.size() << " values";
#ifdef CODL_ENABLE_DEBUG
    LOG(INFO) << "PartitionConfiger: Write file " << out_file_name;
#endif

    return 1;
  } else {
    return 0;
  }
}

void PartitionConfiger::Save() {
  int ret;
  ret = SaveDim();
  if (!ret) {
    return;
  }
  ret = SaveRatio();
  if (!ret) {
    return;
  }
  is_file_open_ = true;
}

std::string PartitionConfiger::to_string() const {
  std::string dim_str = VectorToString<int>(dim_values_);
  std::string ratio_str = VectorToString<float>(ratio_values_);
  return MakeString(dim_str, ",", ratio_str);
}

}  // namespace mace
