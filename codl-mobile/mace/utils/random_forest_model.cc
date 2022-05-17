
#include <fstream>
#include "mace/utils/random_forest_model.h"

//#define CODL_ENABLE_DEBUG_INFO

namespace mace {
namespace utils {

void RandomForestModel::Build() {
  for (size_t i = 0; i < num_estimators_; i ++) {
    std::shared_ptr<RFTree> tree(RFTree::Build(max_depth_, num_features_));
    trees_.push_back(tree);
  }
}

void RandomForestModel::BuildFromJson(const char *json_filepath) {
  std::ifstream in(json_filepath);
  MACE_CHECK(!in.fail(), "Open json file failed");

  LOG(INFO) << "Build RF model named " << name_ << " from " << json_filepath;

  auto forest = json::parse(in).as_object();

  max_depth_ = 0;
  int i = 0;
  for (auto iter = forest.begin(); iter != forest.end(); ++iter) {
    auto &tree = *iter;
    RFTree *rf_tree = RFTree::BuildFromJson(tree.second.as_object());
    trees_.emplace_back(rf_tree);
    if (rf_tree->depth() > max_depth_) {
      max_depth_ = rf_tree->depth();
    }
#if 1
    LOG(INFO) << "Build tree idx " << i
              << ", name " << tree.first
              << ", depth " << rf_tree->depth();
#endif
    i ++;
  }
}

double RFTree::Predict(const std::vector<double> &features,
                       int *out_depth) {
  if (out_depth != nullptr) {
    *out_depth = *out_depth + 1;
  }
  
  if (left_tree_ == nullptr) {
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "Tree predict: return " << leaf_value_;
#endif
    return leaf_value_;
  } else if (features[feature_idx_] <= split_value_) {
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "Tree predict:"
              << " fid " << feature_idx_
              << " feature " << features[feature_idx_]
              << " split_value " << split_value_;
#endif
    return left_tree_->Predict(features, out_depth);
  } else {
#ifdef CODL_ENABLE_DEBUG_INFO
    LOG(INFO) << "Tree predict:"
              << " fid " << feature_idx_
              << " feature " << features[feature_idx_]
              << " split_value " << split_value_;
#endif
    return right_tree_->Predict(features, out_depth);
  }
}

double RFTree::Predict(
    const std::vector<std::pair<std::string, double>> &features,
    int *out_depth) {
  if (out_depth != nullptr) {
    *out_depth = *out_depth + 1;
  }

  if (left_tree_ != nullptr) {
    double feature_value = 0;
    for (auto iter = features.begin(); iter != features.end(); ++ iter) {
      auto feature = *iter;
      std::string feature_name = feature.first;
      if (feature_name == feature_name_) {
        feature_value = feature.second;
        break;
      }
    }
    if (feature_value <= split_value_) {
      return left_tree_->Predict(features, out_depth);
    } else {
      return right_tree_->Predict(features, out_depth);
    }
  } else {
    return leaf_value_;
  }
}

double RandomForestModel::Predict(const std::vector<double> &features,
                                  double *out_avg_depth) const {
  if (trees_.size() == 0) {
    LOG(INFO) << "Trees size is 0";
    return 0.0;
  }

  double sum = 0.0;
  double depth_sum = 0;
  for (size_t i = 0; i < trees_.size(); i ++) {
    int depth = 1;
    sum += trees_[i]->Predict(features, &depth);
    depth_sum += depth;
  }
  
  if (out_avg_depth != nullptr) {
    *out_avg_depth = depth_sum / trees_.size();
  }
  
  return sum / trees_.size();
}

double RandomForestModel::Predict(
    const std::vector<std::pair<std::string, double>> &features,
    double *out_avg_depth) const {
  double sum = 0.0;
  double depth_sum = 0;
  for (size_t i = 0; i < trees_.size(); i ++) {
    int depth = 1;
    const double predict_value = trees_[i]->Predict(features, &depth);
#if 0
    LOG(INFO) << "tree_idx " << i << ", predict_value " << predict_value;
#endif
    sum += predict_value;
    depth_sum += depth;
  }

  if (out_avg_depth != nullptr) {
    *out_avg_depth = depth_sum / trees_.size();
  }

  return sum / trees_.size();
}

void RandomForestModel::DebugPrint() const {
  const size_t num_trees = trees_.size();


  LOG(INFO) << "name " << name_
            << ", num_trees " << num_trees
            << ", max_depth " << max_depth_;
}

}  // namespace utils
}  // namespace mace

#ifdef CODL_ENABLE_DEBUG_INFO
#undef CODL_ENABLE_DEBUG_INFO
#endif
