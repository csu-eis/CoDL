
#ifndef MACE_UTILS_RANDOM_FOREST_MODEL_H_
#define MACE_UTILS_RANDOM_FOREST_MODEL_H_

#include <memory>
#include <vector>
#include "mace/utils/logging.h"
#include "mace/utils/predict_model.h"
#include "third_party/cppjson/cpp_json.h"

namespace mace {
namespace utils {

typedef struct {
  size_t num_features;
  size_t num_estimators;
  size_t max_depth;
  size_t rounds;
} RandomForestParam;

class RFTree {
 public:
  RFTree()
      : depth_(1),
        feature_idx_(0),
        feature_name_(""),
        split_value_(0),
        leaf_value_(0),
        left_tree_(nullptr),
        right_tree_(nullptr) {}

  RFTree(const size_t feature_idx,
         const double split_value,
         const double leaf_value = 0)
      : depth_(1),
        feature_idx_(feature_idx),
        feature_name_(""),
        split_value_(split_value),
        leaf_value_(leaf_value),
        left_tree_(nullptr),
        right_tree_(nullptr) {}

  RFTree(const std::string feature_name,
         const double split_value,
         const double leaf_value = 0)
      : depth_(1),
        feature_idx_(0),
        feature_name_(feature_name),
        split_value_(split_value),
        leaf_value_(leaf_value),
        left_tree_(nullptr),
        right_tree_(nullptr) {}

  static RFTree *Build(const size_t max_depth,
                       const size_t num_features) {
    MACE_UNUSED(num_features);
    RFTree *root = new RFTree(0, 0.5, 0);
    RFTree::BuildInternal(root, 1, max_depth);
    return root;
  }

  static RFTree *BuildFromJson(const json::object &tree) {
    RFTree *root = new RFTree();
    int depth = 1;
    RFTree::BuildFromJsonInternal(root, tree, 1, depth);
    root->set_depth(depth);
    return root;
  }

  inline int depth() const {
    return depth_;
  }

  inline void set_depth(const int d) {
    depth_ = d;
  }

  inline void set_split_feature(const std::string feature) {
    feature_name_ = feature;
  }

  inline void set_split_value(const double v) {
    split_value_ = v;
  }

  inline void set_leaf_value(const double v) {
    leaf_value_ = v;
  }

  inline void set_left_tree(const std::shared_ptr<RFTree> tree) {
    left_tree_ = tree;
  }

  inline void set_right_tree(const std::shared_ptr<RFTree> tree) {
    right_tree_ = tree;
  }

  double Predict(const std::vector<double> &features,
                 int *out_depth = nullptr);

  double Predict(const std::vector<std::pair<std::string, double>> &features,
                 int *out_depth = nullptr);

 private:
  static void BuildInternal(RFTree *tree,
                            const size_t cur_depth,
                            const size_t max_depth) {
    if (cur_depth < (max_depth - 1)) {
      //LOG(INFO) << "depth: " << cur_depth;
      std::shared_ptr<RFTree> left_tree(new RFTree(0, 0.5, 0));
      std::shared_ptr<RFTree> right_tree(new RFTree(0, 0.5, 1));
      tree->set_left_tree(left_tree);
      tree->set_right_tree(right_tree);
      RFTree::BuildInternal(left_tree.get(), cur_depth + 1, max_depth);
    }
  }

  static void BuildFromJsonInternal(RFTree *rf_tree,
                                    const json::object &json_tree,
                                    const int cur_depth,
                                    int &output_depth) {
    if (json_tree.find("left_tree") == json_tree.end() &&
        json_tree.find("right_tree") == json_tree.end()) {
      // Leaf node.
      double leaf_value = json::to_number<double>(json_tree["leaf_value"]);
#if 0
      LOG(INFO) << "leaf_value " << leaf_value;
#endif

      rf_tree->set_leaf_value(leaf_value);

      if (cur_depth + 1 > output_depth) {
        output_depth = cur_depth + 1;
      }
    } else {
      // Non-leaf node.
      json::object &left_json_tree = json_tree["left_tree"].as_object();
      json::object &right_json_tree = json_tree["right_tree"].as_object();
      std::string split_feature = json_tree["split_feature"].as_string();
      double split_value = json::to_number<double>(json_tree["split_value"]);

#if 0
      LOG(INFO) << "split_feature " << split_feature << ", split_value " << split_value;
#endif

      rf_tree->set_split_feature(split_feature);
      rf_tree->set_split_value(split_value);

      std::shared_ptr<RFTree> left_rf_tree(new RFTree());
      std::shared_ptr<RFTree> right_rf_tree(new RFTree());
      rf_tree->set_left_tree(left_rf_tree);
      rf_tree->set_right_tree(right_rf_tree);
      
      BuildFromJsonInternal(left_rf_tree.get(),
                            left_json_tree,
                            cur_depth + 1,
                            output_depth);
      BuildFromJsonInternal(right_rf_tree.get(),
                            right_json_tree,
                            cur_depth + 1,
                            output_depth);
    }
  }

  int depth_;
  size_t feature_idx_;
  std::string feature_name_;
  double split_value_;
  double leaf_value_;
  std::shared_ptr<RFTree> left_tree_;
  std::shared_ptr<RFTree> right_tree_;
};

class RandomForestModel : public PredictModel {
 public:
  RandomForestModel()
      : name_(""),
        num_features_(0),
        num_estimators_(0),
        max_depth_(0) {}

  RandomForestModel(const std::string name)
      : name_(name),
        num_features_(0),
        num_estimators_(0),
        max_depth_(0) {}

  RandomForestModel(const std::string name,
                    const size_t num_features,
                    const size_t num_estimators,
                    const size_t max_depth)
      : name_(name),
        num_features_(num_features),
        num_estimators_(num_estimators),
        max_depth_(max_depth) {}

  void Build();

  void BuildFromJson(const char *json_filepath) override;

  double Predict() const override {
    MACE_NOT_IMPLEMENTED;
    return 0;
  }

  double Predict(const std::vector<double> &features,
                 double *out_avg_depth) const;

  double Predict(const std::vector<double> &features) const override {
    return Predict(features, nullptr);
  }

  double Predict(const std::vector<std::pair<std::string, double>> &features,
                 double *out_avg_depth) const;

  double Predict(const std::vector<std::pair<std::string, double>> &features)
      const override {
    return Predict(features, nullptr);
  }

  void DebugPrint() const override;

 private:
  std::string name_;
  size_t num_features_;
  size_t num_estimators_;
  int max_depth_;
  std::vector<std::shared_ptr<RFTree>> trees_;
};

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_RANDOM_FOREST_MODEL_H_
