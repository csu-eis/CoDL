
#include "mace/utils/cpp_json.h"
#include <iostream>
#include <fstream>
#include <vector>

#define EXAMPLE_JSON_FILENAME "pr_config.json"
#define EXAMPLE_RF_JSON_FILENAME "random_forest_example.json"
#define PR_VALUE_KEY "pr_val"
#define PR_CONFIG_KEY "pr_config"

int DefaultExample() {
  // construct from a file
  std::ifstream file("example1.json");
  if(file) {
    auto v1 = json::parse(file);
    std::cout << stringify(v1, json::PrettyPrint | json::EscapeUnicode) << '\n';
  }
  
  return 0;
}

template <class T>
void PrintVector(const std::vector<T> &vec) {
  std::cout << "[";
  for (const T& v : vec) {
    std::cout << v << ",";
  }
  std::cout << "]";
}

int PrConfigerWriteFileExample() {
  const std::vector<float> pr_val_vec = {0.6, 0.7, 0.8, 0.5, 0.4};
  json::array pr_val_arr;
  // Push value to array
  for (auto it = pr_val_vec.begin(); it != pr_val_vec.end(); ++it) {
    pr_val_arr.push_back(*it);
  }
  json::object pr_configs;
  pr_configs.insert(PR_VALUE_KEY, pr_val_arr);
  // JSON object to string
  std::string json_str = stringify(pr_configs);
  // String to file
  std::ofstream out(EXAMPLE_JSON_FILENAME);
  out << json_str;
  out.close();
  
  return 0;
}

int PrConfigerReadFileExample() {
  // Parse from file
  std::ifstream in(EXAMPLE_JSON_FILENAME);
  const json::object &pr_configs = json::parse(in).as_object();
  // Get array by key
  const json::array pr_val_arr = pr_configs.at(PR_VALUE_KEY).as_array();
  // Iteration
  std::vector<std::string> pr_val_vec;
  //const size_t val_size = pr_val_arr.size();
  //for (size_t i = 0; i < val_size; ++i) {
  //  pr_val_vec.push_back((float) pr_val_arr[i]);
  //}
  for (auto it = pr_val_arr.begin(); it != pr_val_arr.end(); ++it) {
    pr_val_vec.push_back((*it).as_string());
  }
  // How to print a vector?
  PrintVector<std::string>(pr_val_vec);
  
  return 0;
}

int PrConfigerWithJsonExample() {
  // TODO: Complete this example
  PrConfigerWriteFileExample();
  PrConfigerReadFileExample();
  return 0;
}

int TraverseTree(const json::object &tree) {
  if (tree.find("left_tree") == tree.end() &&
      tree.find("right_tree") == tree.end()) {
    // Leaf node.
    auto v = tree["leaf_value"];
    std::cout << "leaf_value " << json::to_number<double>(v) << std::endl;
  } else {
    // Non-leaf node.
    auto left_tree = tree["left_tree"].as_object();
    auto right_tree = tree["right_tree"].as_object();
    auto split_feature = tree["split_feature"];
    auto split_value = tree["split_value"];
    std::cout << "split_feature " << split_feature.as_string()
              << " split_value " << json::to_number<double>(split_value) << std::endl;
    TraverseTree(left_tree);
    TraverseTree(right_tree);
  }

  return 0;
}

int LoadRandomForestExample() {
  std::ifstream in(EXAMPLE_RF_JSON_FILENAME);
  
  //const json::object &forest = json::parse(in).as_object();
  auto forest = json::parse(in).as_object();

  //std::cout << stringify(forest, json::PrettyPrint | json::EscapeUnicode) << std::endl;

  for (auto tree_iter = forest.begin();
      tree_iter != forest.end(); ++ tree_iter) {
    auto &tree = *tree_iter;
    std::cout << tree.first << std::endl;
    auto tree_obj = tree.second.as_object();

    TraverseTree(tree_obj);
  }

  return 0;
}

/**
 * @brief main
 * @return
 */
int main() {
  //DefaultExample();
  //PrConfigerWithJsonExample();
  LoadRandomForestExample();
  
  return 0;
}
