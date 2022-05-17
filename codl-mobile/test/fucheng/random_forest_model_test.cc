
#include <unistd.h>
#include "gflags/gflags.h"
#include "mace/utils/random_forest_model.h"
#include "test/fucheng/random_forest_model_test_param.h"

namespace mace {

MaceStatus RandomForestModelTest(const TestParam *test_param) {
  if (test_param == nullptr) {
    return MaceStatus::MACE_SUCCESS;
  }

  const RandomForestModelTestParam *rfm_test_param =
      reinterpret_cast<const RandomForestModelTestParam *>(test_param);

  const size_t num_features = rfm_test_param->num_features;
  const size_t num_estimators = rfm_test_param->num_estimators;
  const size_t max_depth = rfm_test_param->max_depth;

  const std::vector<double> features(num_features, 0.0);
  utils::RandomForestModel model("EXAMPLE", num_features, num_estimators, max_depth);
  model.Build();
  double result = 0.0;
  double avg_duration = 0.0;
  for (int i = 0; i < rfm_test_param->total_rounds; i ++) {
    double avg_depth = 1;
    const int64_t t0 = NowMicros();
    result = model.Predict(features, &avg_depth);
    const double duration = (NowMicros() - t0) / 1000.0;
    if (i > 10) {
      avg_duration += duration;
    }
    LOG(INFO) << "Time " << duration << " ms, avg_depth " << avg_depth;
  }

  avg_duration /= (rfm_test_param->total_rounds - 10);
  LOG(INFO) << "Time (avg) " << avg_duration << " ms";

  LOG(INFO) << "Predict " << result;

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus RandomForestModelJsonTest() {
  utils::RandomForestModel model("EXAMPLE");
  model.BuildFromJson("random_forest_example.json");

  std::vector<std::pair<std::string, double>> features;
  features.emplace_back("IH", 418);
  features.emplace_back("IW", 418);
  features.emplace_back("IC", 3);
  features.emplace_back("OC", 32);
  features.emplace_back("KH", 3);
  features.emplace_back("KW", 3);
  features.emplace_back("SH", 1);
  features.emplace_back("SW", 1);

  const double value = model.Predict(features);

  LOG(INFO) << "Predicted value: " << value;

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus RandomForestModelJsonRunTest(
    const std::string &path,
    const std::vector<std::string> &feature_names,
    const std::vector<double> &feature_values,
    const int rounds) {
  utils::RandomForestModel model;
  model.BuildFromJson(path.c_str());
  model.DebugPrint();

  std::vector<std::pair<std::string, double>> features;
  const size_t num_features = feature_names.size();
  for (size_t i = 0; i < num_features; i ++) {
    features.emplace_back(feature_names[i], feature_values[i]);
  }

  const int warmup_rounds = 10;
  double avg_duration = 0;
  double result = 0;
  for (int i = 0; i < (warmup_rounds + rounds); i ++) {
    double avg_depth;
    const int64_t t0 = NowMicros();
    
    result = model.Predict(features, &avg_depth);

    const double duration = (NowMicros() - t0) / 1000.0;
    LOG(INFO) << "Duration " << duration << " ms, avg_depth " << avg_depth;
    if (i > warmup_rounds) {
      avg_duration += duration;
    }
  }

  avg_duration /= rounds;

  LOG(INFO) << "Avg duration " << avg_duration << " ms, result " << result;

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace

DEFINE_string(test_case, "", "test case");
DEFINE_int32(num_features, 1, "number of features");
DEFINE_int32(num_estimators, 1, "number of estimators");
DEFINE_int32(max_depth, 1, "maximum depth");
DEFINE_string(model_path, "", "model path");
DEFINE_string(fnames, "", "feature names");
DEFINE_string(fvalues, "", "feature values");
DEFINE_int32(rounds, 1, "rounds");

int PrintFlags() {
  LOG(INFO) << "test_case: " << FLAGS_test_case;
  LOG(INFO) << "model_path: " << FLAGS_model_path;
  LOG(INFO) << "fnames: " << FLAGS_fnames;
  LOG(INFO) << "fvalues: " << FLAGS_fvalues;
  LOG(INFO) << "rounds: " << FLAGS_rounds;
  return 0;
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  PrintFlags();
  const std::string test_case = FLAGS_test_case;
  if (!test_case.compare("rf_run")) {
    mace::RandomForestModelTestParam param;
    param.num_features = FLAGS_num_features;
    param.num_estimators = FLAGS_num_estimators;
    param.max_depth = FLAGS_max_depth;
    param.total_rounds = FLAGS_rounds;
    mace::RandomForestModelTest(&param);
  } else if (!test_case.compare("rf_load_and_predict")) {
    mace::RandomForestModelJsonTest();
  } else if (!test_case.compare("rf_load_and_run")) {
    const std::string model_path = FLAGS_model_path;
    const std::vector<std::string> fnames
        = mace::ArgumentUtils::ParseFeatureNameString(FLAGS_fnames);
    const std::vector<double> fvalues
        = mace::ArgumentUtils::ParseFeatureValueString(FLAGS_fvalues);
    const int rounds = FLAGS_rounds;
    mace::RandomForestModelJsonRunTest(model_path, fnames, fvalues, rounds);
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  
  return 0;
}
