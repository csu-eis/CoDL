
#include "gflags/gflags.h"
#include "test/codl_run/op_chain_latency_predict.h"
#include "test/codl_run/op_chain_search.h"
#include "test/codl_run/op_chain_executor.h"
#include "test/codl_run/op_chain_helper.h"
#include "test/codl_run/nn_model_builder.h"

DEFINE_string(test, "", "test");
DEFINE_int32(chain_idx, 0, "chain idx");
DEFINE_int32(chain_count, 0, "chain count");
DEFINE_int32(chain_param_hint, 0, "chain parameter hint");
DEFINE_int32(op_idx, 0, "op idx");
DEFINE_int32(op_count, -1, "op count");
//DEFINE_int32(part_dim, 0, "0, 1, 2, 3, 4");
DEFINE_int32(cpu_dtype, 1, "cpu data type");
DEFINE_int32(num_threads, 1, "number of thread");
DEFINE_int32(gpu_dtype, 1, "gpu data type");
DEFINE_int32(gpu_mtype, 2, "gpu memory type");
DEFINE_double(part_ratio, 1, "partition ratio");
DEFINE_double(size_ratio, 1, "size ratio");
DEFINE_int32(rounds, 1, "rounds");
DEFINE_int32(latency_acq, 0, "latency acquirement");
DEFINE_int32(lp_backend, 0, "latency prediction backend");
DEFINE_bool(make_partition_plan, false, "make partition plan");
DEFINE_bool(profile_data_transform, false, "profile data transform");
DEFINE_bool(profile_compute, false, "profile compute");
DEFINE_int32(pdim_hint, 0, "partition dimension hint");
DEFINE_int32(pratio_hint, 0, "partition ratio hint");
DEFINE_bool(data_transform, false, "data transform");
DEFINE_bool(compute, true, "compute");
DEFINE_string(search_method, "serial", "search method");
DEFINE_int32(search_baseline, 0, "search baseline");
DEFINE_int32(debug_level, 1, "debug level");

int print_flags() {
  LOG(INFO) << "Test: " << FLAGS_test;
  LOG(INFO) << "Chain idx: " << FLAGS_chain_idx;
  LOG(INFO) << "Chain count: " << FLAGS_chain_count;
  LOG(INFO) << "Chain param hint: " << FLAGS_chain_param_hint;
  LOG(INFO) << "OP idx: " << FLAGS_op_idx;
  LOG(INFO) << "OP count: " << FLAGS_op_count;
  LOG(INFO) << "CPU data type: " << FLAGS_cpu_dtype;
  LOG(INFO) << "Thread number: " << FLAGS_num_threads;
  LOG(INFO) << "GPU data type: " << FLAGS_gpu_dtype;
  LOG(INFO) << "GPU memory type: " << FLAGS_gpu_mtype;
  LOG(INFO) << "Partition ratio: " << FLAGS_part_ratio;
  LOG(INFO) << "Rounds: " << FLAGS_rounds;
  LOG(INFO) << "Latency acquirement: " << FLAGS_latency_acq;
  LOG(INFO) << "Latency prediction backend: " << FLAGS_lp_backend;
  LOG(INFO) << "Make partition plan: " << FLAGS_make_partition_plan;
  LOG(INFO) << "Profile data transform: " << FLAGS_profile_data_transform;
  LOG(INFO) << "Profile compute: " << FLAGS_profile_compute;
  LOG(INFO) << "Partition dimension hint: " << FLAGS_pdim_hint;
  LOG(INFO) << "Partition ratio hint: " << FLAGS_pratio_hint;
  LOG(INFO) << "Data transform: " << FLAGS_data_transform;
  LOG(INFO) << "Compute: " << FLAGS_compute;
  LOG(INFO) << "Search method: " << FLAGS_search_method;
  LOG(INFO) << "Search baseline: " << FLAGS_search_baseline;
  LOG(INFO) << "Debug level: " << FLAGS_debug_level;
  return 0;
}

#define APPEND_CONV2D(...) \
  params.emplace_back(new CodlConv2dChainParam( \
      __VA_ARGS__, common_param));

#define APPEND_MATMUL(...) \
  params.emplace_back(new CodlMatMulChainParam(__VA_ARGS__, common_param));

int build_mobilenetv2(const int pdim,
                      const float pratio,
                      std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = DataType::DT_FLOAT;
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = DataType::DT_FLOAT;
  common_param.gpu_mtype = MemoryType::GPU_IMAGE;
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;
  APPEND_CONV2D(224, 224, 3, 32, 3, 3, 2, 2, pdim, pratio);
  APPEND_CONV2D(112, 112, 32, 16, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(112, 112, 16, 96, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(56, 56, 96, 24, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(56, 56, 24, 144, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(56, 56, 144, 24, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(56, 56, 24, 144, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(28, 28, 144, 32, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(28, 28, 32, 192, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(28, 28, 192, 32, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(28, 28, 32, 192, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(28, 28, 192, 32, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(28, 28, 32, 192, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 192, 64, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 64, 384, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 384, 64, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 64, 384, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 384, 64, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 64, 384, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 384, 64, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 64, 384, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 384, 96, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 96, 576, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 576, 96, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 96, 576, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 576, 96, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(14, 14, 96, 576, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(7, 7, 576, 160, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(7, 7, 160, 960, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(7, 7, 960, 160, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(7, 7, 160, 960, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(7, 7, 960, 160, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(7, 7, 160, 960, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(7, 7, 960, 320, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(7, 7, 320, 1280, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(1, 1, 1280, 1001, 1, 1, 1, 1, pdim, pratio);
  return 0;
}

int build_resnet50v2_chain1(const int pdim,
                            const float pratio,
                            const float sratio,
                            std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = DataType::DT_FLOAT;
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = DataType::DT_FLOAT;
  common_param.gpu_mtype = MemoryType::GPU_IMAGE;
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;
  const index_t in_size = static_cast<index_t>(56 * sratio);
  APPEND_CONV2D(in_size, in_size, 64, 64, 1, 1, 1, 1, pdim, pratio);
  APPEND_CONV2D(in_size + 2, in_size + 2, 64, 64, 3, 3, 1, 1, pdim, pratio);
  APPEND_CONV2D(in_size, in_size, 64, 256, 1, 1, 1, 1, pdim, pratio);
  return 0;
}

int build_matmul_net(const int pdim,
                     const float pratio,
                     std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;
  APPEND_MATMUL(1, 256, 256, 256, false, false, pdim, pratio);
  APPEND_MATMUL(1, 512, 512, 512, false, false, pdim, pratio);
  APPEND_MATMUL(1, 1024, 1024, 1024, false, false, pdim, pratio);
  APPEND_MATMUL(1, 2048, 2048, 2048, false, false, pdim, pratio);
  return 0;
}

int build_bert(const int pdim,
               const float pratio,
               std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;
  APPEND_MATMUL(1, 256, 768, 2, false, false, pdim, pratio);
  APPEND_MATMUL(1, 256, 768, 768, false, false, pdim, pratio);
  APPEND_MATMUL(12, 256, 256, 64, false, false, pdim, pratio);
  APPEND_MATMUL(12, 256, 64, 256, false, true, pdim, pratio);
  APPEND_MATMUL(1, 256, 3072, 768, false, false, pdim, pratio);
  APPEND_MATMUL(1, 256, 768, 3072, false, false, pdim, pratio);
  APPEND_MATMUL(1, 1, 768, 768, false, false, pdim, pratio);
  APPEND_MATMUL(1, 1, 768, 2, false, true, pdim, pratio);
  return 0;
}

int example_yolov2_cpu_only() {
  const int pdim = 1;
  const float pratio = 0.0;
  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;

  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  std::shared_ptr<NnModelBuilder> builder(new Yolov2Builder(common_param));
  builder->Build(pdim, pratio, params);

  mace::CodlOpTaskChain op_chain;
  for (size_t i = 0; i < params.size(); i ++) {
    op_chain.Append(params[i].get());
  }

  for (int i = 0; i < FLAGS_rounds; i ++) {
    int64_t t0 = NowMicros();
    op_chain.SerialRun();
    double duration = (NowMicros() - t0) / 1000.0;
    LOG(INFO) << "ExampleYolov2: cpu only " << duration << " ms";
  }

  return 0;
}

int example_yolov2_gpu_only() {
  const int pdim = 1;
  const float pratio = 1.0;
  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;

  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  std::shared_ptr<NnModelBuilder> builder(new Yolov2Builder(common_param));
  builder->Build(pdim, pratio, params);

  mace::CodlOpTaskChain op_chain;
  for (size_t i = 0; i < params.size(); i ++) {
    op_chain.Append(params[i].get());
  }

  for (int i = 0; i < FLAGS_rounds; i ++) {
    int64_t t0 = NowMicros();
    op_chain.SerialRun();
    double duration = (NowMicros() - t0) / 1000.0;
    LOG(INFO) << "ExampleYolov2: gpu only " << duration << " ms";
  }

  return 0;
}

int example_yolov2_serial() {
  const int pdim = 1;
  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;

  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  APPEND_CONV2D(418, 418, 3, 32, 3, 3, 1, 1, pdim, 1.0);
  APPEND_CONV2D(210, 210, 32, 64, 3, 3, 1, 1, pdim, 0.7);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdim, 0.5);
  APPEND_CONV2D(104, 104, 128, 64, 1, 1, 1, 1, pdim, 1.0);
  APPEND_CONV2D(106, 106, 64, 128, 3, 3, 1, 1, pdim, 0.5);
  APPEND_CONV2D(54, 54, 128, 256, 3, 3, 1, 1, pdim, 0.5);
  APPEND_CONV2D(52, 52, 256, 128, 1, 1, 1, 1, pdim, 1.0);
  APPEND_CONV2D(54, 54, 128, 256, 3, 3, 1, 1, pdim, 0.5);
  APPEND_CONV2D(28, 28, 256, 512, 3, 3, 1, 1, pdim, 0.5);
  APPEND_CONV2D(26, 26, 512, 256, 1, 1, 1, 1, pdim, 1.0);
  APPEND_CONV2D(28, 28, 256, 512, 3, 3, 1, 1, pdim, 0.5);
  APPEND_CONV2D(26, 26, 512, 256, 1, 1, 1, 1, pdim, 1.0);
  APPEND_CONV2D(28, 28, 256, 512, 3, 3, 1, 1, pdim, 0.5);

  mace::CodlOpTaskChain op_chain;
  for (size_t i = 0; i < params.size(); i ++) {
    op_chain.Append(params[i].get());
  }

  for (int i = 0; i < FLAGS_rounds; i ++) {
    mace::DurationCollector<double> dura_collector;
    //int64_t t0 = NowMicros();
    op_chain.SerialRun(&dura_collector);
    //double duration = (NowMicros() - t0) / 1000.0;

    double duration = 0;
    for (size_t j = 0; j < dura_collector.Size(); j ++) {
      const std::vector<double> stat = dura_collector.Get(j);
      LOG(INFO) << "Stat " << VectorToString<double>(stat) << " ms";
      duration += CalcOpDuration(stat);
    }
    
    LOG(INFO) << "ExampleYolov2: serial " << duration << " ms";
  }

  return 0;
}

int example_yolov2_chain() {
  const int pdim = 1;
  const float pratio = FLAGS_part_ratio;
  const int chain_idx = FLAGS_chain_idx;
  const int rounds = FLAGS_rounds;
  const int debug_level = FLAGS_debug_level;
  const LPBackend lp_backend = static_cast<LPBackend>(FLAGS_lp_backend);
  
  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;
  LOG(INFO) << CodlOpChainCommonParamToString(common_param);

  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  
  std::shared_ptr<NnModelBuilder> builder(new Yolov2Builder(common_param));
  builder->Build(chain_idx, pdim, pratio, params);

  mace::CodlOpTaskChain op_chain;
  for (size_t i = 0; i < params.size(); i ++) {
    op_chain.Append(params[i].get());
  }

  for (int i = 0; i < rounds; i ++) {
    mace::DurationCollector<double> dura_collector;
    //int64_t t0 = NowMicros();
    op_chain.SerialRun(&dura_collector);
    //double duration = (NowMicros() - t0) / 1000.0;

    double duration = 0, comp_duration = 0;
    for (size_t j = 0; j < dura_collector.Size(); j ++) {
      const std::vector<double> stat = dura_collector.Get(j);
      //LOG(INFO) << "Stat " << VectorToString<double>(stat) << " ms";
      LOG(INFO) << "op " << j
                << ", type " << CodlOpTypeToString(params[j]->op_type())
                << ", duration " << OpDurationToString(stat) << " ms";
      duration += CalcOpDuration(stat);
      comp_duration += CalcOpComputeDuration(stat);
    }

    LOG(INFO) << "ExampleYolov2: serial " << duration << " ms, comp "
              << comp_duration << " ms";
  }

  double lat_chain;
  LPInitContext lp_init_context;
  lp_init_context.set_backend(lp_backend);
  mace::OpChainLatencyPredictor chain_predictor;
  chain_predictor.Init(&lp_init_context);
  chain_predictor.Predict(op_chain, debug_level, &lat_chain);
  LOG(INFO) << "Predicted chain latency: " << lat_chain << " ms";

  op_chain.Prepare();

  for (int i = 0; i < rounds; i ++) {
    mace::DurationCollector<double> dura_collector;
    //int64_t t0 = NowMicros();
    op_chain.Run(&dura_collector);
    //double duration = (NowMicros() - t0) / 1000.0;
    
    double duration = 0, comp_duration = 0;
    for (size_t j = 0; j < dura_collector.Size(); j ++) {
      const std::vector<double> stat = dura_collector.Get(j);
      //LOG(INFO) << "Stat " << VectorToString<double>(stat) << " ms";
      LOG(INFO) << "op " << j
                << ", type " << CodlOpTypeToString(params[j]->op_type())
                << ", duration " << OpDurationToString(stat) << " ms";
      duration += CalcOpDuration(stat);
      comp_duration += CalcOpComputeDuration(stat);
    }
    
    LOG(INFO) << "ExampleYolov2: chain " << duration << " ms, comp "
              << comp_duration << " ms";
  }

  return 0;
}

int example_yolov2_same_size(const float pratio) {
  const int pdim = 1;
  const int op_count = 20;
  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;

  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  for (int i = 0; i < op_count; i++) {
    APPEND_CONV2D(28, 28, 256, 512, 3, 3, 1, 1, pdim, pratio);
  }

  mace::CodlOpTaskChain op_chain;
  for (int i = 0; i < op_count; i++) {
    op_chain.Append(params[i].get());
  }

  for (int i = 0; i < FLAGS_rounds; i ++) {
    int64_t t0 = NowMicros();
    op_chain.SerialRun();
    double duration = (NowMicros() - t0) / 1000.0;
    LOG(INFO) << "ExampleYolov2: p " << pratio << ", " << duration << " ms";
  }

  return 0;
}

int example_yolov2_same_size_chain() {
  const int pdim = 1;
  const float pratio = FLAGS_part_ratio;
  const int op_count = FLAGS_op_count;
  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;

  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  for (int i = 0; i < op_count; i++) {
    APPEND_CONV2D(28, 28, 256, 512, 3, 3, 1, 1, pdim, pratio);
  }

  mace::CodlOpTaskChain op_chain;
  for (int i = 0; i < op_count; i++) {
    op_chain.Append(params[i].get());
  }

  for (int i = 0; i < FLAGS_rounds; i ++) {
    mace::DurationCollector<double> dura_collector;
    int64_t t0 = NowMicros();
    op_chain.SerialRun(&dura_collector);
    double duration = (NowMicros() - t0) / 1000.0;
    std::vector<double> stat = dura_collector.StatSum();
    LOG(INFO) << "ExampleYolov2: serial " << duration << " ms"
              << ", stat " << VectorToString<double>(stat) << " ms";
  }

  for (int i = 0; i < FLAGS_rounds; i ++) {
    mace::DurationCollector<double> dura_collector;
    int64_t t0 = NowMicros();
    op_chain.Run(&dura_collector);
    double duration = (NowMicros() - t0) / 1000.0;
    std::vector<double> stat = dura_collector.StatAvg();
    LOG(INFO) << "ExampleYolov2: chain " << duration << " ms"
              << ", stat " << VectorToString<double>(stat) << " ms";
  }

  return 0;
}

#undef APPEND_MATMUL
#undef APPEND_CONV2D

int example_yolov2_chain_search() {
  const int chain_idx = FLAGS_chain_idx;
  const int chain_param_hint = FLAGS_chain_param_hint;
  const int op_idx = FLAGS_op_idx;
  const int op_count = FLAGS_op_count;
  const bool do_make_partition_plan = FLAGS_make_partition_plan;
  const bool profile_data_transform = FLAGS_profile_data_transform;
  const bool profile_compute = FLAGS_profile_compute;
  const int pdim_hint = FLAGS_pdim_hint;
  const int pratio_hint = FLAGS_pratio_hint;
  const int rounds = FLAGS_rounds;
  const std::string search_method = FLAGS_search_method;
  const int search_baseline = FLAGS_search_baseline;
  const LatencyAcquirement lat_acq = static_cast<LatencyAcquirement>(FLAGS_latency_acq);
  const LPBackend lp_backend = static_cast<LPBackend>(FLAGS_lp_backend);
  const int debug_level = FLAGS_debug_level;

  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;

  std::vector<std::shared_ptr<CodlOpChainParam>> params;

  Yolov2Builder builder(common_param);
  builder.Build(chain_idx, chain_param_hint, params);
  
  if (op_idx > -1 && op_count > -1) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    for (int i = op_idx; i < (op_idx + op_count)
        && i < static_cast<int>(params.size()); i ++) {
      out_params.emplace_back(params[i]);
    }
    params = out_params;
  }

  if (do_make_partition_plan) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    mace::ChainSearch::OptimalPartitionProfileOrPredict(params,
                                                        lat_acq,
                                                        lp_backend,
                                                        profile_data_transform,
                                                        profile_compute,
                                                        pdim_hint,
                                                        pratio_hint,
                                                        debug_level,
                                                        out_params);
    params = out_params;
  }

  std::vector<mace::CodlOpTaskChain> op_chains;
  
  if (search_method.compare("serial") == 0) {
    op_chains.clear();
    mace::ChainSearch::Serial(params, op_chains);
    mace::CodlOpTaskChainExecutor::Execute(op_chains, rounds,
                                           "yolov2, serial", debug_level);
  }

  if (search_method.compare("heuristic") == 0) {
    op_chains.clear();
    mace::ChainSearch::Heuristic(params, op_chains);
    mace::CodlOpTaskChainExecutor::Execute(op_chains, rounds,
                                           "yolov2, heuristic", debug_level);
  }

  if (search_method.compare("greedy") == 0) {
    op_chains.clear();
    mace::ChainSearch::Greedy(params, lat_acq, search_baseline,
                              pratio_hint, debug_level, op_chains);
    mace::CodlOpTaskChainExecutor::Execute(op_chains, rounds,
                                           "yolov2, greedy", debug_level);
  }

  if (search_method.compare("full") == 0) {
    op_chains.clear();
    mace::ChainSearch::Full(params, op_chains);
    mace::CodlOpTaskChainExecutor::Execute(op_chains, rounds,
                                           "yolov2, full", debug_level);
  }

  double lat_chain;
  LPInitContext lp_init_context;
  lp_init_context.set_backend(lp_backend);
  mace::OpChainLatencyPredictor chain_predictor;
  chain_predictor.Init(&lp_init_context);
  chain_predictor.Predict(op_chains, debug_level, &lat_chain);
  LOG(INFO) << "Predicted chain latency: " << lat_chain << " ms";

  mace::CodlOpTaskChainHelper::HardFree(op_chains);

  return 0;
}

int example_yolov2_real_chain_search() {
  const bool do_make_partition_plan = FLAGS_make_partition_plan;
  const bool profile_data_transform = FLAGS_profile_data_transform;
  const bool profile_compute = FLAGS_profile_compute;
  const std::string search_method = FLAGS_search_method;
  const int search_baseline = FLAGS_search_baseline;
  const int chain_param_hint = FLAGS_chain_param_hint;
  int chain_idx = FLAGS_chain_idx;
  size_t chain_count = static_cast<size_t>(FLAGS_chain_count);
  const int pdim_hint = FLAGS_pdim_hint;
  const int pratio_hint = FLAGS_pratio_hint;
  const LatencyAcquirement lat_acq = static_cast<LatencyAcquirement>(FLAGS_latency_acq);
  const LPBackend lp_backend = static_cast<LPBackend>(FLAGS_lp_backend);
  const int debug_level = FLAGS_debug_level;

  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;

  Yolov2Builder builder(common_param);
  const size_t model_chain_count = builder.op_chain_count();
  if (chain_idx < 0) {
    chain_idx = 0;
  }
  if (chain_count <= 0 || chain_count > model_chain_count) {
    chain_count = model_chain_count;
  }

  std::vector<double> nn_lats(3, 0);
  for (size_t i = chain_idx; i < (chain_idx + chain_count); i ++) {
    std::vector<std::shared_ptr<CodlOpChainParam>> params;
    std::vector<mace::CodlOpTaskChain> op_chains;
    std::vector<double> out_lats;

    builder.Build(i, chain_param_hint, params);

    if (do_make_partition_plan) {
      std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
      mace::ChainSearch::OptimalPartitionProfileOrPredict(params,
                                                          lat_acq,
                                                          lp_backend,
                                                          profile_data_transform,
                                                          profile_compute,
                                                          pdim_hint,
                                                          pratio_hint,
                                                          debug_level,
                                                          out_params);
      params = out_params;
    }
    
    if (search_method.compare("serial") == 0) {
      mace::ChainSearch::Serial(params, op_chains);
      mace::CodlOpTaskChainExecutor::Execute(op_chains, FLAGS_rounds,
                                             "yolov2, serial", debug_level, out_lats);
    }

    if (search_method.compare("heuristic") == 0) {
      mace::ChainSearch::Heuristic(params, op_chains);
      mace::CodlOpTaskChainExecutor::Execute(op_chains, FLAGS_rounds,
                                             "yolov2, heuristic", debug_level, out_lats);
    }

    if (search_method.compare("greedy") == 0) {
      mace::ChainSearch::Greedy(params, lat_acq, search_baseline,
                                pratio_hint, debug_level, op_chains);
      mace::CodlOpTaskChainExecutor::Execute(op_chains, FLAGS_rounds,
                                             "yolov2, greedy", debug_level, out_lats);
    }

    if (search_method.compare("full") == 0) {
      mace::ChainSearch::Full(params, op_chains);
      mace::CodlOpTaskChainExecutor::Execute(op_chains, FLAGS_rounds,
                                             "yolov2, full", debug_level, out_lats);
    }

    params.clear();
    op_chains.clear();

    for (size_t j = 0; j < out_lats.size(); j ++) {
      nn_lats[j] += out_lats[j];
    }
  }

  LOG(INFO) << "total " << nn_lats[0] << " ms"
            << ", dms " << nn_lats[1] << " ms"
            << ", comp " << nn_lats[2] << " ms";

  return 0;
}

int example_nn_model_chain_search(const std::string &model_name) {
  const int chain_idx = FLAGS_chain_idx;
  const int chain_param_hint = FLAGS_chain_param_hint;
  const int op_idx = FLAGS_op_idx;
  int op_count = FLAGS_op_count;
  const int pdim = 1;
  const float pratio = FLAGS_part_ratio;
  const float sratio = FLAGS_size_ratio;
  const bool do_make_partition_plan = FLAGS_make_partition_plan;
  const bool profile_data_transform = FLAGS_profile_data_transform;
  const bool profile_compute = FLAGS_profile_compute;
  const int pdim_hint = FLAGS_pdim_hint;
  const int pratio_hint = FLAGS_pratio_hint;
  const std::string search_method = FLAGS_search_method;
  const int search_baseline = FLAGS_search_baseline;
  const int rounds = FLAGS_rounds;
  const LatencyAcquirement lat_acq = static_cast<LatencyAcquirement>(FLAGS_latency_acq);
  const LPBackend lp_backend = static_cast<LPBackend>(FLAGS_lp_backend);
  const int debug_level = FLAGS_debug_level;

  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;

  std::vector<std::shared_ptr<CodlOpChainParam>> src_params;

  if (!model_name.compare("yolov2")) {
    std::shared_ptr<NnModelBuilder> builder(
        new Yolov2Builder(common_param));
    //builder->Build(pdim, pratio, src_params);
    builder->Build(chain_idx, chain_param_hint, src_params);
  } else if (!model_name.compare("posenet")) {
    std::shared_ptr<NnModelBuilder> builder(
        new PosenetBuilder(common_param));
    builder->Build(chain_idx, chain_param_hint, src_params);
  } else if (!model_name.compare("alexnet")) {
    std::shared_ptr<NnModelBuilder> builder(
        new AlexnetBuilder(common_param));
    builder->Build(chain_idx, chain_param_hint, src_params);
  } else if (!model_name.compare("vgg16")) {
    std::shared_ptr<NnModelBuilder> builder(
        new Vgg16Builder(common_param));
    //builder->Build(chain_idx, pdim, pratio, src_params);
    builder->Build(chain_idx, chain_param_hint, src_params);
  } else if (!model_name.compare("fast_style_transfer")) {
    std::shared_ptr<NnModelBuilder> builder(
        new FastStyleTransferBuilder(common_param));
    builder->Build(chain_idx, chain_param_hint, src_params);
  } else if (!model_name.compare("retinaface")) {
    std::shared_ptr<NnModelBuilder> builder(
        new RetinaFaceBuilder(common_param));
    builder->Build(chain_idx, chain_param_hint, src_params);
  } else if (!model_name.compare("mobilenetv1")) {
    std::shared_ptr<NnModelBuilder> builder(
        new MobileNetV1Builder(common_param));
    builder->Build(chain_idx, chain_param_hint, src_params);
  } else if (!model_name.compare("mobilenetv2")) {
    build_mobilenetv2(pdim, pratio, src_params);
  } else if (!model_name.compare("resnet50v1")) {
    std::shared_ptr<NnModelBuilder> builder(
        new Resnet50v1Builder(common_param));
    builder->Build(chain_idx, chain_param_hint, src_params);
  } else if (!model_name.compare("resnet50v2")) {
    if (chain_idx == 1) {
      build_resnet50v2_chain1(pdim, pratio, sratio, src_params);
    }
  } else if (!model_name.compare("matmul_net")) {
    build_matmul_net(pdim, pratio, src_params);
  } else if (!model_name.compare("bert")) {
    //build_bert(pdim, pratio, src_params);
    std::shared_ptr<NnModelBuilder> builder(
        new BertBuilder(common_param, 24, 1024, 16));
    builder->Build(chain_idx, chain_param_hint, src_params);
  } else if (!model_name.compare("op_chain_net")) {
    std::shared_ptr<NnModelBuilder> builder(
        new OpChainNetBuilder(common_param));
    builder->Build(chain_idx, chain_param_hint, src_params);
  } else {
    LOG(INFO) << "Unsupported NN model " << model_name;
    return 0;
  }

  if (op_count == -1) {
    op_count = static_cast<int>(src_params.size());
  }

  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  for (int i = 0; i < static_cast<int>(src_params.size()); i ++) {
    if (i >= op_idx && i < (op_idx + op_count)) {
      params.emplace_back(src_params[i]);
    }
  }

  if (do_make_partition_plan) {
    std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
    mace::ChainSearch::OptimalPartitionProfileOrPredict(params,
                                                        lat_acq,
                                                        lp_backend,
                                                        profile_data_transform,
                                                        profile_compute,
                                                        pdim_hint,
                                                        pratio_hint,
                                                        debug_level,
                                                        out_params);
    params = out_params;
  }

  std::vector<mace::CodlOpTaskChain> op_chains;

  if (search_method.compare("naive_serial") == 0) {
    mace::CodlOpTaskChain op_chain;
    for (auto &param : params) {
      op_chain.Append(param.get());
    }
    double lat;
    op_chain.SerialRun(10, rounds, false, &lat);
    LOG(INFO) << "Example: " << model_name << ", naive_serial"
              << ", lat " << lat << " ms";
  }

  std::vector<double> out_lats;
  
  if (search_method.compare("serial") == 0) {
    op_chains.clear();
    mace::ChainSearch::Serial(params, op_chains);
    mace::CodlOpTaskChainExecutor::Execute(op_chains, rounds,
                                           model_name + ", serial",
                                           debug_level, out_lats);
  }

  if (search_method.compare("heuristic") == 0) {
    op_chains.clear();
    mace::ChainSearch::Heuristic(params, op_chains);
    mace::CodlOpTaskChainExecutor::Execute(op_chains, rounds,
                                           model_name + ", heuristic",
                                           debug_level, out_lats);
  }

  if (search_method.compare("greedy") == 0) {
    op_chains.clear();
    mace::ChainSearch::Greedy(params, lat_acq, search_baseline,
                              pratio_hint, debug_level, op_chains);
    mace::CodlOpTaskChainExecutor::Execute(op_chains, rounds,
                                           model_name + ", greedy",
                                           debug_level, out_lats);
  }

  if (search_method.compare("full") == 0) {
    op_chains.clear();
    mace::ChainSearch::Full(params, op_chains);
    mace::CodlOpTaskChainExecutor::Execute(op_chains, FLAGS_rounds,
                                           model_name + ", full",
                                           debug_level, out_lats);
  }

  MACE_CHECK(out_lats.size() == 3);
  LOG(INFO) << "total " << out_lats[0] << " ms"
            << ", dms " << out_lats[1] << " ms"
            << ", comp " << out_lats[2] << " ms";

  const index_t total_raw_size
      = mace::CodlOpTaskChainHelper::CalcTotalMemoryUsage(op_chains);
  const double total_size_mb = total_raw_size / 1024.0 / 1024.0;
  LOG(INFO) << "total_raw_size " << total_raw_size << " bytes"
            << ", total_size " << total_size_mb << " MB";

#if 0
  double lat_chain;
  LPInitContext lp_init_context;
  lp_init_context.set_backend(lp_backend);
  mace::OpChainLatencyPredictor chain_predictor;
  chain_predictor.Init(&lp_init_context);
  chain_predictor.Predict(op_chains, debug_level, &lat_chain);
  LOG(INFO) << "Predicted chain latency: " << lat_chain << " ms";
#endif

#if 0
  utils::ThreadPool::Sleep(5000);
#endif

  //mace::CodlOpTaskChainHelper::HardFree(op_chains);

  return 0;
}

int example_nn_model_real_chain_search(const std::string &model_name) {
  const bool do_make_partition_plan = FLAGS_make_partition_plan;
  const bool profile_data_transform = FLAGS_profile_data_transform;
  const bool profile_compute = FLAGS_profile_compute;
  const std::string search_method = FLAGS_search_method;
  const int chain_param_hint = FLAGS_chain_param_hint;
  const int search_baseline = FLAGS_search_baseline;
  int chain_idx = FLAGS_chain_idx;
  size_t chain_count = static_cast<size_t>(FLAGS_chain_count);
  const int pdim_hint = FLAGS_pdim_hint;
  const int pratio_hint = FLAGS_pratio_hint;
  const LatencyAcquirement lat_acq = static_cast<LatencyAcquirement>(FLAGS_latency_acq);
  const LPBackend lp_backend = static_cast<LPBackend>(FLAGS_lp_backend);
  const int rounds = FLAGS_rounds;
  const int debug_level = FLAGS_debug_level;

  CodlOpChainCommonParam common_param;
  common_param.cpu_dtype = static_cast<DataType>(FLAGS_cpu_dtype);
  common_param.num_threads = FLAGS_num_threads;
  common_param.gpu_dtype = static_cast<DataType>(FLAGS_gpu_dtype);
  common_param.gpu_mtype = static_cast<MemoryType>(FLAGS_gpu_mtype);
  common_param.do_data_transform = FLAGS_data_transform;
  common_param.do_compute = FLAGS_compute;

  std::shared_ptr<NnModelBuilder> builder;
  if (!model_name.compare("yolov2")) {
    builder.reset(new Yolov2Builder(common_param));
  } else if (!model_name.compare("posenet")) {
    builder.reset(new PosenetBuilder(common_param));
  } else if (!model_name.compare("alexnet")) {
    builder.reset(new AlexnetBuilder(common_param));
  } else if (!model_name.compare("vgg16")) {
    builder.reset(new Vgg16Builder(common_param));
  } else if (!model_name.compare("fast_style_transfer")) {
    builder.reset(new FastStyleTransferBuilder(common_param));
  } else if (!model_name.compare("retinaface")) {
    builder.reset(new RetinaFaceBuilder(common_param));
  } else if (!model_name.compare("resnet50v1")) {
    builder.reset(new Resnet50v1Builder(common_param));
  } else if (!model_name.compare("bert")) {
    builder.reset(new BertBuilder(common_param, 24, 1024, 16));
  } else {
    LOG(INFO) << "Unsupported NN model " << model_name;
    return 0;
  }
#if 0
  else if (!model_name.compare("mobilenetv2")) {
    build_mobilenetv2(pdim, pratio, params);
  } else if (!model_name.compare("resnet50v2")) {
    if (chain_idx == 1) {
      build_resnet50v2_chain1(pdim, pratio, sratio, params);
    }
  } else if (!model_name.compare("matmul_net")) {
    build_matmul_net(pdim, pratio, params);
  } else {
    LOG(INFO) << "Unsupported NN model " << model_name;
    return 0;
  }
#endif

  index_t max_cpu_input_raw_size = -1;
  index_t max_cpu_output_raw_size = -1;
  index_t const_raw_size = 0;
  std::vector<double> nn_lats(3, 0);
  const size_t model_chain_count = builder->op_chain_count();
  if (chain_idx < 0) {
    chain_idx = 0;
  }
  chain_count += chain_idx;
  if (chain_count <= 0 || chain_count > model_chain_count) {
    chain_count = model_chain_count;
  }

  // Build chain.
  std::vector<std::vector<std::shared_ptr<CodlOpChainParam>>> chain_params;
  if (do_make_partition_plan) {
    MACE_NOT_IMPLEMENTED;
  } else {
    for (size_t i = chain_idx; i < chain_count; i ++) {
      std::vector<std::shared_ptr<CodlOpChainParam>> params;
      builder->Build(i, chain_param_hint, params);
      chain_params.push_back(params);
    }
  }
  
  for (int i = 0; i < rounds; i ++) {
    const int iner_rounds = 10 + 1;
    for (size_t j = 0; j < chain_params.size(); j ++) {
      //std::vector<std::shared_ptr<CodlOpChainParam>> params;
      std::vector<mace::CodlOpTaskChain> op_chains;
      std::vector<double> out_lats;

      //builder->Build(i, chain_param_hint, params);

      std::vector<std::shared_ptr<CodlOpChainParam>> &params = chain_params[j];
      if (params.size() == 0) {
        continue;
      }

      if (do_make_partition_plan) {
        std::vector<std::shared_ptr<CodlOpChainParam>> out_params;
        mace::ChainSearch::OptimalPartitionProfileOrPredict(params,
                                                            lat_acq,
                                                            lp_backend,
                                                            profile_data_transform,
                                                            profile_compute,
                                                            pdim_hint,
                                                            pratio_hint,
                                                            debug_level,
                                                            out_params);
        params = out_params;
      }
      
      std::string exec_tag;
      if (search_method.compare("serial") == 0) {
        mace::ChainSearch::Serial(params, op_chains);
        exec_tag = model_name + ", serial";
      }

      if (search_method.compare("heuristic") == 0) {
        mace::ChainSearch::Heuristic(params, op_chains);
        exec_tag = model_name + ", heuristic";
      }

      if (search_method.compare("greedy") == 0) {
        mace::ChainSearch::Greedy(params, lat_acq, search_baseline,
                                  pratio_hint, debug_level, op_chains);
        exec_tag = model_name + ", greedy";
      }

      if (search_method.compare("full") == 0) {
        mace::ChainSearch::Full(params, op_chains);
        exec_tag = model_name + ", full";
      }

      mace::CodlOpTaskChainExecutor::Execute(op_chains, iner_rounds,
                                             exec_tag, debug_level, out_lats);

      if (i == 0) {
        mace::CodlOpTaskChainHelper::CalcTotalMemoryUsage(
            op_chains,
            debug_level,
            &max_cpu_input_raw_size,
            &max_cpu_output_raw_size,
            &const_raw_size);
      }

      //params.clear();
      op_chains.clear();

      for (size_t j = 0; j < out_lats.size(); j ++) {
        nn_lats[j] += out_lats[j];
      }
    }
  }

  const index_t max_cpu_input_output_raw_size = max_cpu_input_raw_size + max_cpu_output_raw_size;
  const index_t total_raw_size = max_cpu_input_output_raw_size + const_raw_size;
  const double total_size_mb = total_raw_size / 1024.0 / 1024.0;

  LOG(INFO) << "total " << (nn_lats[0] / rounds) << " ms"
            << ", dms " << (nn_lats[1] / rounds) << " ms"
            << ", comp " << (nn_lats[2] / rounds) << " ms";
  LOG(INFO) << "total_raw_size " << total_raw_size << " bytes"
            << ", total_size " << total_size_mb << " MB"
            << ", max_cpu_input_output_raw_size " << max_cpu_input_output_raw_size << " bytes";

  

  return 0;
}

int example_yolov2() {
  if (FLAGS_test.compare("yolo_v2") == 0) {
    example_yolov2_cpu_only();
    example_yolov2_gpu_only();
  } else if (FLAGS_test.compare("yolo_v2_serial") == 0) {
    example_yolov2_serial();
  } else if (FLAGS_test.compare("yolo_v2_chain") == 0) {
    example_yolov2_chain();
  } else if (FLAGS_test.compare("yolo_v2_same_size") == 0) {
    //example_yolov2_same_size(0);
    //example_yolov2_same_size(1);
    example_yolov2_same_size_chain();
  } else if (FLAGS_test.compare("yolo_v2_chain_search") == 0) {
    //example_yolov2_chain_search();
    example_nn_model_chain_search("yolov2");
  } else if (FLAGS_test.compare("yolo_v2_real_chain_search") == 0) {
    //example_yolov2_real_chain_search();
    example_nn_model_real_chain_search("yolov2");
  }

  return 0;
}

int example_posenet() {
  if (FLAGS_test.compare("posenet_chain_search") == 0) {
    example_nn_model_chain_search("posenet");
  } else if (FLAGS_test.compare("posenet_real_chain_search") == 0) {
    example_nn_model_real_chain_search("posenet");
  }

  return 0;
}

int example_alexnet() {
  if (FLAGS_test.compare("alexnet_chain_search") == 0) {
    example_nn_model_chain_search("alexnet");
  } else if (FLAGS_test.compare("alexnet_real_chain_search") == 0) {
    example_nn_model_real_chain_search("alexnet");
  }

  return 0;
}

int example_vgg16() {
  if (FLAGS_test.compare("vgg16_chain_search") == 0) {
    example_nn_model_chain_search("vgg16");
  } else if (FLAGS_test.compare("vgg16_real_chain_search") == 0) {
    example_nn_model_real_chain_search("vgg16");
  }

  return 0;
}

int example_fast_style_transfer() {
  if (FLAGS_test.compare("fast_style_transfer_chain_search") == 0) {
    example_nn_model_chain_search("fast_style_transfer");
  } else if (FLAGS_test.compare("fast_style_transfer_real_chain_search") == 0) {
    example_nn_model_real_chain_search("fast_style_transfer");
  }

  return 0;
}

int example_retinaface() {
  if (FLAGS_test.compare("retinaface_chain_search") == 0) {
    example_nn_model_chain_search("retinaface");
  } else if (FLAGS_test.compare("retinaface_real_chain_search") == 0) {
    example_nn_model_real_chain_search("retinaface");
  }

  return 0;
}

int example_mobilenetv2_cpu_only() {
  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  build_mobilenetv2(1, 0.0, params);

  mace::CodlOpTaskChain op_chain;
  for (size_t i = 0; i < params.size(); i++) {
    op_chain.Append(params[i].get());
  }

  for (int i = 0; i < FLAGS_rounds; i ++) {
    double duration = 0.0;
    op_chain.SerialRun(&duration);
    LOG(INFO) << "Example: mobilenetv2 cpu " << duration << " ms";
  }

  return 0;
}

int example_mobilenetv2_gpu_only() {
  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  build_mobilenetv2(1, 1.0, params);

  mace::CodlOpTaskChain op_chain;
  for (size_t i = 0; i < params.size(); i++) {
    op_chain.Append(params[i].get());
  }

  for (int i = 0; i < FLAGS_rounds; i ++) {
    double duration = 0.0;
    op_chain.SerialRun(&duration);
    LOG(INFO) << "Example: mobilenetv2 gpu " << duration << " ms";
  }

  return 0;
}

int example_mobilenetv1() {
  if (FLAGS_test.compare("mobilenet_v1") == 0) {
    example_nn_model_chain_search("mobilenetv1");
  } else if (FLAGS_test.compare("mobilenet_v1_chain_search") == 0) {
    example_nn_model_chain_search("mobilenetv1");
  }

  return 0;
}

int example_mobilenetv2() {
  if (FLAGS_test.compare("mobilenet_v2") == 0) {
    example_mobilenetv2_cpu_only();
    example_mobilenetv2_gpu_only();
  } else if (FLAGS_test.compare("mobilenet_v2_chain_search") == 0) {
    example_nn_model_chain_search("mobilenetv2");
  }

  return 0;
}

int example_resnet50v1() {
  if (FLAGS_test.compare("resnet50_v1_chain_search") == 0) {
    example_nn_model_chain_search("resnet50v1");
  } else if (FLAGS_test.compare("resnet50_v1_real_chain_search") == 0) {
    example_nn_model_real_chain_search("resnet50v1");
  }

  return 0;
}

int example_resnet50v2() {
  if (FLAGS_test.compare("resnet50_v2_chain_search") == 0) {
    example_nn_model_chain_search("resnet50v2");
  }

  return 0;
}

int exmaple_matmul_net() {
  if (FLAGS_test.compare("matmul_net_chain_search") == 0) {
    example_nn_model_chain_search("matmul_net");
  }

  return 0;
}

int example_bert_cpu_only() {
  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  build_bert(1, 0.0, params);

  mace::CodlOpTaskChain chain;
  for (size_t i = 0; i < params.size(); i++) {
    chain.Append(params[i].get());
  }

  for (int i = 0; i < FLAGS_rounds; i ++) {
    double duration = 0.0;
    chain.SerialRun(&duration);
    LOG(INFO) << "Example: bert cpu " << duration << " ms";
  }

  return 0;
}

int example_bert_gpu_only() {
  std::vector<std::shared_ptr<CodlOpChainParam>> params;
  build_bert(1, 1.0, params);

  mace::CodlOpTaskChain chain;
  for (size_t i = 0; i < params.size(); i++) {
    chain.Append(params[i].get());
  }

  for (int i = 0; i < FLAGS_rounds; i ++) {
    double duration = 0.0;
    chain.SerialRun(&duration);
    LOG(INFO) << "Example: bert gpu " << duration << " ms";
  }

  return 0;
}

int example_bert() {
  if (FLAGS_test.compare("bert") == 0) {
    example_bert_cpu_only();
    example_bert_gpu_only();
  } else if (FLAGS_test.compare("bert_chain_search") == 0) {
    example_nn_model_chain_search("bert");
  }

  return 0;
}

int example_op_chain_net() {
  if (FLAGS_test.compare("op_chain_net_chain_search") == 0) {
    example_nn_model_chain_search("op_chain_net");
  }
  return 0;
}

int example_full_op_chain_search() {
  if (FLAGS_test.compare("full_op_chain_search") == 0) {
    const int op_count = FLAGS_op_count;
    std::vector<std::set<std::vector<int>>> chain_case_sets;
    mace::ChainSearch::BuildChainCases(op_count, chain_case_sets);
    mace::ChainSearch::PrintChainCases(chain_case_sets);
  }
  return 0;
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  print_flags();
  example_yolov2();
  example_posenet();
  example_alexnet();
  example_vgg16();
  example_fast_style_transfer();
  example_retinaface();
  example_mobilenetv1();
  example_mobilenetv2();
  example_resnet50v1();
  example_resnet50v2();
  exmaple_matmul_net();
  example_bert();
  example_op_chain_net();
  example_full_op_chain_search();
  
  return 0;
}
