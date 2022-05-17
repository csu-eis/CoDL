
#include "test/codl_run/op_test_task_chain.h"
#include "test/codl_run/conv2d_test_param.h"
#include "test/codl_run/conv2d_test_task.h"
#include "test/codl_run/pooling_test_param.h"
#include "test/codl_run/pooling_test_task.h"
#include "test/codl_run/fully_connected_test_param.h"
#include "test/codl_run/fully_connected_test_task.h"
#include "test/codl_run/matmul_test_param.h"
#include "test/codl_run/matmul_test_task.h"

namespace mace {

std::string CodlOpTypeToString(const CodlOpType type) {
  switch (type) {
    case CODL_OP_TYPE_NONE: return "None";
    case CODL_OP_TYPE_CONV2D: return "Conv2D";
    case CODL_OP_TYPE_POOLING: return "Pooling";
    case CODL_OP_TYPE_FULLY_CONNECTED: return "FullyConnected";
    case CODL_OP_TYPE_DECONV2D: return "Deconv2D";
    case CODL_OP_TYPE_MATMUL: return "MatMul";
    default: return "Unknown";
  }
}

std::string CodlOpChainCommonParamToString(const CodlOpChainCommonParam &param) {
  std::stringstream stream;
  stream << "do_data_transform " << param.do_data_transform
         << ", do_compute " << param.do_compute
         << ", num_threads " << param.num_threads
         << ", cpu_dtype " << param.cpu_dtype
         << ", gpu_dtype " << param.gpu_dtype
         << ", gpu_mtype " << param.gpu_mtype;
  return stream.str();
}

int CodlOpTaskChain::AddCommonTestParam(const int part_dim,
                                        const float part_ratio,
                                        const CodlOpChainCommonParam &common_param,
                                        TestParam *param) {
  //LOG(INFO) << CodlOpChainCommonParamToString(common_param);
  param->is_debug_on = false;
  param->do_data_transform = common_param.do_data_transform;
  param->do_compute = common_param.do_compute;
  param->do_warmup = false;
  param->cpu_affinity_policy = 1;
  param->num_threads = common_param.num_threads;
  param->cpu_dtype = common_param.cpu_dtype;
  param->gpu_memory_type = common_param.gpu_mtype;
  param->gpu_dtype = common_param.gpu_dtype;
  param->part_dim = part_dim;
  param->part_ratio = part_ratio;
  param->compute_unit_hint = ComputeUnitHint::COMPUTE_UNIT_HINT_DEFAULT;
  return 0;
}

int CodlOpTaskChain::AppendConv2d(const index_t height,
                                  const index_t width,
                                  const index_t in_channel,
                                  const index_t out_channel,
                                  const index_t filter_height,
                                  const index_t filter_width,
                                  const int stride_h,
                                  const int stride_w,
                                  const int part_dim,
                                  const float part_ratio,
                                  const CodlOpChainCommonParam &common_param) {
  Conv2dTestParam param;
  AddCommonTestParam(part_dim, part_ratio, common_param, &param);

  param.wino_block_size = 0;
  param.input_shape = {1, height, width, in_channel};
  param.filter_shape = {out_channel, in_channel, filter_height, filter_width};
  param.strides = {stride_h, stride_w};

  std::shared_ptr<CodlConv2dCpuGpuTestTask> task(new CodlConv2dCpuGpuTestTask());
  task->Prepare(&param);

  tasks_.emplace_back(task);

  return 0;
}

int CodlOpTaskChain::AppendPooling(const index_t height,
                                   const index_t width,
                                   const index_t in_channel,
                                   const index_t filter_height,
                                   const index_t filter_width,
                                   const int stride_h,
                                   const int stride_w,
                                   const int pooling_type,
                                   const int part_dim,
                                   const float part_ratio,
                                   const CodlOpChainCommonParam &common_param) {
  PoolingTestParam param;
  AddCommonTestParam(part_dim, part_ratio, common_param, &param);

  param.input_shape = {1, height, width, in_channel};
  param.filter_shape = {in_channel, in_channel, filter_height, filter_width};
  param.strides = {stride_h, stride_w};
  param.pooling_type = pooling_type;

  std::shared_ptr<CodlPoolingCpuGpuTestTask> task(new CodlPoolingCpuGpuTestTask());
  task->Prepare(&param);

  tasks_.emplace_back(task);

  return 0;
}

int CodlOpTaskChain::AppendFullyConnected(const index_t height,
                                          const index_t width,
                                          const index_t in_channel,
                                          const index_t out_channel,
                                          const int part_dim,
                                          const float part_ratio,
                                          const CodlOpChainCommonParam &common_param) {
  FullyConnectedTestParam param;
  AddCommonTestParam(part_dim, part_ratio, common_param, &param);

  param.input_shape = {1, height, width, in_channel};
  param.weight_shape = {out_channel, in_channel, height, width};

  std::shared_ptr<CodlFullyConnectedCpuGpuTestTask> task(
      new CodlFullyConnectedCpuGpuTestTask());
  task->Prepare(&param);

  tasks_.emplace_back(task);

  return 0;
}

int CodlOpTaskChain::AppendMatMul(const index_t batch,
                                  const index_t height,
                                  const index_t width,
                                  const index_t depth,
                                  const bool transpose_a,
                                  const bool transpose_b,
                                  const int part_dim,
                                  const float part_ratio,
                                  const CodlOpChainCommonParam &common_param) {
  MatMulTestParam param;
  AddCommonTestParam(part_dim, part_ratio, common_param, &param);

  if (batch == 1) {
    if (transpose_a) {
      param.input_shape = {depth, height};
    } else {
      param.input_shape = {height, depth};
    }
    if (transpose_b) {
      param.rhs_shape = {width, depth};
    } else {
      param.rhs_shape = {depth, width};
    }
  } else if (batch > 1) {
    if (transpose_a) {
      param.input_shape = {1, batch, depth, height};
    } else {
      param.input_shape = {1, batch, height, depth};
    }
    if (transpose_b) {
      param.rhs_shape = {1, batch, width, depth};
    } else {
      param.rhs_shape = {1, batch, depth, width};
    }
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  
  param.transpose_a = transpose_a;
  param.transpose_b = transpose_b;

  std::shared_ptr<CodlMatMulCpuGpuTestTask> task(new CodlMatMulCpuGpuTestTask());
  task->Prepare(&param);

  tasks_.emplace_back(task);

  return 0;
}

int CodlOpTaskChain::UpdatePartitionShape() {
  const size_t num_ops = tasks_.size();

#if 0
  LOG(INFO) << "===== Chain Partition Info =====";
  LOG(INFO) << "op_idx " << (num_ops - 1)
            << ", type " << CodlTestTaskTypeToString(tasks_[num_ops - 1]->type())
            << ", cpu_input_part_shape "
            << VectorToString<index_t>(tasks_[num_ops - 1]->part_plan()->cpu_input_part_shape())
            << ", cpu_output_part_shape "
            << VectorToString<index_t>(tasks_[num_ops - 1]->part_plan()->cpu_output_part_shape())
            << ", gpu_input_part_shape "
            << VectorToString<index_t>(tasks_[num_ops - 1]->part_plan()->gpu_input_part_shape())
            << ", gpu_output_part_shape "
            << VectorToString<index_t>(tasks_[num_ops - 1]->part_plan()->gpu_output_part_shape());
#endif

  for (size_t i = (num_ops - 1); i > 0; i --) {
    CodlOpCpuGpuTestTask *cur_op = tasks_[i].get();
    CodlOpCpuGpuTestTask *prev_op = tasks_[i - 1].get();
    // Update partial output and input shape of previous OP.
    prev_op->part_plan()->set_cpu_output_part_shape(
        cur_op->part_plan()->cpu_input_part_shape());
    prev_op->part_plan()->set_gpu_output_part_shape(
        cur_op->part_plan()->gpu_input_part_shape());
    prev_op->part_plan()->UpdateInputPartShape();
#if 0
    LOG(INFO) << "op_idx " << (i - 1)
              << ", type " << CodlTestTaskTypeToString(prev_op->type())
              << ", cpu_input_part_shape "
              << VectorToString<index_t>(prev_op->part_plan()->cpu_input_part_shape())
              << ", cpu_output_part_shape "
              << VectorToString<index_t>(prev_op->part_plan()->cpu_output_part_shape())
              << ", gpu_input_part_shape "
              << VectorToString<index_t>(prev_op->part_plan()->gpu_input_part_shape())
              << ", gpu_output_part_shape "
              << VectorToString<index_t>(prev_op->part_plan()->gpu_output_part_shape());
#endif
    // Update partial input and output tensor of previous OP.
    prev_op->UpdatePartTensors();
  }

  return 0;
}

int CodlOpTaskChain::Prepare() {
  UpdatePartitionShape();
  is_ready_ = true;
  return 0;
}

int CodlOpTaskChain::SerialRun(
    DurationCollector<double> *collector) const {
  int64_t t0;
  
  OpenCLRuntime *opencl_runtime =
      GetDeviceContext()->GetGpuDevice()->gpu_runtime()->opencl_runtime();
  OpenCLEventManager *event_manager = opencl_runtime->event_manager();
  
  for (size_t i = 0; i < tasks_.size(); i ++) {
    // [0]: Enqueue
    // [1]: DT+Map+Sync
    // [2]: CPU Comp
    // [3]: GPU Comp
    // [4]: Finish
    std::vector<double> delays;
    //LOG(INFO) << "Run layer idx " << i;
    StatsFuture map_out_future;
    StatsFuture gpu_compute_future;
    StatsFuture tr_out_future;
    cl::UserEvent *trans_out_user_event = nullptr;

    t0 = NowMicros();
    
    //ShowText("Enqueue input data transform kernel");
    tasks_[i]->EnqueueInputDataTransformKernel();
    //ShowText("Enqueue map kernel");
    tasks_[i]->EnqueueMapKernel(&map_out_future);
    //ShowText("Enqueue gpu compute kernel");
    tasks_[i]->EnqueueGpuComputeKerenl(&gpu_compute_future);
    //ShowText("Enqueue unmap kernel");
    tasks_[i]->EnqueueUnmapKernel(&trans_out_user_event);
    //ShowText("Enqueue output data transform kernel");
    tasks_[i]->EnqueueOutputDataTransformKernel(&tr_out_future);

    delays.push_back((NowMicros() - t0) / 1000.0);

    t0 = NowMicros();
    map_out_future.wait_fn(nullptr);
    delays.push_back((NowMicros() - t0) / 1000.0);
    
    t0 = NowMicros();
    tasks_[i]->RunCpuComputeKernel();
    delays.push_back((NowMicros() - t0) / 1000.0);

    if (trans_out_user_event != nullptr) {
      // Synchronize for input unmapping.
      event_manager->SetUserEventComplete(trans_out_user_event);
    }

    delays.push_back(FutureToMillis(&gpu_compute_future));

    t0 = NowMicros();
    tr_out_future.wait_fn(nullptr);
    delays.push_back((NowMicros() - t0) / 1000.0);
    
    opencl_runtime->command_queue().finish();

    if (collector != nullptr) {
#if 1
      LOG(INFO) << "delays " << VectorToString<double>(delays) << " ms";
#endif
      collector->Add(delays);
    }
  }

#if 0
  std::vector<double> delays;
  delays.push_back(0);
  delays.push_back(0);
  delays.push_back(0);
  delays.push_back(0);
  t0 = NowMicros();
  opencl_runtime->command_queue().finish();
  delays.push_back((NowMicros() - t0) / 1000.0);


  if (collector != nullptr) {
    collector->Add(delays);
  }
#endif
  
  event_manager->Clear();

  return 0;
}

int CodlOpTaskChain::SerialRun(const int si,
                               const int rounds,
                               const bool is_compute_only,
                               double *lat) const {
  const int total_rounds = si + rounds;
  DurationCollector<double> collector;
  for (int i = 0; i < total_rounds; i ++) {
#if 1
    LOG(INFO) << "round " << i;
#endif
    SerialRun(&collector);
  }
  if (lat != nullptr) {
    const size_t num_ops = tasks_.size();
    const std::vector<double> stat = collector.StatSum(si * num_ops);
    const double lat_dms = (stat[1] + stat[4]) / rounds;
    const double lat_compute = (fmax(stat[2], stat[3])) / rounds;
#if 0
    LOG(INFO) << "lat_dms " << lat_dms << " ms"
              << ", lat_compute " << lat_compute << " ms";
#endif
    if (is_compute_only) {
      *lat = lat_compute;
    } else {
      *lat = lat_dms + lat_compute;
    }
  }

  return 0;
}

int CodlOpTaskChain::SerialRun(double *lat) const {
  DurationCollector<double> collector;
  SerialRun(&collector);
  if (lat != nullptr) {
    const std::vector<double> stat = collector.StatSum();
    *lat = stat[1] + fmax(stat[2], stat[3]) + stat[4];
  }

  return 0;
}

int CodlOpTaskChain::Run(DurationCollector<double> *collector) const {
  MACE_CHECK(is_ready_, "OP chain is not ready");
  int64_t t0;
  StatsFuture tr_in_future, map_in_future, map_out_future, unmap_in_future, unmap_out_future, tr_out_future;
  std::vector<StatsFuture> gpu_compute_futures(tasks_.size());
  // [0]: Enqueue
  // [1]: DT+Map+Sync
  // [2]: CPU Comp
  // [3]: GPU Comp
  // [4]: Finish
  std::vector<double> delays;

  t0 = NowMicros();

  //ShowText("Enqueue input data transform kernel");
  tasks_[0]->EnqueueInputDataTransformKernel(&tr_in_future);

  //ShowText("Enqueue map kernel");
  for (size_t i = 0; i < tasks_.size(); i ++) {
    tasks_[i]->EnqueueMapKernel(&map_out_future, &map_in_future);
  }

  //ShowText("Enqueue gpu compute kernel");
  for (size_t i = 0; i < tasks_.size(); i ++) {
    //LOG(INFO) << "Enqueue GPU computing kernel " << i;
    for (size_t j = 0; j < 1; j ++) {
      tasks_[i]->EnqueueGpuComputeKerenl(&gpu_compute_futures[i]);
      //int64_t tmp_t0 = NowMicros();
      //FutureToMillis(&gpu_compute_futures[i]);
      //LOG(INFO) << "FutureToMillis: " << (NowMicros() - tmp_t0) / 1000.0;
    }
  }
  
  OpenCLRuntime *opencl_runtime =
      GetDeviceContext()->GetGpuDevice()->gpu_runtime()->opencl_runtime();
  OpenCLEventManager *event_manager = opencl_runtime->event_manager();
  cl::UserEvent *trans_out_user_event = nullptr;
  
  const size_t ei = tasks_.size() - 1;
  tasks_[ei]->EnqueueUnmapKernel(&trans_out_user_event,
                                 &unmap_in_future,
                                 &unmap_out_future);
  tasks_[ei]->EnqueueOutputDataTransformKernel(&tr_out_future);

  delays.push_back((NowMicros() - t0) / 1000.0);

  t0 = NowMicros();
  opencl_runtime->command_queue().flush();
  map_out_future.wait_fn(nullptr);
  const double cpu_pre_delay = (NowMicros() - t0) / 1000.0;
  delays.push_back(cpu_pre_delay);

  t0 = NowMicros();
  for (size_t i = 0; i < tasks_.size(); i ++) {
    //LOG(INFO) << "Run cpu compute kernel idx " << i;
    tasks_[i]->RunCpuComputeKernel();
  }
  delays.push_back((NowMicros() - t0) / 1000.0);

  if (trans_out_user_event != nullptr) {
    // Synchronize for input unmapping.
    event_manager->SetUserEventComplete(trans_out_user_event);
  }

#if 0
  if (gpu_compute_futures.size() > 0) {
    gpu_compute_futures[gpu_compute_futures.size() - 1].wait_fn(nullptr);
  }
#endif
  for (size_t i = 0; i < gpu_compute_futures.size(); i ++) {
    gpu_compute_futures[i].wait_fn(nullptr);
  }

  t0 = NowMicros();
  tr_out_future.wait_fn(nullptr);
  //const double post_delay = (NowMicros() - t0) / 1000.0;
  //opencl_runtime->command_queue().finish();
#if 0
  const double tr_in_delay = FutureToMillis(&tr_in_future);
  const double map_in_delay = FutureToMillis(&map_in_future);
  const double map_out_delay = FutureToMillis(&map_out_future);
  const double gpu_pre_delay = tr_in_delay + map_in_delay + map_out_delay;
  LOG(INFO) << "cpu_pre_delay " << cpu_pre_delay << " ms"
            << ", gpu_pre_delay " << gpu_pre_delay << " ms";
#endif
#if 1
  const double unmap_in_delay = FutureToMillis(&unmap_in_future);
  const double unmap_out_delay = FutureToMillis(&unmap_out_future);
  const double tr_out_delay = FutureToMillis(&tr_out_future);
  const double post_delay = unmap_in_delay + unmap_out_delay + tr_out_delay;
#endif

  double gpu_compute_delay_sum = 0;
  for (size_t i = 0; i < gpu_compute_futures.size(); i ++) {
    gpu_compute_delay_sum += FutureToMillis(&gpu_compute_futures[i]);
  }
  delays.push_back(gpu_compute_delay_sum);

  delays.push_back(post_delay);

  //opencl_runtime->command_queue().finish();
  //event_manager->Clear();

  if (collector != nullptr) {
#if 0
    LOG(INFO) << "delays " << VectorToString<double>(delays) << " ms";
#endif
    collector->Add(delays);
  }

  return 0;
}

int CodlOpTaskChain::Run(const int si,
                         const int rounds,
                         const bool is_compute_only,
                         double *lat) const {
  const int total_rounds = si + rounds;
  DurationCollector<double> collector;
  for (int i = 0; i < total_rounds; i ++) {
    Run(&collector);
  }
  if (lat != nullptr) {
    const std::vector<double> stat = collector.StatSum(si);
    const double lat_dms = (stat[1] + stat[4]) / rounds;
    const double lat_comp = fmax(stat[2], stat[3]) / rounds;
    if (is_compute_only) {
      *lat = lat_comp;
    } else {
      *lat = lat_dms + lat_comp;
    }
    LOG(INFO) << "op chain run, lat " << *lat << " ms"
              << ", lat_dms " << lat_dms << " ms"
              << ", lat_comp " << lat_comp << " ms";
  }

  return 0;
}

int CodlOpTaskChain::Run(double *lat) const {
  DurationCollector<double> collector;
  Run(&collector);
  if (lat != nullptr) {
    const std::vector<double> stat = collector.StatSum();
    *lat = stat[1] + fmax(stat[2], stat[3]) + stat[4];
  }

  return 0;
}

}  // namespace mace
