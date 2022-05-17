
#include "test/codl_run/core/test_param.h"
#include "test/codl_run/core/test_task.h"

namespace mace {

std::string CodlTestTaskTypeToString(const CodlTestTaskType type) {
  switch(type) {
    case NONE_CPU_GPU_TEST_TASK: return "NONE";
    case CONV2D_CPU_GPU_TEST_TASK: return "CONV2D";
    case POOLING_CPU_GPU_TEST_TASK: return "POOLING";
    case FC_CPU_GPU_TEST_TASK: return "FC";
    case DECONV2D_CPU_GPU_TEST_TASK: return "DECONV2D";
    case MATMUL_CPU_GPU_TEST_TASK: return "MATMUL";
    default: return "UNKNOWN";
  }
}

static std::unique_ptr<TestDeviceContext> global_context;

double FutureToMillis(const StatsFuture *future) {
  CallStats call_stats;
  future->wait_fn(&call_stats);
  double duration = (call_stats.end_micros - call_stats.start_micros) / 1000.0;
  if (duration < 0 || duration > 1000 * 1000) {
    duration = 0;
  }
  return duration;
}

void SetDeviceContext(TestDeviceContext *ctx) {
  global_context.reset(ctx);
}

void ClearDeviceContext() {
  global_context.reset(nullptr);
}

TestDeviceContext *GetDeviceContext() {
  return global_context.get();
}

void ShowText(const std::string &text) {
  VLOG(1) << text;
}

int CodlOpCpuGpuTestTask::RunCpu(
    mace::DurationCollector<double> *dura_collector) {
  if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
    int64_t t0;
    double wait_map_out_time = 0;
    double cpu_compute_time = 0;
    double wait_unmap_in_time = 0;
    double wait_transform_out_time = 0;
    StatsFuture future;
    StatsFuture transform_in_future;
    StatsFuture map_in_future;
    StatsFuture map_out_future;
    StatsFuture unmap_in_future;
    StatsFuture unmap_out_future;
    StatsFuture transform_out_future;
    cl::UserEvent *unmap_in_user_event = nullptr;

    if (do_data_transform_) {
      EnqueueInputDataTransformKernel(&transform_in_future);
    }

    EnqueueMapKernel(&map_out_future, &map_in_future);

    t0 = NowMicros();
    map_out_future.wait_fn(nullptr);
    wait_map_out_time = (NowMicros() - t0) / 1000.0;

    if (!do_data_transform_) {
      cpu_context_->set_dura_collector(dura_collector);
    } else {
      cpu_context_->set_dura_collector(nullptr);
    }

    t0 = NowMicros();
    if (do_compute_) {
      RunCpuComputeKernel();
    }
    cpu_compute_time = (NowMicros() - t0) / 1000.0;
    cpu_context_->set_dura_collector(nullptr);

    EnqueueUnmapKernel(&unmap_in_user_event,
                       &unmap_in_future,
                       &unmap_out_future);

    if (do_data_transform_) {
      EnqueueOutputDataTransformKernel(&transform_out_future);
    }

    if (unmap_in_user_event != nullptr) {
      OpenCLRuntime *opencl_runtime =
          GetDeviceContext()->GetGpuDevice()->gpu_runtime()->opencl_runtime();
      OpenCLEventManager *event_manager = opencl_runtime->event_manager();
      event_manager->SetUserEventComplete(unmap_in_user_event);
    }

    if (do_data_transform_) {
      t0 = NowMicros();
      transform_out_future.wait_fn(nullptr);
      wait_transform_out_time = (NowMicros() - t0) / 1000.0;
    }

    if (is_debug_on_) {
      const double tranform_in_time = FutureToMillis(&transform_in_future);
      const double map_in_time = FutureToMillis(&map_in_future);
      const double map_out_time = FutureToMillis(&map_out_future);
      const double unmap_in_time = FutureToMillis(&unmap_in_future);
      const double unmap_out_time = FutureToMillis(&unmap_out_future);
      const double transform_out_time = FutureToMillis(&transform_out_future);

      LOG(INFO) << "Latency:"
                << " tr_in " << tranform_in_time << " ms"
                << " map_in " << map_in_time << " ms"
                << " map_out " << map_out_time << " ms"
                << " wait_map_out " << wait_map_out_time << " ms"
                << " c_cpu " << cpu_compute_time << " ms"
                << " wait_unmap_in " << wait_unmap_in_time << " ms"
                << " unmap_in " << unmap_in_time << " ms"
                << " unmap_out " << unmap_out_time << " ms"
                << " tr_out " << transform_out_time << " ms"
                << " wait_tr_out " << wait_transform_out_time << " ms";
    }

    if (do_data_transform_ && dura_collector != nullptr) {
      // Data transfoming related latency.
      std::vector<double> dt_latencies;
      dt_latencies.push_back(FutureToMillis(&transform_in_future));  // 0
      dt_latencies.push_back(FutureToMillis(&map_in_future));        // 1
      dt_latencies.push_back(FutureToMillis(&map_out_future));       // 2
      dt_latencies.push_back(wait_map_out_time
          - dt_latencies[0] - dt_latencies[1] - dt_latencies[2]);    // 3
      dt_latencies.push_back(FutureToMillis(&unmap_in_future));      // 4
      dt_latencies.push_back(FutureToMillis(&unmap_out_future));     // 5
      dt_latencies.push_back(FutureToMillis(&transform_out_future)); // 6
      dt_latencies.push_back(wait_unmap_in_time - dt_latencies[4]);  // 7
      dura_collector->Add(dt_latencies);
    }
  }
  
  return 1;
}

int CodlOpCpuGpuTestTask::RunGpu(
    mace::DurationCollector<double> *dura_collector) {
  if (do_compute_ && IsComputeUnitOn(COMPUTE_UNIT_TYPE_GPU, compute_unit_hint_)) {
    StatsFuture future;
    const int64_t t0 = NowMicros();
    gpu_context_->set_future(&future);
    EnqueueGpuComputeKerenl(&future);
    future.wait_fn(nullptr);
    if (is_debug_on_) {
      const double gpu_compute_time = (NowMicros() - t0) / 1000.0;
      const double gpu_compute_time_cl = FutureToMillis(&future);
      LOG(INFO) << "Latency:"
                << " c_gpu " << gpu_compute_time << " ms"
                << " c_gpu_cl " << gpu_compute_time_cl << " ms";
    }
    if (dura_collector != nullptr) {
      dura_collector->Add(FutureToMillis(&future));
    }
  }
  return 1;
}

int CodlOpCpuGpuTestTask::RunCpuGpu(
    mace::DurationCollector<double> *dura_collector) {
  MACE_UNUSED(dura_collector);

  int64_t t0;
  double enqueued_gpu_time;
  double wait_map_out_time;
  double cpu_compute_time;
  double wait_transform_out_time;
  StatsFuture transform_in_future;
  StatsFuture map_in_future;
  StatsFuture map_out_future;
  StatsFuture gpu_compute_future;
  StatsFuture unmap_in_future;
  StatsFuture unmap_out_future;
  StatsFuture transform_out_future;
  cl::UserEvent *trans_out_user_event = nullptr;

  t0 = NowMicros();
  if (do_data_transform_) {
    if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
      EnqueueInputDataTransformKernel(&transform_in_future);
    }
  }

  if (do_compute_ && IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
    EnqueueMapKernel(&map_out_future, &map_in_future);
  }

  if (do_compute_ && IsComputeUnitOn(COMPUTE_UNIT_TYPE_GPU, compute_unit_hint_)) {
    EnqueueGpuComputeKerenl(&gpu_compute_future);
  }

  if (do_compute_ && IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
    EnqueueUnmapKernel(&trans_out_user_event,
                       &unmap_in_future,
                       &unmap_out_future);
  }

  if (do_data_transform_) {
    if (IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
      EnqueueOutputDataTransformKernel(&transform_out_future);
    }
  }
  enqueued_gpu_time = (NowMicros() - t0) / 1000.0;

  if (do_compute_ && IsComputeUnitOn(COMPUTE_UNIT_TYPE_CPU, compute_unit_hint_)) {
    t0 = NowMicros();
    map_out_future.wait_fn(nullptr);
    wait_map_out_time = (NowMicros() - t0) / 1000.0;

    t0 = NowMicros();
    RunCpuComputeKernel();
    cpu_compute_time = (NowMicros() - t0) / 1000.0;

    if (trans_out_user_event != nullptr) {
      OpenCLRuntime *opencl_runtime =
          GetDeviceContext()->GetGpuDevice()->gpu_runtime()->opencl_runtime();
      OpenCLEventManager *event_manager = opencl_runtime->event_manager();
      event_manager->SetUserEventComplete(trans_out_user_event);
    }

    t0 = NowMicros();
    transform_out_future.wait_fn(nullptr);
    wait_transform_out_time = (NowMicros() - t0) / 1000.0;
  }

  if (do_compute_ && IsComputeUnitOn(COMPUTE_UNIT_TYPE_GPU, compute_unit_hint_)) {
    gpu_compute_future.wait_fn(nullptr);
  }

  if (is_debug_on_) {
    const double tranform_in_time = FutureToMillis(&transform_in_future);
    const double map_in_time = FutureToMillis(&map_in_future);
    const double map_out_time = FutureToMillis(&map_out_future);
    const double gpu_compute_time = FutureToMillis(&gpu_compute_future);
    const double unmap_in_time = FutureToMillis(&unmap_in_future);
    const double unmap_out_time = FutureToMillis(&unmap_out_future);
    const double transform_out_time = FutureToMillis(&transform_out_future);
    const double sync_time = wait_map_out_time
        - (tranform_in_time + map_in_time + map_out_time);
    const double concurrency_time = wait_map_out_time
        + unmap_in_time + unmap_out_time + transform_out_time;
    const double compute_time = fmax(cpu_compute_time, gpu_compute_time);

    LOG(INFO) << "Latency:"
              << " tr_in " << tranform_in_time << " ms"
              << " map_in " << map_in_time << " ms"
              << " map_out " << map_out_time << " ms"
              << " tr_out " << transform_out_time << " ms"
              << " c_cpu " << cpu_compute_time << " ms"
              << " c_gpu " << gpu_compute_time << " ms";
    LOG(INFO) << "CPU-side Latency:"
              << " enq " << enqueued_gpu_time << " ms"
              << " wait_map_out " << wait_map_out_time << " ms"
              << " c_cpu " << cpu_compute_time << " ms"
              << " wait_tr_out " << wait_transform_out_time << " ms";
    LOG(INFO) << "GPU-side Latency:"
              << " tr_in " << tranform_in_time << " ms"
              << " map_in " << map_in_time << " ms"
              << " map_out " << map_out_time << " ms"
              << " c_gpu " << gpu_compute_time << " ms"
              << " unmap_in " << unmap_in_time << " ms"
              << " unmap_out " << unmap_out_time << " ms"
              << " tr_out " << transform_out_time << " ms";
    LOG(INFO) << "Concurrency Latency:"
              << " sync " << sync_time << " ms"
              << ", concurrency " << concurrency_time << " ms"
              << ", compute " << compute_time << " ms";
  }

  return 1;
}

int CodlOpCpuGpuTestTask::Run(
    mace::DurationCollector<double> *dura_collector) {
  if (part_plan_->IsCpuGpu()) {
    RunCpuGpu(dura_collector);
  } else if (part_plan_->IsGpuOnly()) {
    RunGpu(dura_collector);
  } else if (part_plan_->IsCpuOnly()) {
    RunCpu(dura_collector);
  } else {
    LOG(ERROR) << "Partition plan error.";
    return -1;
  }

  return 1;
}

int CodlOpCpuGpuTestTask::Destroy(const CodlTestTaskDestroyType type) {
  TestDeviceContext *ctx = GetDeviceContext();
  if (ctx) {
    OpenCLRuntime *opencl_runtime =
        ctx->GetGpuDevice()->gpu_runtime()->opencl_runtime();
    OpenCLEventManager *event_manager = opencl_runtime->event_manager();
    opencl_runtime->command_queue().finish();
    event_manager->Clear();
    
    PostProcess();

    if (tensor_manage_util_ != nullptr) {
      tensor_manage_util_->DeleteTensors();
    }

    //TensorUtils::DeleteTensorMemory(ctx);
  }

  if (type == DESTROY_TYPE_HARD) {
    ClearDeviceContext();
  }

  return 1;
}

}  // namespace mace
