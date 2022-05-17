
#include <memory>
#include <vector>

#include "mace/public/mace.h"
#include "mace/utils/thread_pool.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/gpu_device.h"
#endif
#include "test/codl_run/utils/device_util.h"

#ifdef MACE_ENABLE_OPENCL
#define DEFAULT_STORAGE_PATH "/data/local/tmp/mace_run/interior"
#endif

namespace mace {

TestDeviceContext::~TestDeviceContext() {}

int TestDeviceContext::InitThreadPool() {
  MACE_CHECK(thread_pool_ == nullptr);
  LOG(INFO) << "Init thread pool: num_threads " << thread_count_
            << ", affinity_policy " << affinity_policy_;
  thread_pool_.reset(new utils::ThreadPool(thread_count_, affinity_policy_));
  thread_pool_->Init();
  thread_pool_->SetSpinWaitTime(kMaxSpinWaitTime);
  return 0;
}

int TestDeviceContext::InitCpuDevice() {
  if (thread_pool_ == nullptr) {
    InitThreadPool();
  }
  
  cpu_device_.reset(new CPUDevice(thread_count_,
                                  affinity_policy_,
                                  thread_pool_.get()));
  
  return 0;
}

Device* TestDeviceContext::GetCpuDevice() {
  return cpu_device_.get();
}

#ifdef MACE_ENABLE_OPENCL

int TestDeviceContext::InitGpuDevice() {
  const char *storage_path_ptr = getenv("MACE_INTERNAL_STORAGE_PATH");
  const std::string storage_path = 
      std::string(storage_path_ptr == nullptr ?
                  DEFAULT_STORAGE_PATH : storage_path_ptr);
  std::vector<std::string> opencl_binary_paths = {""};

  const char *opencl_parameter_file_ptr = getenv("MACE_OPENCL_PARAMETER_FILE");
  const std::string opencl_parameter_file = 
      std::string(opencl_parameter_file_ptr == nullptr ?
                  "" : opencl_parameter_file_ptr);
  
  gpu_context_ = GPUContextBuilder()
                  .SetStoragePath(storage_path)
                  .SetOpenCLBinaryPaths(opencl_binary_paths)
                  .SetOpenCLParameterPath(opencl_parameter_file)
                  .Finalize();
                  
  const GPUPriorityHint gpu_priority_hint = GPUPriorityHint::PRIORITY_HIGH;
  const GPUPerfHint     gpu_perf_hint     = GPUPerfHint::PERF_HIGH;
  utils::ThreadPool *thread_pool;
  if (thread_pool_) {
    // Reuse.
    thread_pool = thread_pool_.get();
  } else {
    InitThreadPool();
    thread_pool = thread_pool_.get();
  }
  
  gpu_device_.reset(new GPUDevice(gpu_context_->opencl_tuner(),
                                  gpu_context_->opencl_cache_storage(),
                                  gpu_priority_hint,
                                  gpu_perf_hint,
                                  gpu_context_->opencl_binary_storage(),
                                  thread_count_,
                                  affinity_policy_,
                                  thread_pool));
  
  return 0;
}

Device* TestDeviceContext::GetGpuDevice() {
  return gpu_device_.get();
}

#endif  // MACE_ENABLE_OPENCL

Workspace* GetWorkspace() {
  static Workspace ws;
  return &ws;
}

}  // namespace mace
