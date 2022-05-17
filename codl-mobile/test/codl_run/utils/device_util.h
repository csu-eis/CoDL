
#ifndef TEST_CODL_RUN_UTILS_DEVICE_UTIL_H_
#define TEST_CODL_RUN_UTILS_DEVICE_UTIL_H_

#include "mace/core/device.h"
#include "mace/core/workspace.h"

namespace mace {

class TestDeviceContext {
public:
  TestDeviceContext(const int thread_count,
                    const CPUAffinityPolicy affinity_policy)
        : is_initialized_(false),
          thread_count_(thread_count),
          affinity_policy_(affinity_policy),
          thread_pool_(nullptr) {
#ifdef MACE_ENABLE_OPENCL
    gpu_context_ = nullptr;
#endif
  }

  ~TestDeviceContext();

  inline bool is_initialized() {
    return is_initialized_;
  }

  inline void set_is_initialized(bool initialzed) {
    is_initialized_ = initialzed;
  }
  
  int InitCpuDevice();
  Device* GetCpuDevice();
#ifdef MACE_ENABLE_OPENCL
  int InitGpuDevice();
  Device* GetGpuDevice();
#endif // MACE_ENABLE_OPENCL

private:
  int InitThreadPool();
  
  bool is_initialized_;
  int thread_count_;
  CPUAffinityPolicy affinity_policy_;
  std::unique_ptr<utils::ThreadPool> thread_pool_;
  std::unique_ptr<Device> cpu_device_;
#ifdef MACE_ENABLE_OPENCL
  std::unique_ptr<utils::ThreadPool> gpu_thread_pool_;
  std::shared_ptr<GPUContext> gpu_context_;
  std::unique_ptr<Device> gpu_device_;
#endif // MACE_ENABLE_OPENCL
};

Workspace* GetWorkspace();

}  // namespace mace

#endif  // TEST_CODL_RUN_UTILS_DEVICE_UTIL_H_
