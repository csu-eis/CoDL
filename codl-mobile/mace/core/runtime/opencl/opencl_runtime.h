// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
#define MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_

#include <map>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <set>
#include <string>
#include <vector>

#include "mace/core/kv_storage.h"
#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/scratch_image.h"
#include "mace/proto/mace.pb.h"
#include "mace/utils/string_util.h"
#include "mace/utils/timer.h"
#include "mace/utils/tuner.h"

namespace mace {

enum GPUType {
  QUALCOMM_ADRENO,
  MALI,
  PowerVR,
  UNKNOWN,
};

enum OpenCLVersion {
  CL_VER_1_0,
  CL_VER_1_1,
  CL_VER_1_2,
  CL_VER_2_0,
  CL_VER_UNKNOWN,
};

enum GPUPrecision {
  GPU_PRECISION_INT8 = 0,
  GPU_PRECISION_FP16 = 1,
  GPU_PRECISION_FP32 = 2
};

enum class EventActionType {
  SET = 0,
  WAIT = 1,
};

enum class EventOpType {
  NONE = 0,
  TRANSFORM_INPUT = 1,
  CONV2D = 2,
  TRANSFORM_OUTPUT = 3,
};

#ifdef MACE_ENABLE_RPCMEM
enum IONType {
  QUALCOMM_ION,
  NONE_ION,
};
#else
enum IONType {
  NONE_ION,
};
#endif  // MACE_ENABLE_RPCMEM

const std::string OpenCLErrorToString(cl_int error);

const std::string GetGPUPrecisionName(GPUPrecision gpu_precision);

#define MACE_CL_RET_ERROR(error)                            \
  if (error != CL_SUCCESS) {                                \
    LOG(ERROR) << "error: " << OpenCLErrorToString(error);  \
    return error;                                           \
  }

#define MACE_CL_RET_STATUS(error)                           \
  if (error != CL_SUCCESS) {                                \
    LOG(ERROR) << "error: " << OpenCLErrorToString(error);  \
    return MaceStatus::MACE_OUT_OF_RESOURCES;               \
  }

class OpenCLEventWrapper {
public:
  explicit OpenCLEventWrapper(const EventOpType op_type,
                              std::vector<cl::Event> *events)
      : op_type_(op_type),
        events_(events) {}

  ~OpenCLEventWrapper() {}

  EventOpType op_type() const {
    return op_type_;
  }

  std::vector<cl::Event> *events() const {
    return events_.get();
  }

private:
  const EventOpType op_type_;
  std::shared_ptr<std::vector<cl::Event>> events_;
};

class OpenCLEventManager {
public:
  OpenCLEventManager();
  ~OpenCLEventManager();

  cl::Event *CreateSingleEvent(const EventActionType act_type,
                               const EventOpType op_type);

  cl::UserEvent *CreateSingleUserEvent(cl::Context &context,
                                       const EventActionType act_type,
                                       const EventOpType op_type);

  void InsertSingleWaitEvent(const cl::Event *event);

  void CreateSetAndInsertWaitSingleEvent(const EventOpType op_type);

  void CreateWaitEventFromSetEvent();

  void InsertNullEvent(const EventActionType act_type);

  std::vector<cl::Event> *GetLastEvents(const EventActionType act_type) const;

  std::vector<cl::Event> *GetLastEvents(const EventActionType act_type,
                                        const EventOpType op_type,
                                        const size_t n) const;

  void PrintLastEventInfo(const EventActionType act_type) const;

  static void SetUserEventStatus(cl::UserEvent *event, cl_int status);

  static void SetUserEventComplete(cl::UserEvent *event);

  void Clear();

private:
  std::vector<cl::Event> *GetLastEventsInternal(
      const std::vector<OpenCLEventWrapper> &event_list,
      const EventOpType op_type,
      const size_t n) const;

  void ClearInternal(std::vector<OpenCLEventWrapper> &event_list);

  std::vector<OpenCLEventWrapper> set_event_list_;
  std::vector<OpenCLEventWrapper> wait_event_list_;
};

class OpenCLRuntime {
 public:
  OpenCLRuntime(
      std::shared_ptr<KVStorage> cache_storage = nullptr,
      const GPUPriorityHint priority_hint = GPUPriorityHint::PRIORITY_NORMAL,
      const GPUPerfHint perf_hint = GPUPerfHint::PERF_NORMAL,
      std::shared_ptr<KVStorage> precompiled_binary_storage = nullptr,
      std::shared_ptr<Tuner<uint32_t>> tuner = nullptr);
  ~OpenCLRuntime();
  OpenCLRuntime(const OpenCLRuntime &) = delete;
  OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

  cl::Context &context();
  cl::Device &device();
  cl::CommandQueue &command_queue();
  GPUType gpu_type() const;
  const std::string platform_info() const;
  uint32_t device_global_mem_cacheline_size() const;
  uint64_t device_global_mem_cache_size() const;
  uint32_t device_compute_units() const;
  uint32_t warp_size() const;
  uint32_t kwg_size() const;
  Tuner<uint32_t> *tuner();
  bool is_opencl_avaliable();
  OpenCLEventManager *event_manager();

  IONType ion_type() const;
#ifdef MACE_ENABLE_RPCMEM
  uint32_t qcom_ext_mem_padding() const;
  uint32_t qcom_page_size() const;
  uint32_t qcom_host_cache_policy() const;
#endif  // MACE_ENABLE_RPCMEM

  void GetCallStats(const cl::Event &event, CallStats *stats);
  uint64_t GetDeviceMaxWorkGroupSize();
  uint64_t GetDeviceMaxMemAllocSize();
  bool IsImageSupport();
  std::vector<uint64_t> GetMaxImage2DSize();
  uint64_t GetKernelMaxWorkGroupSize(const cl::Kernel &kernel);
  uint64_t GetKernelWaveSize(const cl::Kernel &kernel);
  bool IsNonUniformWorkgroupsSupported() const;
  bool IsOutOfRangeCheckEnabled() const;
  bool is_profiling_enabled() const;

  MaceStatus BuildKernel(const std::string &program_name,
                         const std::string &kernel_name,
                         const std::set<std::string> &build_options,
                         cl::Kernel *kernel);

  void SaveBuiltCLProgram();

 private:
  bool BuildProgram(const std::string &program_file_name,
                    const std::string &binary_file_name,
                    const std::string &build_options,
                    cl::Program *program);
  bool BuildProgramFromCache(
      const std::string &built_program_key,
      const std::string &build_options_str,
      cl::Program *program);
  bool BuildProgramFromPrecompiledBinary(
      const std::string &built_program_key,
      const std::string &build_options_str,
      cl::Program *program);
  bool BuildProgramFromSource(
      const std::string &program_name,
      const std::string &built_program_key,
      const std::string &build_options_str,
      cl::Program *program);
  OpenCLVersion ParseDeviceVersion(const std::string &device_version);

 private:
  std::shared_ptr<KVStorage> cache_storage_;
  std::shared_ptr<KVStorage> precompiled_binary_storage_;
  std::shared_ptr<Tuner<uint32_t>> tuner_;
  bool is_opencl_avaliable_;
  bool is_profiling_enabled_;
  OpenCLVersion opencl_version_;
  GPUType gpu_type_;
  std::unique_ptr<OpenCLEventManager> event_manager_;
  // All OpenCL object must be a pointer and manually deleted before unloading
  // OpenCL library.
  std::shared_ptr<cl::Context> context_;
  std::shared_ptr<cl::Device> device_;
  std::shared_ptr<cl::CommandQueue> command_queue_;
  std::map<std::string, cl::Program> built_program_map_;
  std::map<std::string, cl::Kernel> built_kernel_map_;
  std::mutex program_build_mutex_;
  std::string platform_info_;
  std::string precompiled_binary_platform_info_;
  bool out_of_range_check_;
  uint64_t device_global_mem_cacheline_size_;
  uint64_t device_global_mem_cache_size_;
  uint64_t device_compute_units_;
  uint64_t warp_size_;
  uint64_t max_wgp_size_;
  uint64_t kwg_size_;

  IONType ion_type_;
#ifdef MACE_ENABLE_RPCMEM
  uint32_t qcom_ext_mem_padding_;
  uint32_t qcom_page_size_;
  uint32_t qcom_host_cache_policy_;
#endif  // MACE_ENABLE_RPCMEM
};

class OpenCLProfilingTimer : public Timer {
 public:
  OpenCLProfilingTimer(OpenCLRuntime *runtime, const cl::Event *event)
      : runtime_(runtime), event_(event), accumulated_micros_(0) {}
  void StartTiming() override;
  void StopTiming() override;
  void AccumulateTiming() override;
  void ClearTiming() override;
  double ElapsedMicros() override;
  double AccumulatedMicros() override;

 private:
  OpenCLRuntime *runtime_;
  const cl::Event *event_;
  double start_nanos_;
  double stop_nanos_;
  double accumulated_micros_;
};
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
