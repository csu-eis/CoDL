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

#include "mace/core/runtime/opencl/opencl_runtime.h"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include "mace/codegen/opencl/encrypt_opencl_kernel.h"
#include "mace/core/kv_storage.h"
#include "mace/core/runtime/opencl/opencl_extension.h"
#include "mace/utils/macros.h"
#include "mace/utils/tuner.h"

namespace mace {

const std::string OpenCLErrorToString(cl_int error) {
  switch (error) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_COMPILE_PROGRAM_FAILURE:
      return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
      return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
      return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
      return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
      return "CL_INVALID_PROPERTY";
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
      return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
      return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "CL_INVALID_DEVICE_PARTITION_COUNT";
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    case CL_INVALID_PIPE_SIZE:
      return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE:
      return "CL_INVALID_DEVICE_QUEUE";
#endif
    default:
      return MakeString("UNKNOWN: ", error);
  }
}

const std::string GetGPUPrecisionName(GPUPrecision gpu_precision) {
  switch (gpu_precision) {
    case GPU_PRECISION_INT8:
      return "INT8";
    case GPU_PRECISION_FP16:
      return "FP16";
    case GPU_PRECISION_FP32:
      return "FP32";
    default:
      return "NONE";
  }
}

const std::string EventActionTypeToString(const EventActionType type) {
  switch (type) {
    case EventActionType::SET:
      return "SET";
    case EventActionType::WAIT:
      return "WAIT";
    default:
      return "UNKNOWN";
  }
}

const std::string EventOpTypeToString(const EventOpType type) {
  switch (type) {
    case EventOpType::NONE:
      return "NONE";
    case EventOpType::TRANSFORM_INPUT:
      return "TRANSFORM_INPUT";
    case EventOpType::TRANSFORM_OUTPUT:
      return "TRANSFORM_OUTPUT";
    default:
      return "UNKNOWN";
  }
}

namespace {
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
void OpenCLPrintfCallback(const char *buffer,
                          size_t length,
                          size_t final,
                          void *user_data) {
  MACE_UNUSED(final);
  MACE_UNUSED(user_data);
  fwrite(buffer, 1, length, stdout);
}
#endif

void GetAdrenoContextProperties(std::vector<cl_context_properties> *properties,
                                GPUPerfHint gpu_perf_hint,
                                GPUPriorityHint gpu_priority_hint) {
  MACE_CHECK_NOTNULL(properties);
  switch (gpu_perf_hint) {
    case GPUPerfHint::PERF_LOW:
      properties->push_back(CL_CONTEXT_PERF_HINT_QCOM);
      properties->push_back(CL_PERF_HINT_LOW_QCOM);
      break;
    case GPUPerfHint::PERF_NORMAL:
      properties->push_back(CL_CONTEXT_PERF_HINT_QCOM);
      properties->push_back(CL_PERF_HINT_NORMAL_QCOM);
      break;
    case GPUPerfHint::PERF_HIGH:
      properties->push_back(CL_CONTEXT_PERF_HINT_QCOM);
      properties->push_back(CL_PERF_HINT_HIGH_QCOM);
      break;
    default:
      break;
  }
  switch (gpu_priority_hint) {
    case GPUPriorityHint::PRIORITY_LOW:
      properties->push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
      properties->push_back(CL_PRIORITY_HINT_LOW_QCOM);
      break;
    case GPUPriorityHint::PRIORITY_NORMAL:
      properties->push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
      properties->push_back(CL_PRIORITY_HINT_NORMAL_QCOM);
      break;
    case GPUPriorityHint::PRIORITY_HIGH:
      properties->push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
      properties->push_back(CL_PRIORITY_HINT_HIGH_QCOM);
      break;
    default:
      break;
  }
  // The properties list should be terminated with 0
  properties->push_back(0);
}

GPUType ParseGPUType(const std::string &device_name) {
  constexpr const char *kQualcommAdrenoGPUStr = "QUALCOMM Adreno(TM)";
  constexpr const char *kMaliGPUStr = "Mali";
  constexpr const char *kPowerVRGPUStr = "PowerVR";

  if (device_name == kQualcommAdrenoGPUStr) {
    return GPUType::QUALCOMM_ADRENO;
  } else if (device_name.find(kMaliGPUStr) != std::string::npos) {
    return GPUType::MALI;
  } else if (device_name.find(kPowerVRGPUStr) != std::string::npos) {
    return GPUType::PowerVR;
  } else {
    return GPUType::UNKNOWN;
  }
}

#ifdef MACE_ENABLE_RPCMEM
IONType ParseIONType(const std::string &device_extensions) {
  constexpr const char *kQualcommIONStr = "cl_qcom_ion_host_ptr";

  if (device_extensions.find(kQualcommIONStr) != std::string::npos) {
    return IONType::QUALCOMM_ION;
  } else {
    return IONType::NONE_ION;
  }
}

uint32_t ParseQcomHostCachePolicy(const std::string &device_extensions) {
  constexpr const char *kQualcommIocoherentStr =
      "cl_qcom_ext_host_ptr_iocoherent";

  if (device_extensions.find(kQualcommIocoherentStr) != std::string::npos) {
    return CL_MEM_HOST_IOCOHERENT_QCOM;
  } else {
    return CL_MEM_HOST_WRITEBACK_QCOM;
  }
}

std::string QcomHostCachePolicyToString(uint32_t policy) {
  switch (policy) {
    case CL_MEM_HOST_IOCOHERENT_QCOM: return "CL_MEM_HOST_IOCOHERENT_QCOM";
    case CL_MEM_HOST_WRITEBACK_QCOM: return "CL_MEM_HOST_WRITEBACK_QCOM";
    default: return MakeString("UNKNOWN: ", policy);
  }
}
#endif  // MACE_ENABLE_RPCMEM

const char *kOpenCLPlatformInfoKey =
    "mace_opencl_precompiled_platform_info_key";
}  // namespace

void OpenCLProfilingTimer::StartTiming() {}

void OpenCLProfilingTimer::StopTiming() {
  runtime_->command_queue().finish();
  start_nanos_ = event_->getProfilingInfo<CL_PROFILING_COMMAND_START>();
  stop_nanos_ = event_->getProfilingInfo<CL_PROFILING_COMMAND_END>();
}

double OpenCLProfilingTimer::ElapsedMicros() {
  return (stop_nanos_ - start_nanos_) / 1000.0;
}

double OpenCLProfilingTimer::AccumulatedMicros() { return accumulated_micros_; }

void OpenCLProfilingTimer::AccumulateTiming() {
  StopTiming();
  accumulated_micros_ += (stop_nanos_ - start_nanos_) / 1000.0;
}

void OpenCLProfilingTimer::ClearTiming() {
  start_nanos_ = 0;
  stop_nanos_ = 0;
  accumulated_micros_ = 0;
}

OpenCLEventManager::OpenCLEventManager() {}

OpenCLEventManager::~OpenCLEventManager() {
  Clear();
}

cl::Event
*OpenCLEventManager::CreateSingleEvent(const EventActionType act_type,
                                       const EventOpType op_type) {
  std::vector<cl::Event> *events = new std::vector<cl::Event>(1);
  switch (act_type) {
    case EventActionType::SET:
      set_event_list_.emplace_back(op_type, events);
      break;
    case EventActionType::WAIT:
      wait_event_list_.emplace_back(op_type, events);
      break;
    default:
      LOG(ERROR) << "Not supported event action";
  }
  return &events->at(0);
}

cl::UserEvent
*OpenCLEventManager::CreateSingleUserEvent(cl::Context &context,
                                           const EventActionType act_type,
                                           const EventOpType op_type) {
  cl_int error;
  std::vector<cl::Event> *events = new std::vector<cl::Event>;
  events->push_back(cl::UserEvent(context, &error));
  if (error != CL_SUCCESS) {
    LOG(ERROR) << "Create user event failed, error: "
               << OpenCLErrorToString(error);
    return nullptr;
  } else {
    switch (act_type) {
      case EventActionType::SET:
        set_event_list_.emplace_back(op_type, events);
        break;
      case EventActionType::WAIT:
        wait_event_list_.emplace_back(op_type, events);
        break;
      default:
        LOG(ERROR) << "Not supported event action";
    }
    return reinterpret_cast<cl::UserEvent *>(&events->at(0));
  }
}

void OpenCLEventManager::InsertSingleWaitEvent(const cl::Event *event) {
  MACE_CHECK_NOTNULL(event);
  std::vector<cl::Event> *events = new std::vector<cl::Event>;
  events->push_back(*event);
  wait_event_list_.emplace_back(EventOpType::NONE, events);
}

void OpenCLEventManager::CreateSetAndInsertWaitSingleEvent(
    const EventOpType op_type) {
  std::vector<cl::Event> *events = new std::vector<cl::Event>(1);
  set_event_list_.emplace_back(op_type, events);
  wait_event_list_.emplace_back(op_type, events);
}

void OpenCLEventManager::CreateWaitEventFromSetEvent() {
  OpenCLEventWrapper &warpper = set_event_list_.back();
  wait_event_list_.emplace_back(warpper.op_type(), warpper.events());
}

void OpenCLEventManager::InsertNullEvent(const EventActionType act_type) {
  switch (act_type) {
    case EventActionType::SET:
      set_event_list_.emplace_back(EventOpType::NONE, nullptr);
      break;
    case EventActionType::WAIT:
      wait_event_list_.emplace_back(EventOpType::NONE, nullptr);
      break;
    default:
      LOG(ERROR) << "Not supported event action";
  }
}

std::vector<cl::Event>
*OpenCLEventManager::GetLastEvents(const EventActionType act_type) const {
  switch (act_type) {
    case EventActionType::SET:
      return set_event_list_.size() > 0 ? set_event_list_.back().events()
                                        : nullptr;
    case EventActionType::WAIT:
      return wait_event_list_.size() > 0 ? wait_event_list_.back().events()
                                         : nullptr;
    default:
      LOG(ERROR) << "Not supported event action";
      return nullptr;
  }
}

std::vector<cl::Event>
*OpenCLEventManager::GetLastEventsInternal(
    const std::vector<OpenCLEventWrapper> &event_list,
    const EventOpType op_type,
    const size_t n) const {
  if (event_list.size() == 0) {
    return nullptr;
  }

  size_t num_event_found = 0;
  for (int i = event_list.size() - 1; i >= 0; i --) {
    if (event_list[i].op_type() == op_type) {
      if (num_event_found == n) {
        return event_list[i].events();
      }
      num_event_found ++;
    }
  }

  return nullptr;
}

std::vector<cl::Event>
*OpenCLEventManager::GetLastEvents(const EventActionType act_type,
                                   const EventOpType op_type,
                                   const size_t n) const {
  switch (act_type) {
    case EventActionType::SET:
      return GetLastEventsInternal(set_event_list_, op_type, n);
    case EventActionType::WAIT:
      return GetLastEventsInternal(wait_event_list_, op_type, n);
    default:
      LOG(ERROR) << "Not supported event action";
      return nullptr;
  }
}

void
OpenCLEventManager::PrintLastEventInfo(const EventActionType act_type) const {
  const OpenCLEventWrapper *last_event_warpper;
  int event_id;
  switch (act_type) {
    case EventActionType::SET:
      MACE_CHECK(set_event_list_.size() > 0);
      last_event_warpper = &(set_event_list_.back());
      event_id = set_event_list_.size() - 1;
      break;
    case EventActionType::WAIT:
      MACE_CHECK(wait_event_list_.size() > 0);
      last_event_warpper = &(wait_event_list_.back());
      event_id = wait_event_list_.size() - 1;
      break;
  }

  LOG(INFO) << "Last Event:"
            << " Id=" << event_id
            << " ActType=" << EventActionTypeToString(act_type)
            << " OpType=" << EventOpTypeToString(last_event_warpper->op_type())
            << " Addr=" << last_event_warpper->events();
}

void OpenCLEventManager::SetUserEventStatus(cl::UserEvent *event, cl_int status) {
  cl_int error = event->setStatus(status);
  if (error != CL_SUCCESS) {
    LOG(ERROR) << "Set user event status failed, error: "
               << OpenCLErrorToString(error);
  }
}

void OpenCLEventManager::SetUserEventComplete(cl::UserEvent *event) {
  SetUserEventStatus(event, CL_COMPLETE);
}

void OpenCLEventManager::ClearInternal(
    std::vector<OpenCLEventWrapper> &event_list) {
  event_list.clear();
}

void OpenCLEventManager::Clear() {
  ClearInternal(set_event_list_);
  ClearInternal(wait_event_list_);
}

void ShowCLMemoryFlags() {
  LOG(INFO) << "OpenCL memory flags:"
            << " CL_MEM_READ_WRITE " << CL_MEM_READ_WRITE
            << ", CL_MEM_WRITE_ONLY " << CL_MEM_WRITE_ONLY
            << ", CL_MEM_READ_ONLY " << CL_MEM_READ_ONLY;
}

OpenCLRuntime::OpenCLRuntime(
    std::shared_ptr<KVStorage> cache_storage,
    const GPUPriorityHint priority_hint,
    const GPUPerfHint perf_hint,
    std::shared_ptr<KVStorage> precompiled_binary_storage,
    std::shared_ptr<Tuner<uint32_t>> tuner) :
    cache_storage_(cache_storage),
    precompiled_binary_storage_(precompiled_binary_storage),
    tuner_(tuner),
    is_opencl_avaliable_(false),
    is_profiling_enabled_(false),
    opencl_version_(CL_VER_UNKNOWN),
    gpu_type_(UNKNOWN),
    event_manager_(new OpenCLEventManager()) {
  LOG(INFO) << "Creating OpenCL runtime";
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.empty()) {
    LOG(ERROR) << "No OpenCL platforms found";
    return;
  }
  cl::Platform default_platform = all_platforms[0];
  std::stringstream ss;
  ss << default_platform.getInfo<CL_PLATFORM_NAME>()
     << ", " << default_platform.getInfo<CL_PLATFORM_PROFILE>() << ", "
     << default_platform.getInfo<CL_PLATFORM_VERSION>() << ", "
     << MaceVersion();
  platform_info_ = ss.str();
  VLOG(1) << "Using platform: " << platform_info_;
  LOG(INFO) << "Using platform: " << platform_info_;

  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if (all_devices.empty()) {
    LOG(ERROR) << "No OpenCL devices found";
    return;
  }

  bool gpu_detected = false;
  device_ = std::make_shared<cl::Device>();
  for (auto device : all_devices) {
    if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
      *device_ = device;
      gpu_detected = true;

      const std::string device_name = device.getInfo<CL_DEVICE_NAME>();
      gpu_type_ = ParseGPUType(device_name);

      const std::string device_version = device.getInfo<CL_DEVICE_VERSION>();
      opencl_version_ = ParseDeviceVersion(device_version);
      if (opencl_version_ == OpenCLVersion::CL_VER_UNKNOWN) {
        return;
      }

      const std::string device_extensions =
          device.getInfo<CL_DEVICE_EXTENSIONS>();
#ifdef MACE_ENABLE_RPCMEM
      ion_type_ = ParseIONType(device_extensions);
      if (ion_type_ == IONType::QUALCOMM_ION) {
        qcom_ext_mem_padding_ = 0;
        cl_int err = device.getInfo(CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM,
                                    &qcom_ext_mem_padding_);
        if (err != CL_SUCCESS) {
          LOG(ERROR) << "Failed to get CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM "
                     << OpenCLErrorToString(err);
        }

        qcom_page_size_ = 4096;
        err = device.getInfo(CL_DEVICE_PAGE_SIZE_QCOM, &qcom_page_size_);
        if (err != CL_SUCCESS) {
          LOG(ERROR) << "Failed to get CL_DEVICE_PAGE_SIZE_QCOM: "
                     << OpenCLErrorToString(err);
        }

        qcom_host_cache_policy_ = ParseQcomHostCachePolicy(device_extensions);

        LOG(INFO) << "Using QUALCOMM ION buffer with padding size: "
                  << qcom_ext_mem_padding_ << ", page size: " << qcom_page_size_
                  << ", with host cache policy: "
                  << QcomHostCachePolicyToString(qcom_host_cache_policy_);
      }
#else
      ion_type_ = IONType::NONE_ION;
#endif  // MACE_ENABLE_RPCMEM

      VLOG(1) << "Using device: " << device_name;
      // INFO(fucheng): Show device information.
      LOG(INFO) << "OpenCL device: " << device_name;
      LOG(INFO) << "OpenCL device version: " << device_version;
      LOG(INFO) << "OpenCL device extensions: " << device_extensions;
      break;
    }
  }
  
  // NOTE(fucheng): Support CPU as OpenCL device
  if (!gpu_detected) {
    LOG(WARNING) << "No GPU device found";

    // Detect CPU device
    bool cpu_detected = false;
    for (auto device : all_devices) {
      if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU) {
        *device_ = device;
        cpu_detected = true;

        const std::string device_name = device.getInfo<CL_DEVICE_NAME>();
        gpu_type_ = ParseGPUType(device_name);

        const std::string device_version = device.getInfo<CL_DEVICE_VERSION>();
        opencl_version_ = ParseDeviceVersion(device_version);
        if (opencl_version_ == OpenCLVersion::CL_VER_UNKNOWN) {
          return;
        }

        VLOG(1) << "Using device: " << device_name;
        LOG(INFO) << "Using platform: " << platform_info_;
        break;
      }
    }

    if (!cpu_detected) {
      LOG(ERROR) << "No CPU device found";
      return;
    }
  }

  cl_command_queue_properties properties = 0;

  const char *profiling = getenv("MACE_OPENCL_PROFILING");
  if (tuner_->IsTuning() ||
      (profiling != nullptr && strlen(profiling) == 1 && profiling[0] == '1')) {
    properties |= CL_QUEUE_PROFILING_ENABLE;
    is_profiling_enabled_ = true;
    LOG(INFO) << "OpenCL profiling: Enabled";
  }

  cl_int err;
  if (gpu_type_ == GPUType::QUALCOMM_ADRENO
      && opencl_version_ == OpenCLVersion::CL_VER_2_0) {
    std::vector<cl_context_properties> context_properties;
    context_properties.reserve(5);
    GetAdrenoContextProperties(&context_properties,
                               perf_hint,
                               priority_hint);
    context_ = std::shared_ptr<cl::Context>(
        new cl::Context({*device_}, context_properties.data(),
                        nullptr, nullptr, &err));
  } else {
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    if (is_profiling_enabled_ && gpu_type_ == GPUType::MALI) {
      std::vector<cl_context_properties> context_properties = {
          CL_CONTEXT_PLATFORM, (cl_context_properties) default_platform(),
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties) OpenCLPrintfCallback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000, 0
      };
      context_ = std::shared_ptr<cl::Context>(
          new cl::Context({*device_}, context_properties.data(),
                          nullptr, nullptr, &err));
    } else {
      context_ = std::shared_ptr<cl::Context>(
          new cl::Context({*device_}, nullptr, nullptr, nullptr, &err));
    }
#else
    context_ = std::shared_ptr<cl::Context>(
          new cl::Context({*device_}, nullptr, nullptr, nullptr, &err));
#endif
  }
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to create OpenCL Context: "
               << OpenCLErrorToString(err);
    return;
  }

  command_queue_ = std::make_shared<cl::CommandQueue>(*context_,
                                                      *device_,
                                                      properties,
                                                      &err);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to create OpenCL CommandQueue: "
               << OpenCLErrorToString(err);
    return;
  }

  std::string cached_binary_platform_info;
  if (cache_storage_ != nullptr) {
    if (cache_storage_->Load() != 0) {
      LOG(WARNING) << "Load OpenCL cached compiled kernel file failed. "
                   << "Please make sure the storage directory exist "
                   << "and you have Write&Read permission";
    }
    auto platform_info_array =
        this->cache_storage_->Find(kOpenCLPlatformInfoKey);
    if (platform_info_array != nullptr) {
      cached_binary_platform_info =
          std::string(platform_info_array->begin(),
                      platform_info_array->end());
      if (cached_binary_platform_info != platform_info_) {
        cache_storage_->Clear();
      }
    }
  }

  if (cached_binary_platform_info != platform_info_) {
    if (precompiled_binary_storage_ == nullptr) {
      VLOG(1) << "There is no precompiled OpenCL binary in"
                 " all OpenCL binary paths.";
    } else {
      if (precompiled_binary_storage_->Load() != 0) {
        LOG(WARNING) << "Load OpenCL precompiled kernel file failed. "
                     << "Please make sure the storage directory exist "
                     << "and you have Write&Read permission";
      }

      auto platform_info_array =
          this->precompiled_binary_storage_->Find(kOpenCLPlatformInfoKey);
      if (platform_info_array != nullptr) {
        precompiled_binary_platform_info_ =
            std::string(platform_info_array->begin(),
                        platform_info_array->end());
      }
    }
  }

  device_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                   &device_global_mem_cacheline_size_);
  device_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                   &device_global_mem_cache_size_);

  device_->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,
                   &device_compute_units_);
  const char *out_of_range_check = getenv("MACE_OUT_OF_RANGE_CHECK");
  if (out_of_range_check != nullptr && strlen(out_of_range_check) == 1
      && out_of_range_check[0] == '1') {
    this->out_of_range_check_ = true;
  } else {
    this->out_of_range_check_ = false;
  }

  is_opencl_avaliable_ = true;

  // NOTE(fucheng): Read warp size from configure file.
  utils::CodlConfig *codl_config = utils::GetGlobalCodlConfig();
  if (codl_config->is_loaded()) {
    warp_size_ = codl_config->gpu_warp_size();
    kwg_size_ = codl_config->kwg_size();
  }

  // DEBUG(fucheng): Show device info.
  LOG(INFO) << "OpenCLRuntime:"
            << " global_mem_cacheline_size " << device_global_mem_cacheline_size_
            << ", global_mem_cache_size " << device_global_mem_cache_size_
            << ", compute_units " << device_compute_units_
            << ", warp_size " << warp_size_
            << ", kwg_size " << kwg_size_;
  ShowCLMemoryFlags();
}

OpenCLRuntime::~OpenCLRuntime() {
  if (command_queue_ != nullptr) {
    command_queue_->finish();
  }
  built_program_map_.clear();
  // We need to control the destruction order, which has dependencies
  command_queue_.reset();
  context_.reset();
  device_.reset();
}

bool OpenCLRuntime::is_opencl_avaliable() {
  static const uint64_t kMinWorkGroupSize = 64;
  return is_opencl_avaliable_
      && GetDeviceMaxWorkGroupSize() >= kMinWorkGroupSize;
}

OpenCLEventManager *OpenCLRuntime::event_manager() {
  return event_manager_.get();
  //return &event_manager_;
}

cl::Context &OpenCLRuntime::context() { return *context_; }

cl::Device &OpenCLRuntime::device() { return *device_; }

cl::CommandQueue &OpenCLRuntime::command_queue() { return *command_queue_; }

Tuner<uint32_t> *OpenCLRuntime::tuner() { return tuner_.get(); }

uint32_t OpenCLRuntime::device_global_mem_cacheline_size() const {
  return device_global_mem_cacheline_size_;
}

uint64_t OpenCLRuntime::device_global_mem_cache_size() const {
  return device_global_mem_cache_size_;
}

uint32_t OpenCLRuntime::device_compute_units() const {
  return device_compute_units_;
}

uint32_t OpenCLRuntime::warp_size() const {
  return warp_size_;
}

uint32_t OpenCLRuntime::kwg_size() const {
  return kwg_size_;
}

bool OpenCLRuntime::BuildProgramFromCache(
    const std::string &built_program_key,
    const std::string &build_options_str,
    cl::Program *program) {
  // Find from binary
  if (this->cache_storage_ == nullptr) return false;
  auto content = this->cache_storage_->Find(built_program_key);
  if (content == nullptr) {
    return false;
  }

  *program = cl::Program(context(), {device()}, {*content});
  cl_int ret = program->build({device()}, build_options_str.c_str());
  if (ret != CL_SUCCESS) {
    if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
        CL_BUILD_ERROR) {
      std::string build_log =
          program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
      LOG(INFO) << "Program build log: " << build_log;
    }
    LOG(WARNING) << "Build program "
                 << built_program_key << " from Cache failed:"
                 << MakeString(ret);
    return false;
  }
  VLOG(3) << "Program from Cache: " << built_program_key;
  return true;
}

bool OpenCLRuntime::BuildProgramFromPrecompiledBinary(
    const std::string &built_program_key,
    const std::string &build_options_str,
    cl::Program *program) {
  // Find from binary
  if (this->precompiled_binary_storage_ == nullptr) return false;
  if (precompiled_binary_platform_info_ != platform_info_) {
    VLOG(3) << "precompiled OpenCL binary version "
            << precompiled_binary_platform_info_
            << " is not same with current version";
    return false;
  }
  auto content = this->precompiled_binary_storage_->Find(built_program_key);
  if (content == nullptr) {
    return false;
  }

  *program = cl::Program(context(), {device()}, {*content});
  cl_int ret = program->build({device()}, build_options_str.c_str());
  if (ret != CL_SUCCESS) {
    if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
        CL_BUILD_ERROR) {
      std::string build_log =
          program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
      LOG(INFO) << "Program build log: " << build_log;
    }
    LOG(WARNING) << "Build program "
                 << built_program_key << " from precompiled binary failed:"
                 << MakeString(ret);
    return false;
  }
  VLOG(3) << "Program from precompiled binary: " << built_program_key;
  return true;
}

MaceStatus GetProgramSourceByName(const std::string &program_name,
                              std::string *source) {
  MACE_CHECK_NOTNULL(source);
  std::stringstream source_stream;
  const auto &kEncryptedProgramMap = mace::codegen::kEncryptedProgramMap;
  const auto &it_program = kEncryptedProgramMap.find(program_name);
  if (it_program == kEncryptedProgramMap.end()) {
    LOG(ERROR) << "Find program " << program_name << " failed.";
    return MaceStatus::MACE_RUNTIME_ERROR;
  }

  const std::vector<std::string> &headers = it_program->second.headers_;
  for (const std::string &header : headers) {
    const auto &header_program = kEncryptedProgramMap.find(header);
    if (header_program == kEncryptedProgramMap.end()) {
      LOG(WARNING) << "Program header(" << header << ") is empty.";
      continue;
    }

    const auto &header_source = header_program->second.encrypted_code_;
    source_stream << ObfuscateString(
        std::string(header_source.begin(), header_source.end()));
  }

  const auto &it_source = it_program->second.encrypted_code_;
  source_stream << ObfuscateString(
      std::string(it_source.begin(), it_source.end()));
  *source = source_stream.str();

  return MaceStatus::MACE_SUCCESS;
}

bool OpenCLRuntime::BuildProgramFromSource(
    const std::string &program_name,
    const std::string &built_program_key,
    const std::string &build_options_str,
    cl::Program *program) {
  std::string kernel_source;
  MaceStatus status = GetProgramSourceByName(program_name, &kernel_source);
  if (status == MaceStatus::MACE_SUCCESS && !kernel_source.empty()) {
    cl::Program::Sources sources;
    sources.push_back(kernel_source);
    *program = cl::Program(context(), sources);
    cl_int ret = program->build({device()}, build_options_str.c_str());
    if (ret != CL_SUCCESS) {
      if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
          CL_BUILD_ERROR) {
        std::string build_log =
            program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
        LOG(INFO) << "Program build log: " << build_log;
      }
      LOG(WARNING) << "Build program "
                   << program_name << " from source failed: "
                   << MakeString(ret);
      return false;
    }

    // Keep built program binary
    size_t device_list_size = 1;
    std::unique_ptr<size_t[]> program_binary_sizes(
        new size_t[device_list_size]);
    cl_int err = clGetProgramInfo((*program)(), CL_PROGRAM_BINARY_SIZES,
                                  sizeof(size_t) * device_list_size,
                                  program_binary_sizes.get(), nullptr);
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "error: " << OpenCLErrorToString(err);
      return false;
    }
    std::unique_ptr<std::unique_ptr<unsigned char[]>[]> program_binaries(
        new std::unique_ptr<unsigned char[]>[device_list_size]);
    for (cl_uint i = 0; i < device_list_size; ++i) {
      program_binaries[i] = std::unique_ptr<unsigned char[]>(
          new unsigned char[program_binary_sizes[i]]);
    }

    err = clGetProgramInfo((*program)(), CL_PROGRAM_BINARIES,
                           sizeof(unsigned char *) * device_list_size,
                           program_binaries.get(), nullptr);
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "error: " << OpenCLErrorToString(err);
      return false;
    }
    std::vector<unsigned char> content(
        reinterpret_cast<unsigned char const *>(program_binaries[0].get()),
        reinterpret_cast<unsigned char const *>(program_binaries[0].get()) +
            program_binary_sizes[0]);

    if (this->cache_storage_ != nullptr) {
      this->cache_storage_->Insert(built_program_key, content);
      // update platform info
      this->cache_storage_->Insert(
          kOpenCLPlatformInfoKey,
          std::vector<unsigned char>(platform_info_.begin(),
                                     platform_info_.end()));
    }

    VLOG(3) << "Program from source: " << built_program_key;
  }
  return true;
}

bool OpenCLRuntime::BuildProgram(const std::string &program_name,
                                 const std::string &built_program_key,
                                 const std::string &build_options,
                                 cl::Program *program) {
  MACE_CHECK_NOTNULL(program);

  std::string build_options_str =
      build_options + " -Werror -cl-mad-enable -cl-fast-relaxed-math";
  // Build flow: cache -> precompiled binary -> source
  bool ret = BuildProgramFromCache(built_program_key,
                                   build_options_str, program);
  if (!ret) {
    ret = BuildProgramFromPrecompiledBinary(built_program_key,
                                            build_options_str, program);
    if (!ret) {
      ret = BuildProgramFromSource(program_name, built_program_key,
                                   build_options_str, program);
    }
  }
  return ret;
}

MaceStatus OpenCLRuntime::BuildKernel(
    const std::string &program_name,
    const std::string &kernel_name,
    const std::set<std::string> &build_options,
    cl::Kernel *kernel) {
  std::string build_options_str;
  for (auto &option : build_options) {
    build_options_str += " " + option;
  }
  std::string built_program_key = program_name + build_options_str;

  std::lock_guard<std::mutex> lock(program_build_mutex_);
  auto built_program_it = built_program_map_.find(built_program_key);
  cl::Program program;
  if (built_program_it != built_program_map_.end()) {
    VLOG(1) << "Program found";
    program = built_program_it->second;
  } else {
    VLOG(1) << "Program not found";
    bool ret = this->BuildProgram(program_name, built_program_key,
                                  build_options_str, &program);
    if (!ret) {
      return MaceStatus::MACE_OUT_OF_RESOURCES;
    }
    built_program_map_.emplace(built_program_key, program);
  }
  
  auto build_kernel_it = built_kernel_map_.find(kernel_name);
  if (build_kernel_it != built_kernel_map_.end()) {
    VLOG(1) << "Kernel found";
    *kernel = build_kernel_it->second;
  } else {
    VLOG(1) << "Kernel not found";
    cl_int err;
    *kernel = cl::Kernel(program, kernel_name.c_str(), &err);
    MACE_CL_RET_STATUS(err);
    //built_kernel_map_.emplace(kernel_name, *kernel);
  }
  
  return MaceStatus::MACE_SUCCESS;
}

void OpenCLRuntime::SaveBuiltCLProgram() {
  if (cache_storage_ != nullptr) {
    if (cache_storage_->Flush() != 0) {
      LOG(FATAL) << "Store OPENCL compiled kernel to file failed. "
                 << "Please make sure the storage directory exist "
                 << "and you have Write&Read permission";
    }
  }
}

void ShowEventTime(const cl::Event &event) {
  int64_t start_micros[4], end_micros[4];
  double time_millis[4];

  // Queued -> Submit
  start_micros[0] = event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() / 1000;
  end_micros[0] = event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() / 1000;
  time_millis[0] = (end_micros[0] - start_micros[0]) / 1000.0;

  // Submit -> Start
  start_micros[1] = event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() / 1000;
  end_micros[1] = event.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
  time_millis[1] = (end_micros[1] - start_micros[1]) / 1000.0;

  // Start -> End
  start_micros[2] = event.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
  end_micros[2] = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;
  time_millis[2] = (end_micros[2] - start_micros[2]) / 1000.0;

  // End -> Complete
  start_micros[3] = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;
  //end_micros[3] = event.getProfilingInfo<CL_PROFILING_COMMAND_COMPLETE>() / 1000;
  end_micros[3] = start_micros[3];
  time_millis[3] = (end_micros[3] - start_micros[3]) / 1000.0;
  time_millis[3] = 0.0f;

  VLOG(1) << "EventTime:"
          << " queued " << start_micros[0]
          << ", submit " << start_micros[1]
          << ", start " << start_micros[2]
          << ", end " << start_micros[3]
          << ", complete " << 0;

  VLOG(1) << "EventLatency:"
          << " submit " << time_millis[0] << " ms"
          << ", start " << time_millis[1] << " ms"
          << ", end " << time_millis[2] << " ms"
          << ", complete " << time_millis[3] << " ms";
}

void OpenCLRuntime::GetCallStats(const cl::Event &event, CallStats *stats) {
  //LOG(INFO) << "GetCallStats";
  ShowEventTime(event);
  if (stats != nullptr) {
    // NOTE(fucheng): CL_PROFILING_COMMAND_START
    const int64_t start
        = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    const int64_t end
        = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    //MACE_CHECK(start != end);
    stats->start_micros = start / 1000;
    stats->end_micros = end / 1000;
  }
}

uint64_t OpenCLRuntime::GetDeviceMaxWorkGroupSize() {
  uint64_t size = 0;
  cl_int err = device_->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    size = 0;
  }
  return size;
}

uint64_t OpenCLRuntime::GetDeviceMaxMemAllocSize() {
  uint64_t size = 0;
  cl_int err = device_->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    size = 0;
  }
  return size;
}

bool OpenCLRuntime::IsImageSupport() {
  cl_bool res;
  cl_int err = device_->getInfo(CL_DEVICE_IMAGE_SUPPORT, &res);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    return false;
  }
  return res == CL_TRUE;
}
std::vector<uint64_t> OpenCLRuntime::GetMaxImage2DSize() {
  size_t max_height, max_width;
  cl_int err = device_->getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &max_height);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    return {};
  }
  err = device_->getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &max_width);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    return {};
  }
  return {max_width, max_height};
}

uint64_t OpenCLRuntime::GetKernelMaxWorkGroupSize(const cl::Kernel &kernel) {
  uint64_t size = 0;
  cl_int err = kernel.getWorkGroupInfo(*device_, CL_KERNEL_WORK_GROUP_SIZE,
                                       &size);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    size = 0;
  }
  return size;
}

uint64_t OpenCLRuntime::GetKernelWaveSize(const cl::Kernel &kernel) {
  uint64_t size = 0;
  cl_int err = kernel.getWorkGroupInfo(*device_, CL_KERNEL_WAVE_SIZE_QCOM,
                                       &size);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    size = 0;
  }
  return size;
}

bool OpenCLRuntime::IsNonUniformWorkgroupsSupported() const {
  return (gpu_type_ == GPUType::QUALCOMM_ADRENO &&
      opencl_version_ == OpenCLVersion::CL_VER_2_0);
}

GPUType OpenCLRuntime::gpu_type() const {
  return gpu_type_;
}

IONType OpenCLRuntime::ion_type() const {
  return ion_type_;
}

#ifdef MACE_ENABLE_RPCMEM
uint32_t OpenCLRuntime::qcom_ext_mem_padding() const {
  return qcom_ext_mem_padding_;
}

uint32_t OpenCLRuntime::qcom_page_size() const {
  return qcom_page_size_;
}

uint32_t OpenCLRuntime::qcom_host_cache_policy() const {
  return qcom_host_cache_policy_;
}
#endif  // MACE_ENABLE_RPCMEM

const std::string OpenCLRuntime::platform_info() const {
  return platform_info_;
}

OpenCLVersion OpenCLRuntime::ParseDeviceVersion(
    const std::string &device_version) {
  // OpenCL Device version string format:
  // OpenCL<space><major_version.minor_version><space>
  // <vendor-specific information>

  // NOTE(fucheng): Support 2.1 as 2.0
  auto words = Split(device_version, ' ');
  if (words[1] == "2.0" || words[1] == "2.1") {
    return OpenCLVersion::CL_VER_2_0;
  } else if (words[1] == "1.2") {
    return OpenCLVersion::CL_VER_1_2;
  } else if (words[1] == "1.1") {
    return OpenCLVersion::CL_VER_1_1;
  } else if (words[1] == "1.0") {
    return OpenCLVersion::CL_VER_1_0;
  } else {
    LOG(ERROR) << "Do not support OpenCL version: " << words[1];
    return OpenCLVersion::CL_VER_UNKNOWN;
  }
}

bool OpenCLRuntime::IsOutOfRangeCheckEnabled() const {
  return out_of_range_check_;
}

bool OpenCLRuntime::is_profiling_enabled() const {
  return is_profiling_enabled_;
}

}  // namespace mace
