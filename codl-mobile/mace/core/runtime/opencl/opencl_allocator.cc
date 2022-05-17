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

#include <memory>

#include "mace/core/runtime/opencl/opencl_allocator.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/utils/count_down_latch.h"

namespace mace {

namespace {

static cl_channel_type DataTypeToCLChannelType(const DataType t) {
  switch (t) {
    case DT_HALF:
      return CL_HALF_FLOAT;
    case DT_FLOAT:
      return CL_FLOAT;
    case DT_INT32:
      return CL_SIGNED_INT32;
    case DT_UINT8:
      return CL_UNSIGNED_INT32;
    default:
      LOG(FATAL) << "Image doesn't support the data type: " << t;
      return 0;
  }
}
}  // namespace

OpenCLAllocator::OpenCLAllocator(
    OpenCLRuntime *opencl_runtime):
    opencl_runtime_(opencl_runtime) {}

OpenCLAllocator::~OpenCLAllocator() {
  //DeleteAll();
}

MaceStatus OpenCLAllocator::New(size_t nbytes, void **result) {
  if (nbytes == 0) {
    return MaceStatus::MACE_SUCCESS;
  }
  VLOG(3) << "Allocate OpenCL buffer: " << nbytes;

  if (ShouldMockRuntimeFailure()) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  cl_int error = CL_SUCCESS;
  cl::Buffer *buffer = nullptr;
#ifdef MACE_ENABLE_RPCMEM
  if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
    cl_mem_ion_host_ptr ion_host;
    CreateQualcommBufferIONHostPtr(nbytes, &ion_host);

    buffer = new cl::Buffer(
        opencl_runtime_->context(),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        nbytes, &ion_host, &error);

    cl_to_host_map_[static_cast<void *>(buffer)] = ion_host.ion_hostptr;
  } else {
#endif  // MACE_ENABLE_RPCMEM
    buffer = new cl::Buffer(opencl_runtime_->context(),
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            nbytes, nullptr, &error);
#ifdef MACE_ENABLE_RPCMEM
  }
#endif  // MACE_ENABLE_RPCMEM
  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL Buffer with "
                 << nbytes << " bytes failed because of "
                 << OpenCLErrorToString(error);
    delete buffer;
    *result = nullptr;
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  } else {
    alloc_buffer_list_.insert(
        std::pair<const cl::Buffer *, const size_t>(buffer, nbytes));
    *result = buffer;
    return MaceStatus::MACE_SUCCESS;
  }
}

MaceStatus OpenCLAllocator::NewImage(const std::vector<size_t> &image_shape,
                                     const DataType dt,
                                     void **result) {
  MACE_CHECK(image_shape.size() == 2, "Image shape's size must equal 2");
  MACE_CHECK(image_shape[0] > 0 && image_shape[1] > 0,
             "Image shape's size must be more than 0");
  VLOG(3) << "Allocate OpenCL image: " << image_shape[0] << ", "
          << image_shape[1];

  if (ShouldMockRuntimeFailure()) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  cl::ImageFormat img_format(CL_RGBA, DataTypeToCLChannelType(dt));

  cl_int error;
  cl::Image2D *cl_image =
      new cl::Image2D(opencl_runtime_->context(),
                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, img_format,
                      image_shape[0], image_shape[1], 0, nullptr, &error);
  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL image with shape: ["
                 << image_shape[0] << ", " << image_shape[1]
                 << "] failed because of "
                 << OpenCLErrorToString(error);
    // Many users have doubts at CL_INVALID_IMAGE_SIZE, add some tips.
    if (error == CL_INVALID_IMAGE_SIZE) {
      auto max_2d_size = opencl_runtime_->GetMaxImage2DSize();
      LOG(WARNING) << "The allowable OpenCL image size is: "
                   << max_2d_size[0] << "x" << max_2d_size[1];
    }
    delete cl_image;
    *result = nullptr;
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  } else {
    alloc_image_list_.insert(
        std::pair<const cl::Image2D *, const std::vector<size_t>>(cl_image, image_shape));
    *result = cl_image;
    return MaceStatus::MACE_SUCCESS;
  }
}

void OpenCLAllocator::Delete(void *buffer) {
  VLOG(3) << "Free OpenCL buffer";
  if (buffer != nullptr) {
    cl::Buffer *cl_buffer = static_cast<cl::Buffer *>(buffer);
    auto iter = alloc_buffer_list_.find(cl_buffer);
    if (iter != alloc_buffer_list_.end()) {
      alloc_buffer_list_.erase(iter);
    }

    delete cl_buffer;
#ifdef MACE_ENABLE_RPCMEM
    if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
      auto it = cl_to_host_map_.find(buffer);
      MACE_CHECK(it != cl_to_host_map_.end(), "OpenCL buffer not found!");
      rpcmem_.Delete(it->second);
      cl_to_host_map_.erase(buffer);
    }
#endif  // MACE_ENABLE_RPCMEM
  }
}

void OpenCLAllocator::DeleteImage(void *buffer) {
  VLOG(3) << "Free OpenCL image";
  if (buffer != nullptr) {
    cl::Image2D *cl_image = static_cast<cl::Image2D *>(buffer);
    auto iter = alloc_image_list_.find(cl_image);
    if (iter != alloc_image_list_.end()) {
      alloc_image_list_.erase(iter);
    }
    
    delete cl_image;
  }
}

void OpenCLAllocator::DeleteAll() {
  if (alloc_buffer_list_.size() > 0) {
    LOG(WARNING) << alloc_buffer_list_.size()
                 << " OpenCL buffers are not deleted,"
                 << " allocator will delete them automatically.";
    for (auto iter = alloc_buffer_list_.begin();
        iter != alloc_buffer_list_.end(); iter++) {
      const cl::Buffer *buffer = iter->first;
      delete buffer;
    }
  }

  if (alloc_image_list_.size() > 0) {
    LOG(WARNING) << alloc_image_list_.size()
                 << " OpenCL images are not deleted,"
                 << " allocator will delete them automatically.";
    for (auto iter = alloc_image_list_.begin();
        iter != alloc_image_list_.end(); iter++) {
      const cl::Image2D *image = iter->first;
      delete image;
    }
  }
}

void OpenCLAllocator::Read(void *buffer, void *dst, size_t offset, size_t nbytes) const {
  VLOG(3) << "Read OpenCL buffer";
  auto cl_buffer = static_cast<cl::Buffer *>(buffer);
  auto queue = opencl_runtime_->command_queue();
  
  queue.enqueueReadBuffer(*cl_buffer, CL_TRUE, offset, nbytes,
                          dst, nullptr, nullptr);
}

void CL_CALLBACK event_complete_callback(cl_event event,
                                         cl_int command_status,
                                         void *data) {
  MACE_UNUSED(event);
  MACE_UNUSED(command_status);
  MACE_UNUSED(data);
  utils::SimpleCountDownLatch *count_down_latch
      = static_cast<utils::SimpleCountDownLatch *>(data);
  count_down_latch->CountDown();
}

void *OpenCLAllocator::Map(void *buffer, size_t offset, size_t nbytes,
                           const AllocatorMapType map_type,
                           const BlockFlag block_flag,
                           StatsFuture *future,
                           utils::SimpleCountDownLatch *count_down_latch) {
  VLOG(3) << "Map OpenCL buffer";
  auto queue = opencl_runtime_->command_queue();
  void *mapped_ptr = nullptr;
#ifdef MACE_ENABLE_RPCMEM
  if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
    auto it = cl_to_host_map_.find(buffer);
    MACE_CHECK(it != cl_to_host_map_.end(), "Try to map unallocated Buffer!");
    mapped_ptr = it->second;

    const bool finish_cmd_queue = false;
    if (finish_cmd_queue) {
      queue.finish();
    }

    if (opencl_runtime_->qcom_host_cache_policy() ==
        CL_MEM_HOST_WRITEBACK_QCOM) {
      MACE_CHECK(rpcmem_.SyncCacheStart(mapped_ptr) == 0);
    }
  } else {
#endif  // MACE_ENABLE_RPCMEM
    auto cl_buffer = static_cast<cl::Buffer *>(buffer);
    // TODO(heliangliang) Non-blocking call
    OpenCLEventManager *event_manager = opencl_runtime_->event_manager();
    const std::vector<cl::Event> *wait_events
        = event_manager->GetLastEvents(EventActionType::WAIT);
    if (wait_events != nullptr) {
      LOG(INFO) << "Wait events from manager, addr " << wait_events;
    }
    std::vector<cl::Event> *events
        = event_manager->GetLastEvents(EventActionType::SET);
    cl::Event *event = (events != nullptr) ? &(events->at(0)) : nullptr;
    if (event != nullptr) {
      event_manager->PrintLastEventInfo(EventActionType::SET);
    }
    if (event == nullptr) {
      event = event_manager->CreateSingleEvent(EventActionType::SET,
                                               EventOpType::NONE);
      event_manager->InsertNullEvent(EventActionType::SET);
    }
    cl_int error;
    //void *mapped_ptr =
    //    queue.enqueueMapBuffer(*cl_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
    //                           offset, nbytes, nullptr, nullptr, &error);
    // NOTE(fucheng): Support blocking and non-blocking call.
    cl_bool blocking_map = static_cast<cl_bool>(block_flag);
    cl_mem_flags mem_flags = static_cast<cl_mem_flags>(map_type);
    mapped_ptr = 
        queue.enqueueMapBuffer(*cl_buffer, blocking_map, mem_flags,
                               offset, nbytes, wait_events, event, &error);
    if (error != CL_SUCCESS) {
      LOG(ERROR) << "Map buffer failed, error: " << OpenCLErrorToString(error);
      mapped_ptr = nullptr;
    }

    if (count_down_latch != nullptr) {
      event->setCallback(CL_COMPLETE, event_complete_callback, count_down_latch);
    }

    if (future != nullptr) {
      OpenCLRuntime *runtime = opencl_runtime_;
      future->wait_fn = [runtime, event](CallStats *stats) {
#define CODL_EVENT_STATUS_WAIT

#ifdef CODL_EVENT_WAIT
        event->wait();
#endif

#ifdef CODL_EVENT_STATUS_WAIT
        cl_int event_status;
        do {
          event->getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &event_status);
          //LOG(INFO) << "Event status " << static_cast<int>(event_status);
          for (size_t k = 0; k < 1000000; k ++);
        } while (event_status != CL_COMPLETE);
#endif

        if (stats != nullptr) {
          runtime->GetCallStats(*event, stats);
        }
      };
    }
#ifdef MACE_ENABLE_RPCMEM
  }
#endif  // MACE_ENABLE_RPCMEM

  return mapped_ptr;
}

// TODO(liuqi) there is something wrong with half type.
void *OpenCLAllocator::MapImage(void *buffer,
                                const std::vector<size_t> &image_shape,
                                std::vector<size_t> *mapped_image_pitch,
                                AllocatorMapType map_type, BlockFlag block_flag,
                                StatsFuture *future,
                                utils::SimpleCountDownLatch *count_down_latch) const {
  VLOG(3) << "Map OpenCL Image";
  MACE_CHECK(image_shape.size() == 2) << "Just support map 2d image";
  auto cl_image = static_cast<cl::Image2D *>(buffer);
  std::array<size_t, 3> origin = {{0, 0, 0}};
  std::array<size_t, 3> region = {{image_shape[0], image_shape[1], 1}};

  mapped_image_pitch->resize(2);
  cl::Event event;
  cl_int error;
  //void *mapped_ptr = opencl_runtime_->command_queue().enqueueMapImage(
  //    *cl_image, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, origin, region,
  //    mapped_image_pitch->data(), mapped_image_pitch->data() + 1, nullptr,
  //    nullptr, &error);
  // NOTE(fucheng): Support blocking and non-blocking call.
  cl_bool blocking_map = static_cast<cl_bool>(block_flag);
  cl_mem_flags mem_flags = static_cast<cl_mem_flags>(map_type);
  void *mapped_ptr = opencl_runtime_->command_queue().enqueueMapImage(
      *cl_image, blocking_map, mem_flags, origin, region,
      mapped_image_pitch->data(), mapped_image_pitch->data() + 1, nullptr,
      &event, &error);
  if (error != CL_SUCCESS) {
    LOG(ERROR) << "Map Image failed, error: " << OpenCLErrorToString(error);
    mapped_ptr = nullptr;
  }

  if (future != nullptr) {
    OpenCLRuntime *runtime = opencl_runtime_;
    future->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }

  MACE_UNUSED(count_down_latch);

  return mapped_ptr;
}

void OpenCLAllocator::Unmap(void *buffer,
                            void *mapped_ptr,
                            StatsFuture *future) {
  VLOG(3) << "Unmap OpenCL buffer/Image";
#ifdef MACE_ENABLE_RPCMEM
  if (opencl_runtime_->ion_type() == IONType::QUALCOMM_ION) {
    if (opencl_runtime_->qcom_host_cache_policy() ==
        CL_MEM_HOST_WRITEBACK_QCOM) {
      MACE_CHECK(rpcmem_.SyncCacheEnd(mapped_ptr) == 0);
    }
  } else {
#endif
    auto cl_buffer = static_cast<cl::Buffer *>(buffer);
    const std::vector<cl::Event> *wait_events
        = opencl_runtime_->event_manager()->GetLastEvents(EventActionType::WAIT);
    cl::Event event;
    auto queue = opencl_runtime_->command_queue();
    cl_int error = queue.enqueueUnmapMemObject(*cl_buffer, mapped_ptr,
                                               wait_events, &event);
    if (error != CL_SUCCESS) {
      LOG(ERROR) << "Unmap buffer failed, error: " << OpenCLErrorToString(error);
    }

    if (future != nullptr) {
      OpenCLRuntime *runtime = opencl_runtime_;
      future->wait_fn = [runtime, event](CallStats *stats) {
        event.wait();
        if (stats != nullptr) {
          runtime->GetCallStats(event, stats);
        }
      };
    }
#ifdef MACE_ENABLE_RPCMEM
  }
#endif
}

bool OpenCLAllocator::OnHost() const { return false; }

#ifdef MACE_ENABLE_RPCMEM
Rpcmem *OpenCLAllocator::rpcmem() {
  return &rpcmem_;
}

void OpenCLAllocator::CreateQualcommBufferIONHostPtr(
    const size_t nbytes,
    cl_mem_ion_host_ptr *ion_host) {
  void *host = rpcmem_.New(nbytes + opencl_runtime_->qcom_ext_mem_padding());
  MACE_CHECK_NOTNULL(host);
  auto host_addr = reinterpret_cast<std::uintptr_t>(host);
  auto page_size = opencl_runtime_->qcom_page_size();
  MACE_CHECK(host_addr % page_size == 0, "ION memory address: ", host_addr,
             " must be aligned to page size: ", page_size);
  int fd = rpcmem_.ToFd(host);
  MACE_CHECK(fd >= 0, "Invalid rpcmem file descriptor: ", fd);

  ion_host->ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
  ion_host->ext_host_ptr.host_cache_policy =
      opencl_runtime_->qcom_host_cache_policy();
  ion_host->ion_filedesc = fd;
  ion_host->ion_hostptr = host;
}

void OpenCLAllocator::CreateQualcommImageIONHostPtr(
    const std::vector<size_t> &shape,
    const cl::ImageFormat &format,
    size_t *pitch,
    cl_mem_ion_host_ptr *ion_host) {
  cl_int error = clGetDeviceImageInfoQCOM(
      opencl_runtime_->device().get(), shape[0], shape[1], &format,
      CL_IMAGE_ROW_PITCH, sizeof(*pitch), pitch, nullptr);
  MACE_CHECK(error == CL_SUCCESS, "clGetDeviceImageInfoQCOM failed, error: ",
             OpenCLErrorToString(error));

  CreateQualcommBufferIONHostPtr(*pitch * shape[1], ion_host);
}
#endif  // MACE_ENABLE_RPCMEM

}  // namespace mace
