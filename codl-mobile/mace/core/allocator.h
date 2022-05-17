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

#ifndef MACE_CORE_ALLOCATOR_H_
#define MACE_CORE_ALLOCATOR_H_

#include <cstdlib>
#include <map>
#include <limits>
#include <vector>
#include <cstring>

#include "mace/utils/macros.h"
#include "mace/core/future.h"
#include "mace/core/types.h"
#include "mace/core/runtime_failure_mock.h"
#include "mace/public/mace.h"
#include "mace/utils/logging.h"
#include "mace/utils/thread_count_down_latch.h"

#ifdef MACE_ENABLE_RPCMEM
#include "mace/core/rpcmem.h"
#endif  // MACE_ENABLE_RPCMEM

namespace mace {

// NOTE(fucheng): Add allocator map and block type.
enum AllocatorMapType {
  AMT_READ_WRITE = (1 << 0),
  AMT_WRITE_ONLY = (1 << 1),
  AMT_READ_ONLY  = (1 << 2)
};

enum BlockFlag {
  BF_FALSE = 0,
  BF_TRUE  = 1
};

#if defined(__hexagon__)
constexpr size_t kMaceAlignment = 128;
#elif defined(__ANDROID__)
// arm cache line
constexpr size_t kMaceAlignment = 64;
#else
// 32 bytes = 256 bits (AVX512)
constexpr size_t kMaceAlignment = 32;
#endif

inline index_t PadAlignSize(index_t size) {
  return (size + kMaceAlignment - 1) & (~(kMaceAlignment - 1));
}

class Allocator {
 public:
  Allocator() {}
  virtual ~Allocator() noexcept {}
  virtual MaceStatus New(size_t nbytes, void **result) = 0;
  virtual MaceStatus NewImage(const std::vector<size_t> &image_shape,
                              const DataType dt,
                              void **result) = 0;
  virtual void Delete(void *data) = 0;
  virtual void DeleteImage(void *data) = 0;
  virtual void DeleteAll() = 0;
  virtual void Read(void *buffer, void *dst,
                    size_t offset, size_t length) const = 0;
  virtual void *Map(void *buffer, size_t offset, size_t nbytes,
                    const AllocatorMapType map_type,
                    const BlockFlag block_flag,
                    StatsFuture *future = nullptr,
                    utils::SimpleCountDownLatch *count_down_latch = nullptr) = 0;
  virtual void *MapImage(void *buffer,
                         const std::vector<size_t> &image_shape,
                         std::vector<size_t> *mapped_image_pitch,
                         AllocatorMapType map_type, BlockFlag block_flag,
                         StatsFuture *future = nullptr,
                         utils::SimpleCountDownLatch *count_down_latch = nullptr) const = 0;
  virtual void Unmap(void *buffer, void *mapper_ptr,
                     StatsFuture *future = nullptr) = 0;
  virtual bool OnHost() const = 0;
#ifdef MACE_ENABLE_RPCMEM
  virtual Rpcmem *rpcmem() = 0;
#endif  // MACE_ENABLE_RPCMEM
};

class CPUAllocator : public Allocator {
 public:
  ~CPUAllocator() override {}
  MaceStatus New(size_t nbytes, void **result) override {
    VLOG(3) << "Allocate CPU buffer: " << nbytes;
    if (nbytes == 0) {
      return MaceStatus::MACE_SUCCESS;
    }

    if (ShouldMockRuntimeFailure()) {
      return MaceStatus::MACE_OUT_OF_RESOURCES;
    }

    MACE_RETURN_IF_ERROR(Memalign(result, kMaceAlignment, nbytes));
    // TODO(heliangliang) This should be avoided sometimes
    memset(*result, 0, nbytes);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus NewImage(const std::vector<size_t> &shape,
                      const DataType dt,
                      void **result) override {
    MACE_UNUSED(shape);
    MACE_UNUSED(dt);
    MACE_UNUSED(result);
    LOG(FATAL) << "Allocate CPU image";
    return MaceStatus::MACE_SUCCESS;
  }

  void Delete(void *data) override {
    MACE_CHECK_NOTNULL(data);
    VLOG(3) << "Free CPU buffer";
    free(data);
  }
  
  void DeleteImage(void *data) override {
    LOG(FATAL) << "Free CPU image";
    free(data);
  };

  void DeleteAll() override {}
  
  void Read(void *buffer, void *dst,
            size_t offset, size_t nbytes) const override {
    MACE_UNUSED(buffer);
    MACE_UNUSED(dst);
    MACE_UNUSED(offset);
    MACE_UNUSED(nbytes);
    LOG(FATAL) << "Read CPU buffer";
  }
  
  void *Map(void *buffer, size_t offset, size_t nbytes,
            const AllocatorMapType map_type,
            const BlockFlag block_flag,
            StatsFuture *future,
            utils::SimpleCountDownLatch *count_down_latch) override {
    MACE_UNUSED(nbytes);
    MACE_UNUSED(map_type);
    MACE_UNUSED(block_flag);
    MACE_UNUSED(future);
    MACE_UNUSED(count_down_latch);
    return reinterpret_cast<char*>(buffer) + offset;
  }
  
  void *MapImage(void *buffer,
                 const std::vector<size_t> &image_shape,
                 std::vector<size_t> *mapped_image_pitch,
                 AllocatorMapType map_type, BlockFlag block_flag,
                 StatsFuture *future,
                 utils::SimpleCountDownLatch *count_down_latch) const override {
    MACE_UNUSED(image_shape);
    MACE_UNUSED(mapped_image_pitch);
    MACE_UNUSED(map_type);
    MACE_UNUSED(block_flag);
    MACE_UNUSED(future);
    MACE_UNUSED(count_down_latch);
    return buffer;
  }
  
  void Unmap(void *buffer, void *mapper_ptr,
             StatsFuture *future = nullptr) override {
    MACE_UNUSED(buffer);
    MACE_UNUSED(mapper_ptr);
    MACE_UNUSED(future);
  }
  
  bool OnHost() const override { return true; }

#ifdef MACE_ENABLE_RPCMEM
  Rpcmem *rpcmem() override {
    LOG(FATAL) << "Get CPU rpcmem";
    return nullptr;
  }
#endif  // MACE_ENABLE_RPCMEM
};

// Global CPU allocator used for CPU/GPU/DSP
Allocator *GetCPUAllocator();

}  // namespace mace

#endif  // MACE_CORE_ALLOCATOR_H_
