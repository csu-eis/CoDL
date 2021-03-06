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

#ifndef MACE_CORE_RUNTIME_OPENCL_OPENCL_ALLOCATOR_H_
#define MACE_CORE_RUNTIME_OPENCL_OPENCL_ALLOCATOR_H_

#include <memory>
#include <vector>

#include "mace/core/allocator.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {

class OpenCLAllocator : public Allocator {
 public:
  explicit OpenCLAllocator(OpenCLRuntime *opencl_runtime);

  ~OpenCLAllocator() override;

  MaceStatus New(size_t nbytes, void **result) override;

  /*
   * Use Image2D with RGBA (128-bit) format to represent the image.
   *
   * @ shape : [depth, ..., height, width ].
   */
  MaceStatus NewImage(const std::vector<size_t> &image_shape,
                      const DataType dt,
                      void **result) override;

  void Delete(void *buffer) override;

  void DeleteImage(void *buffer) override;

  void DeleteAll() override;

  void Read(void *buffer, void *dst, size_t offset, size_t nbytes) const override;

  void *Map(void *buffer, size_t offset, size_t nbytes,
            const AllocatorMapType map_type,
            const BlockFlag block_flag,
            StatsFuture *future,
            utils::SimpleCountDownLatch *count_down_latch) override;

  void *MapImage(void *buffer,
                 const std::vector<size_t> &image_shape,
                 std::vector<size_t> *mapped_image_pitch,
                 AllocatorMapType map_type, BlockFlag block_flag,
                 StatsFuture *future,
                 utils::SimpleCountDownLatch *count_down_latch) const override;

  void Unmap(void *buffer, void *mapped_ptr, StatsFuture *future) override;

  bool OnHost() const override;

#ifdef MACE_ENABLE_RPCMEM
  Rpcmem *rpcmem() override;
#endif  // MACE_ENABLE_RPCMEM

 private:
#ifdef MACE_ENABLE_RPCMEM
  void CreateQualcommBufferIONHostPtr(const size_t nbytes,
                                      cl_mem_ion_host_ptr *ion_host);
  void CreateQualcommImageIONHostPtr(const std::vector<size_t> &shape,
                                     const cl::ImageFormat &format,
                                     size_t *pitch,
                                     cl_mem_ion_host_ptr *ion_host);

  std::unordered_map<void *, void *> cl_to_host_map_;
  Rpcmem rpcmem_;
#endif  // MACE_ENABLE_RPCMEM

  OpenCLRuntime *opencl_runtime_;
  std::map<const cl::Buffer *, const size_t> alloc_buffer_list_;
  std::map<const cl::Image2D *, const std::vector<size_t>> alloc_image_list_;
};

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_ALLOCATOR_H_
