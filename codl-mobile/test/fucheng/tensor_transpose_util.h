
#ifndef TEST_FUCHENG_TENSOR_TRANSPOSE_UTIL_H_
#define TEST_FUCHENG_TENSOR_TRANSPOSE_UTIL_H_

#include <vector>
#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/utils/thread_pool.h"
#include "test/fucheng/tensor_buffer_util.h"

#define DST_DIMS_NHWC_TO_NCHW std::vector<int>{0, 3, 1, 2}  // NHWC->NCHW
#define DST_DIMS_NCHW_TO_NHWC std::vector<int>{0, 2, 3, 1}  // NCHW->NHWC
#define DST_DIMS_IMAGE_TO_NCHW DST_DIMS_NHWC_TO_NCHW
#define DST_DIMS_IMAGE_TO_OIHW std::vector<int>{1, 0, 2, 3}
#define DST_DIMS_NCHW_TO_IMAGE DST_DIMS_NCHW_TO_NHWC
#define DST_DIMS_NHWC_TO_IMAGE std::vector<int>{3, 2, 0, 1}
#define DST_DIMS_OIHW_TO_IMAGE DST_DIMS_IMAGE_TO_OIHW

namespace mace {

class TensorTransposeUtil {
public:
  int Transpose(OpContext *op_context,
                Tensor *src, Tensor *dst,
                OdimRanges *odim_ranges_ptr = nullptr,
                const int wino_block_size = 0,
                const bool use_default_mapping = true);

private:
  std::vector<int64_t> ShapeTranspose(const std::vector<int64_t> &input_shape,
                                      const std::vector<int> &dst_dims);

  template<typename T1, typename T2>
  int TransposeImage(utils::ThreadPool *thread_pool,
                     const T1 *input,
                     const std::vector<int64_t> &input_shape,
                     const std::vector<int> &dst_dims,
                     OdimRanges &odim_ranges,
                     T2 *output);

  template<typename T1, typename T2>
  int TransposeImageToBufferConv2dFilter(
      utils::ThreadPool *thread_pool,
      const T1 *input,
      const std::vector<int64_t> &input_shape,
      const std::vector<int> &dst_dims,
      OdimRanges &odim_ranges,
      T2 *output);

  template<typename T1, typename T2>
  int TransposeBufferToImageConv2dFilter(
      utils::ThreadPool *thread_pool,
      const T1 *input,
      const std::vector<int64_t> &input_shape,
      const std::vector<int> &dst_dims,
      OdimRanges &odim_ranges,
      T2 *output);

  int TransposeImageV2(OpContext *context,
                       Tensor *input,
                       const MemoryType in_mem_type,
                       const MemoryType out_mem_type,
                       const OpenCLBufferType buffer_type,
                       const int wino_block_size,
                       const std::vector<int> &dst_dims,
                       OdimRanges &odim_ranges,
                       Tensor *output);

  int TransposeImageV3(OpContext *context,
                       const Tensor *input,
                       const MemoryType in_mem_type,
                       const MemoryType out_mem_type,
                       const OpenCLBufferType buffer_type,
                       const int wino_block_size,
                       const std::vector<int> &dst_dims,
                       OdimRanges &odim_ranges,
                       Tensor *output);

  int TransposeImageV4(OpContext *context,
                       const Tensor *input,
                       const MemoryType in_mem_type,
                       const MemoryType out_mem_type,
                       const OpenCLBufferType buffer_type,
                       const int wino_block_size,
                       OdimRanges &odim_ranges,
                       Tensor *output);
};

}  // namespace name

#endif  // TEST_FUCHENG_TENSOR_TRANSPOSE_UTIL_H_
