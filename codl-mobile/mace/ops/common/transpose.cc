
#include "mace/ops/common/transpose.h"

namespace mace {
namespace ops {

MaceStatus TensorTranspose(utils::ThreadPool *thread_pool,
                           Tensor *src,
                           Tensor *dst,
                           OdimRanges *odim_ranges_ptr,
                           const int wino_block_size,
                           const bool use_default_mapping) {
    MACE_UNUSED(odim_ranges_ptr);
    MACE_UNUSED(wino_block_size);
    MACE_UNUSED(use_default_mapping);
    
    Tensor::MappingGuard guard_src(src,
                                   AllocatorMapType::AMT_READ_ONLY,
                                   BlockFlag::BF_FALSE);
    Tensor::MappingGuard guard_dst(dst,
                                   AllocatorMapType::AMT_WRITE_ONLY,
                                   BlockFlag::BF_FALSE);

    std::vector<int> dst_dims;
    if (src->data_format() == DataFormat::NHWC &&
        dst->data_format() == DataFormat::NCHW) {
        dst_dims = DST_DIMS_NHWC_TO_NCHW;
    } else if (src->data_format() == DataFormat::NCHW &&
               dst->data_format() == DataFormat::NHWC) {
        dst_dims = DST_DIMS_NCHW_TO_NHWC;
    } else {
        LOG(ERROR) << "Unsupported tensor transpose";
        return MaceStatus::MACE_RUNTIME_ERROR;
    }

    Tensor *input  = src;
    Tensor *output = dst;
    
    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();
    
    Transpose<float>(thread_pool, input_data, input->shape(),
                     dst_dims, output_data);

    return MaceStatus::MACE_SUCCESS;
}

}  // namespace ops
}  // namespace mace
