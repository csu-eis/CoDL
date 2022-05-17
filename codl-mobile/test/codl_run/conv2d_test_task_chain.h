
#ifndef TEST_CODL_RUN_CONV2D_TEST_TASK_CHAIN_H_
#define TEST_CODL_RUN_CONV2D_TEST_TASK_CHAIN_H_

#include "test/codl_run/op_test_task_chain.h"

namespace mace {
#if 0
class CodlConv2dTaskChain : public CodlOpTaskChain {
 public:
  int Append(const index_t height,
             const index_t width,
             const index_t in_channel,
             const index_t out_channel,
             const index_t filter_height,
             const index_t filter_width,
             const int stride_h,
             const int stride_w,
             const int part_dim,
             const float part_ratio,
             const bool do_data_transform = false,
             const bool do_compute = true);

  int Append(const CodlOpChainParam &param,
             const bool do_data_transform = false,
             const bool do_compute = true) {
    const CodlConv2dChainParam *param_ptr =
        reinterpret_cast<const CodlConv2dChainParam *>(&param);
    return Append(param_ptr->height(), param_ptr->width(),
                  param_ptr->in_channel(), param_ptr->out_channel(),
                  param_ptr->filter_height(), param_ptr->filter_width(),
                  param_ptr->stride_h(), param_ptr->stride_w(),
                  param_ptr->part_dim(), param_ptr->part_ratio(),
                  do_data_transform, do_compute);
  }
};
#endif
}  // namespace mace

#endif  // TEST_CODL_RUN_CONV2D_TEST_TASK_CHAIN_H_
