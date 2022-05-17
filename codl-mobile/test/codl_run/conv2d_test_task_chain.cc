
#include "test/codl_run/conv2d_test_param.h"
#include "test/codl_run/conv2d_test_task.h"
#include "test/codl_run/conv2d_test_task_chain.h"

namespace mace {

#if 0
int CodlConv2dTaskChain::Append(const index_t height,
                                const index_t width,
                                const index_t in_channel,
                                const index_t out_channel,
                                const index_t filter_height,
                                const index_t filter_width,
                                const int stride_h,
                                const int stride_w,
                                const int part_dim,
                                const float part_ratio,
                                const bool do_data_transform,
                                const bool do_compute) {
  ratio_ = part_ratio;

  Conv2dTestParam param;
  param.is_debug_on = false;
  param.do_data_transform = do_data_transform;
  param.do_compute = do_compute;
  param.cpu_affinity_policy = 1;
  param.num_threads = 4;
  param.gpu_memory_type = 0;
  param.wino_block_size = 0;
  param.part_dim = part_dim;
  param.part_ratio = part_ratio;
  param.compute_unit_hint = ComputeUnitHint::COMPUTE_UNIT_HINT_DEFAULT;
  param.input_shape = {1, height, width, in_channel};
  param.filter_shape = {out_channel, in_channel, filter_height, filter_width};
  param.strides = {stride_h, stride_w};

  std::shared_ptr<CodlConv2dCpuGpuTestTask> task(new CodlConv2dCpuGpuTestTask());
  task->Prepare(&param);

  tasks_.emplace_back(task);

  return 0;
}
#endif

}  // namespace mace
