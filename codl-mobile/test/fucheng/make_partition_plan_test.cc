
#include <vector>
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/conv_2d_part_plan.h"
#ifdef FUCHENG_ENABLE_CONV_CHAIN
#include "mace/ops/conv_2d_chain_part_plan.h"
#endif

namespace mace {

MaceStatus MakePartitionPlanTest() {
  std::vector<index_t> input_shape{1, 14, 14, 256};
  std::vector<index_t> filter_shape{64, 256, 3, 3};
  std::vector<int> strides{1, 1};
  std::vector<int> dilations{1, 1};
  Padding padding_type = static_cast<Padding>(static_cast<int>(VALID));
  std::vector<int> paddings{0, 0};

  float part_ratio = 0.5f;

  ops::Conv2dPartPlan plan(PartitionDim::DIM_INPUT_HEIGHT,
                           part_ratio,
                           DataFormat::NCHW);
  int ret = plan.Make(input_shape, filter_shape,
                      strides, dilations,
                      padding_type, paddings);
  if (!ret) {
    LOG(ERROR) << "Make conv2d part plan failed";
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
  
  plan.Show();

#ifdef FUCHENG_ENABLE_CONV_CHAIN
  // Test of making partition plan for convolution chain
  std::vector<std::vector<index_t>> input_shapes;
  std::vector<std::vector<index_t>> filter_shapes;
  std::vector<std::vector<int>> strideses;
  std::vector<std::vector<int>> dilationses;
  std::vector<Padding> padding_types;
  std::vector<std::vector<int>> paddingses;

  input_shapes.push_back(std::vector<index_t>{1,14,14,256});
  input_shapes.push_back(std::vector<index_t>{1,14,14,64});

  filter_shapes.push_back(std::vector<index_t>{64,256,1,1});
  filter_shapes.push_back(std::vector<index_t>{64,64,3,3});

  strideses.push_back(std::vector<int>{1,1});
  strideses.push_back(std::vector<int>{1,1});

  dilationses.push_back(std::vector<int>{1,1});
  dilationses.push_back(std::vector<int>{1,1});

  padding_types.push_back(static_cast<Padding>(static_cast<int>(VALID)));
  padding_types.push_back(static_cast<Padding>(static_cast<int>(VALID)));

  paddingses.push_back(std::vector<int>{0,0});
  paddingses.push_back(std::vector<int>{0,0});

  ops::Conv2dChainPartPlan cplan(part_ratio);
  MaceStatus status = cplan.Make(input_shapes, filter_shapes,
                                 strideses, dilationses,
                                 padding_types, paddingses);
  if (status != MaceStatus::MACE_SUCCESS) {
      LOG(ERROR) << "Make conv2d chain part plan failed";
      return status;
  }

  cplan.Show();
#endif // FUCHENG_ENABLE_CONV_CHAIN

  return MaceStatus::MACE_SUCCESS;
}

} // namcespace mace
