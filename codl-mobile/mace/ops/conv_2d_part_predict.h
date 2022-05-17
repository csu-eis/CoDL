
#ifndef MACE_OPS_CONV_2D_PART_PREDICT_H_
#define MACE_OPS_CONV_2D_PART_PREDICT_H_

#include <memory>
#include <vector>

#include "mace/core/part_ratio_predictor.h"
#include "mace/ops/op_latency_predict.h"
#include "mace/ops/conv_2d_latency_predict.h"

namespace mace {
namespace ops {

class Conv2dPartitionRatioPredictor : public PartitionRatioPredictor {
 public:
  Conv2dPartitionRatioPredictor() : PartitionRatioPredictor(1) {}
  
  ~Conv2dPartitionRatioPredictor() {}
  
  MaceStatus Init() override;
  
  void Predict(OpContext *context,
               std::vector<double> &inputs,
               std::vector<double> &outputs) override;

  void PredictChain(OpContext *context,
                    std::vector<std::vector<double>> &inputs,
                    std::vector<double> &outputs) override;
  
 private:
  std::shared_ptr<DataSharingLatencyPredictor> dt_predictor_;
  std::shared_ptr<SyncLatencyPredictor> map_predictor_;
  std::shared_ptr<Conv2dCpuLatencyPredictor> cpu_predictor_;
  std::shared_ptr<Conv2dGpuLatencyPredictor> gpu_predictor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CONV_2D_PART_PREDICT_H_
