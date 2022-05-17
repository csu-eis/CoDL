
#include "mace/ops/conv_2d_part_predict.h"
#include "mace/ops/conv_2d_part_plan.h"
#include "mace/ops/common/conv_pool_2d_util.h"

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/conv_2d.h"
#endif

namespace mace {
namespace ops {

MaceStatus Conv2dPartitionRatioPredictor::Init() {
  dt_predictor_.reset(new DataSharingLatencyPredictor());
  map_predictor_.reset(new SyncLatencyPredictor());
  cpu_predictor_.reset(new Conv2dCpuLatencyPredictor());
  gpu_predictor_.reset(new Conv2dGpuLatencyPredictor());
  LPInitContext context;
  dt_predictor_->Init(&context);
  map_predictor_->Init(&context);
  context.set_cur_idx(model_start_idx_);
  cpu_predictor_->Init(&context);
  gpu_predictor_->Init(&context);
  return MaceStatus::MACE_SUCCESS;
}

void Conv2dPartitionRatioPredictor::Predict(OpContext *context,
                                            std::vector<double> &inputs,
                                            std::vector<double> &outputs) {
  MACE_CHECK(inputs.size() == 11);

  LOG(INFO) << "Predictor inputs: " << VectorToString<double>(inputs);

  const std::vector<PartitionDim> pd_candidates
      = {DIM_INPUT_HEIGHT, DIM_OUTPUT_CHANNEL};
  const std::vector<double> pr_candidates
      = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

  const index_t ih = static_cast<index_t>(inputs[0]);
  const index_t iw = static_cast<index_t>(inputs[1]);
  const index_t ic = static_cast<index_t>(inputs[2]);
  const index_t oc = static_cast<index_t>(inputs[3]);
  const index_t kh = static_cast<index_t>(inputs[4]);
  const index_t kw = static_cast<index_t>(inputs[5]);
  const int sh = static_cast<int>(inputs[6]);
  const int sw = static_cast<int>(inputs[7]);
  const int dh = static_cast<int>(inputs[8]);
  const int dw = static_cast<int>(inputs[9]);

  const std::vector<index_t> input_shape = {1, ih, iw, ic};
  const std::vector<index_t> filter_shape = {oc, ic, kh, kw};
  const std::vector<int> strides = {sh, sw};
  const std::vector<int> dilations = {dh, dw};
  const Padding padding_type = static_cast<Padding>(inputs[10]);
  const std::vector<int> paddings;

  std::vector<double> cpu_inputs = inputs;
  std::vector<double> gpu_inputs = inputs;

  double min_lat = DBL_MAX;
  PartitionDim opt_pd = DIM_INPUT_HEIGHT;
  double opt_pr = 1.0;
  for (auto pd_iter = pd_candidates.begin();
      pd_iter != pd_candidates.end(); ++pd_iter) {
    const PartitionDim pd = *pd_iter;
    for (auto pr_iter = pr_candidates.begin();
        pr_iter != pr_candidates.end(); ++pr_iter) {
      const double pr = *pr_iter;

      ConvPool2dPartPlan part_plan(pd, pr, DataFormat::NHWC);
      part_plan.Make(input_shape,
                     filter_shape,
                     strides,
                     dilations,
                     padding_type,
                     paddings);
      //part_plan.Show();
      const std::vector<index_t> cpu_input_part_shape
          = part_plan.cpu_input_part_shape();
      const std::vector<index_t> cpu_output_part_shape
          = part_plan.cpu_output_part_shape();
      const std::vector<index_t> gpu_input_part_shape
          = part_plan.gpu_input_part_shape();
      const std::vector<index_t> gpu_output_part_shape
          = part_plan.gpu_output_part_shape();
      cpu_inputs[0] = cpu_input_part_shape[1];
      cpu_inputs[3] = cpu_output_part_shape[3];
      gpu_inputs[0] = gpu_input_part_shape[1];
      gpu_inputs[3] = gpu_output_part_shape[3];

      const double lat_dt = dt_predictor_->Predict(context, cpu_inputs);
      const double lat_map = map_predictor_->Predict(context, cpu_inputs);
      const double lat_cpu = cpu_predictor_->Predict(context, cpu_inputs);
      const double lat_gpu = gpu_predictor_->Predict(context, gpu_inputs);

      const double lat = lat_dt + lat_map + fmax(lat_cpu, lat_gpu);
      if (lat < min_lat) {
        min_lat = lat;
        opt_pd = pd;
        opt_pr = pr;
      }

      std::vector<double> results;
      results.push_back(static_cast<double>(pd));
      results.push_back(pr);
      results.push_back(lat);
      results.push_back(min_lat);
      results.push_back(lat_dt);
      results.push_back(lat_map);
      results.push_back(lat_cpu);
      results.push_back(lat_gpu);
      LOG(INFO) << "Predicted results: " << VectorToString<double>(results);
    }
  }

  if (outputs.size() != 2) {
    outputs = std::vector<double>(2);
  }
  outputs[0] = static_cast<double>(opt_pd);
  outputs[1] = static_cast<double>(opt_pr);
}

void Conv2dPartitionRatioPredictor::PredictChain(
    OpContext *context,
    std::vector<std::vector<double>> &inputs,
    std::vector<double> &outputs) {

  const std::vector<PartitionDim> pd_candidates
      = {DIM_INPUT_HEIGHT};
  const std::vector<double> pr_candidates
      = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

  double min_lat = DBL_MAX;
  PartitionDim opt_pd = DIM_INPUT_HEIGHT;
  double opt_pr = 1.0;

  for (auto pd_iter = pd_candidates.begin();
      pd_iter != pd_candidates.end(); ++pd_iter) {
    const PartitionDim pd = *pd_iter;
    for (auto pr_iter = pr_candidates.begin();
        pr_iter != pr_candidates.end(); ++pr_iter) {
      const double pr = *pr_iter;
      double total_lat = 0;
      for (size_t i = 0; i < inputs.size(); i ++) {
        auto &layer_inputs = inputs[i];
        MACE_CHECK(layer_inputs.size() == 11);

        LOG(INFO) << "Predictor inputs: " << VectorToString<double>(layer_inputs);

        const index_t ih = static_cast<index_t>(layer_inputs[0]);
        const index_t iw = static_cast<index_t>(layer_inputs[1]);
        const index_t ic = static_cast<index_t>(layer_inputs[2]);
        const index_t oc = static_cast<index_t>(layer_inputs[3]);
        const index_t kh = static_cast<index_t>(layer_inputs[4]);
        const index_t kw = static_cast<index_t>(layer_inputs[5]);
        const int sh = static_cast<int>(layer_inputs[6]);
        const int sw = static_cast<int>(layer_inputs[7]);
        const int dh = static_cast<int>(layer_inputs[8]);
        const int dw = static_cast<int>(layer_inputs[9]);

        const std::vector<index_t> input_shape = {1, ih, iw, ic};
        const std::vector<index_t> filter_shape = {oc, ic, kh, kw};
        const std::vector<int> strides = {sh, sw};
        const std::vector<int> dilations = {dh, dw};
        const Padding padding_type = static_cast<Padding>(layer_inputs[10]);
        const std::vector<int> paddings;

        std::vector<double> cpu_inputs = layer_inputs;
        std::vector<double> gpu_inputs = layer_inputs;

        ConvPool2dPartPlan part_plan(pd, pr, DataFormat::NHWC);
        part_plan.Make(input_shape,
                       filter_shape,
                       strides,
                       dilations,
                       padding_type,
                       paddings);
        //part_plan.Show();
        const std::vector<index_t> cpu_input_part_shape
            = part_plan.cpu_input_part_shape();
        const std::vector<index_t> cpu_output_part_shape
            = part_plan.cpu_output_part_shape();
        const std::vector<index_t> gpu_input_part_shape
            = part_plan.gpu_input_part_shape();
        const std::vector<index_t> gpu_output_part_shape
            = part_plan.gpu_output_part_shape();
        cpu_inputs[0] = cpu_input_part_shape[1];
        cpu_inputs[3] = cpu_output_part_shape[3];
        gpu_inputs[0] = gpu_input_part_shape[1];
        gpu_inputs[3] = gpu_output_part_shape[3];

        double lat_dt = 0;
        double lat_map = 0;
        if (i == 0) {
          lat_dt = dt_predictor_->Predict(context, cpu_inputs);
          lat_map = map_predictor_->Predict(context, cpu_inputs);
        }

        const double lat_cpu = cpu_predictor_->Predict(context, cpu_inputs);
        const double lat_gpu = gpu_predictor_->Predict(context, gpu_inputs);
        total_lat += lat_dt + lat_map + fmax(lat_cpu, lat_gpu);

        std::vector<double> results;
        results.push_back(static_cast<double>(pd));
        results.push_back(pr);
        results.push_back(total_lat);
        results.push_back(min_lat);
        results.push_back(lat_dt);
        results.push_back(lat_map);
        results.push_back(lat_cpu);
        results.push_back(lat_gpu);
        LOG(INFO) << "Predicted results: " << VectorToString<double>(results);
      }

      if (total_lat < min_lat) {
        min_lat = total_lat;
        opt_pd = pd;
        opt_pr = pr;
      }
    }
  }

  if (outputs.size() != 2) {
    outputs = std::vector<double>(2);
  }
  outputs[0] = static_cast<double>(opt_pd);
  outputs[1] = static_cast<double>(opt_pr);
}

void RegisterConv2dPartRatioPredictor(
    PartRatioPredictorRegistryBase *pr_registry) {
  pr_registry->Register(
      "Conv2D",
      PartRatioPredictorRegistryBase::DefaultCreator<Conv2dPartitionRatioPredictor>);
}

}  // namespace ops
}  // namespace mace
