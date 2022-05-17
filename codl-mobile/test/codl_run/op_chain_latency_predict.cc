
#include "mace/ops/op_latency_predict.h"
#include "mace/ops/conv_2d_latency_predict.h"
#include "mace/ops/pooling_latency_predict.h"
#include "mace/ops/fully_connected_latency_predict.h"
#include "mace/ops/matmul_latency_predict.h"
#include "mace/utils/thread_pool.h"
#include "test/codl_run/op_chain_latency_predict.h"
#include "test/codl_run/conv2d_test_task.h"
#include "test/codl_run/pooling_test_task.h"
#include "test/codl_run/fully_connected_test_task.h"

namespace mace {

MaceStatus OpChainLatencyPredictor::Init(LPInitContext *context) {
  LOG(INFO) << "Initialize predictors";

  backend_ = context->backend();

  if (backend_ == CONCURRENCY_AWARE_BACKEND) {
    dt_in_predictor_.reset(new ops::DataSharingLatencyPredictor());
    map_in_predictor_.reset(new ops::DataSharingLatencyPredictor());
    map_out_predictor_.reset(new ops::DataSharingLatencyPredictor());
    sync_predictor_.reset(new ops::SyncLatencyPredictor());
    dt_out_predictor_.reset(new ops::DataSharingLatencyPredictor());
    dt_in_predictor_->Init(context);
    map_in_predictor_->Init(context);
    map_out_predictor_->Init(context);
    sync_predictor_->Init(context);
    dt_out_predictor_->Init(context);
  }

  const std::vector<CodlOpType> op_types = {
      CODL_OP_TYPE_CONV2D,
      CODL_OP_TYPE_POOLING,
      CODL_OP_TYPE_FULLY_CONNECTED,
      //CODL_OP_TYPE_MATMUL
  };

  for (auto &op_type : op_types) {
    LatencyPredictor *cpu_predictor = nullptr, *gpu_predictor = nullptr;
    switch (op_type) {
      case CODL_OP_TYPE_CONV2D:
        if (backend_ == CONCURRENCY_AWARE_BACKEND) {
          cpu_predictor = new ops::Conv2dCpuLatencyPredictor();
          gpu_predictor = new ops::Conv2dGpuLatencyPredictor();
        } else if (backend_ == FLOPS_BACKEND) {
          cpu_predictor = new ops::Conv2dFLOPsLatencyPredictor();
          gpu_predictor = new ops::Conv2dFLOPsLatencyPredictor();
        }
        break;
      case CODL_OP_TYPE_POOLING:
        if (backend_ == CONCURRENCY_AWARE_BACKEND) {
          cpu_predictor = new ops::PoolingCpuLatencyPredictor();
          gpu_predictor = new ops::PoolingGpuLatencyPredictor();
        } else if (backend_ == FLOPS_BACKEND) {
          cpu_predictor = new ops::PoolingFLOPsLatencyPredictor();
          gpu_predictor = new ops::PoolingFLOPsLatencyPredictor();
        }
        break;
      case CODL_OP_TYPE_FULLY_CONNECTED:
        if (backend_ == CONCURRENCY_AWARE_BACKEND) {
          cpu_predictor = new ops::FullyConnectedCpuLatencyPredictor();
          gpu_predictor = new ops::FullyConnectedGpuLatencyPredictor();
        } else if (backend_ == FLOPS_BACKEND) {
          cpu_predictor = new ops::FullyConnectedFLOPsLatencyPredictor();
          gpu_predictor = new ops::FullyConnectedFLOPsLatencyPredictor();
        }
        break;
      case CODL_OP_TYPE_MATMUL:
        cpu_predictor = new ops::MatMulCpuLatencyPredictor();
        gpu_predictor = new ops::MatMulGpuLatencyPredictor();
        break;
      default:
        MACE_NOT_IMPLEMENTED;
    }
    cpu_predictors_[op_type] = std::shared_ptr<LatencyPredictor>(cpu_predictor);
    gpu_predictors_[op_type] = std::shared_ptr<LatencyPredictor>(gpu_predictor);
    cpu_predictors_[op_type]->Init(context);
    gpu_predictors_[op_type]->Init(context);
  }

  LOG(INFO) << "Initialize predictors success";

  return MaceStatus::MACE_SUCCESS;
}

double OpChainLatencyPredictor::Predict(
    OpContext *context,
    const std::vector<CodlOpType> &op_types,
    const std::vector<std::vector<double>> &inputs,
    const std::vector<ops::OpPartPlan *> part_plans,
    const int debug_level) {
  double dt_lat = 0;
  double map_lat = 0;
  double sync_lat = 0;
  double compute_lat = 0;
  for (size_t i = 0; i < inputs.size(); i ++) {
    const CodlOpType op_type = op_types[i];
    const std::vector<double> &layer_inputs = inputs[i];
    std::vector<double> cpu_inputs = layer_inputs;
    std::vector<double> gpu_inputs = layer_inputs;
    std::vector<double> cpu_outputs = layer_inputs;
    std::vector<double> gpu_outputs = layer_inputs;
#if 0
    std::shared_ptr<ops::OpPartPlan> op_part_plan;
#endif
    ops::OpPartPlan *op_part_plan = part_plans[i];
    if (op_type == CODL_OP_TYPE_CONV2D || op_type == CODL_OP_TYPE_POOLING) {
#if 0
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
      const Padding padding_type = static_cast<Padding>(layer_inputs[10]);
      const PartitionDim part_dim = static_cast<PartitionDim>(layer_inputs[11]);
      const float part_ratio = layer_inputs[12];

      const std::vector<index_t> input_shape = {1, ih, iw, ic};
      const std::vector<index_t> filter_shape = {oc, ic, kh, kw};
      const std::vector<int> strides = {sh, sw};
      const std::vector<int> dilations = {dh, dw};
      const std::vector<int> paddings;
#endif

      cpu_inputs.pop_back(); cpu_inputs.pop_back();
      gpu_inputs.pop_back(); gpu_inputs.pop_back();

#if 0
      ops::ConvPool2dPartPlan *conv_pool_part_plan
          = new ops::ConvPool2dPartPlan(part_dim,
                                        part_ratio,
                                        DataFormat::NHWC);
      conv_pool_part_plan->Make(input_shape, filter_shape,
                                strides, dilations,
                                padding_type, paddings);
      conv_pool_part_plan->Show();
      op_part_plan.reset(conv_pool_part_plan);
#endif
    } else if (op_type == CODL_OP_TYPE_FULLY_CONNECTED) {
#if 0
      const index_t ih = static_cast<index_t>(layer_inputs[0]);
      const index_t iw = static_cast<index_t>(layer_inputs[1]);
      const index_t ic = static_cast<index_t>(layer_inputs[2]);
      const index_t oc = static_cast<index_t>(layer_inputs[3]);
      const index_t kh = static_cast<index_t>(layer_inputs[4]);
      const index_t kw = static_cast<index_t>(layer_inputs[5]);
      const float part_ratio = layer_inputs[6];

      const std::vector<index_t> input_shape = {1, ih, iw, ic};
      const std::vector<index_t> weight_shape = {oc, ic, kh, kw};
#endif

      cpu_inputs.pop_back();
      gpu_inputs.pop_back();

#if 0
      ops::FullyConnectedPartPlan *fc_part_plan
          = new ops::FullyConnectedPartPlan(DIM_OUTPUT_CHANNEL,
                                            part_ratio,
                                            DataFormat::NHWC);
      fc_part_plan->Make(input_shape, weight_shape);
      fc_part_plan->Show();
      op_part_plan.reset(fc_part_plan);
#endif
    } else {
      LOG(ERROR) << "Unsupported OP type";
      MACE_NOT_IMPLEMENTED;
    }

    const std::vector<index_t> cpu_input_part_shape
        = op_part_plan->cpu_input_part_shape();
    const std::vector<index_t> gpu_input_part_shape
        = op_part_plan->gpu_input_part_shape();
    const std::vector<index_t> cpu_output_part_shape
        = op_part_plan->cpu_output_part_shape();
    const std::vector<index_t> gpu_output_part_shape
        = op_part_plan->gpu_output_part_shape();
    const int kHeightDim = 0;
    const int kWidthDim = 1;
    const int kOutputChannelDim = 3;
    // Update input shape.
    cpu_inputs[kHeightDim] = cpu_input_part_shape[H_NCHW];
    gpu_inputs[kHeightDim] = gpu_input_part_shape[H_NHWC];
    cpu_inputs[kWidthDim] = cpu_input_part_shape[W_NCHW];
    gpu_inputs[kWidthDim] = gpu_input_part_shape[W_NHWC];
    cpu_inputs[kOutputChannelDim] = cpu_output_part_shape[C_NCHW];
    gpu_inputs[kOutputChannelDim] = gpu_output_part_shape[C_NHWC];
    // Update output shape.
    cpu_outputs[kHeightDim] = cpu_output_part_shape[H_NCHW];
    gpu_outputs[kHeightDim] = gpu_output_part_shape[H_NHWC];
    cpu_outputs[kWidthDim] = cpu_output_part_shape[W_NCHW];
    gpu_outputs[kWidthDim] = gpu_output_part_shape[W_NHWC];
    cpu_outputs[kOutputChannelDim] = cpu_output_part_shape[C_NCHW];
    gpu_outputs[kOutputChannelDim] = gpu_output_part_shape[C_NHWC];
    
    dt_lat = 0; map_lat = 0; sync_lat = 0;
    if (backend_ == CONCURRENCY_AWARE_BACKEND) {
      if (i == 0) {
        dt_lat += dt_in_predictor_->Predict(context, cpu_inputs);
        map_lat += map_in_predictor_->Predict(context, cpu_inputs);
        map_lat += map_out_predictor_->Predict(context, cpu_outputs);
        sync_lat += sync_predictor_->Predict(context, cpu_inputs);
      } else if (i == inputs.size() - 1) {
        dt_lat += dt_out_predictor_->Predict(context, cpu_outputs);
      }
    }

    if (debug_level >= 2) {
      LOG(INFO) << "cpu_inputs " << VectorToString<double>(cpu_inputs)
                << ", gpu_inputs " << VectorToString<double>(gpu_inputs);
    }

    double cpu_compute_lat = cpu_predictors_[op_type]->Predict(context, cpu_inputs);
    double gpu_compute_lat = gpu_predictors_[op_type]->Predict(context, gpu_inputs);

    if (debug_level >= 1) {
      LOG(INFO) << "i " << i << ", op_type " << CodlOpTypeToString(op_type)
                << ", dt " << dt_lat << " ms"
                << ", map " << map_lat << " ms"
                << ", sync " << sync_lat << " ms"
                << ", cpu " << cpu_compute_lat << " ms"
                << ", gpu " << gpu_compute_lat << " ms";
    }

    //utils::ThreadPool::Sleep(500);

    compute_lat += fmax(cpu_compute_lat, gpu_compute_lat);
  }

  return dt_lat + map_lat + sync_lat + compute_lat;
}

int OpChainLatencyPredictor::Predict(
    std::vector<CodlOpTaskChain> &op_chains,
    const int debug_level,
    double *lat_ptr) {
  MACE_CHECK(lat_ptr != nullptr);

  double lat = 0;
  for (auto &chain : op_chains) {
    if (!chain.is_ready()) {
      chain.Prepare();
    }

    std::vector<CodlOpType> op_types;
    std::vector<std::vector<double>> inputs;
    std::vector<ops::OpPartPlan *> part_plans;
    for (size_t i = 0; i < chain.size(); i ++) {
      std::vector<double> layer_inputs;
      const CodlTestTaskType task_type = chain.GetTask(i)->type();
      part_plans.push_back(chain.GetTask(i)->part_plan());
      if (task_type == CONV2D_CPU_GPU_TEST_TASK) {
        op_types.push_back(CODL_OP_TYPE_CONV2D);
        CodlConv2dCpuGpuTestTask *task
            = reinterpret_cast<CodlConv2dCpuGpuTestTask *>(chain.GetTask(i).get());

        Tensor *input = task->input();
        Tensor *filter = task->filter();
        const std::vector<int> strides = task->strides();
        const std::vector<int> dilations = task->dilations();
        const Padding padding_type = task->padding_type();
        const PartitionDim part_dim = task->part_plan()->dimension();
        const float part_ratio = task->part_plan()->ratio();

        layer_inputs.push_back(static_cast<double>(input->dim(1)));
        layer_inputs.push_back(static_cast<double>(input->dim(2)));
        layer_inputs.push_back(static_cast<double>(input->dim(3)));
        layer_inputs.push_back(static_cast<double>(filter->dim(0)));
        layer_inputs.push_back(static_cast<double>(filter->dim(2)));
        layer_inputs.push_back(static_cast<double>(filter->dim(3)));
        layer_inputs.push_back(static_cast<double>(strides[0]));
        layer_inputs.push_back(static_cast<double>(strides[1]));
        layer_inputs.push_back(static_cast<double>(dilations[0]));
        layer_inputs.push_back(static_cast<double>(dilations[1]));
        layer_inputs.push_back(static_cast<double>(padding_type));
        layer_inputs.push_back(static_cast<double>(part_dim));
        layer_inputs.push_back(static_cast<double>(part_ratio));
      } else if (task_type == POOLING_CPU_GPU_TEST_TASK) {
        op_types.push_back(CODL_OP_TYPE_POOLING);
        CodlPoolingCpuGpuTestTask *task
            = reinterpret_cast<CodlPoolingCpuGpuTestTask *>(chain.GetTask(i).get());

        Tensor *input = task->input();
        const std::vector<int> kernels = task->kernels();
        const std::vector<int> strides = task->strides();
        const std::vector<int> dilations = task->dilations();
        const Padding padding_type = task->padding_type();
        const PartitionDim part_dim = task->part_plan()->dimension();
        const float part_ratio = task->part_plan()->ratio();

        layer_inputs.push_back(static_cast<double>(input->dim(1)));
        layer_inputs.push_back(static_cast<double>(input->dim(2)));
        layer_inputs.push_back(static_cast<double>(input->dim(3)));
        layer_inputs.push_back(static_cast<double>(input->dim(3)));
        layer_inputs.push_back(static_cast<double>(kernels[0]));
        layer_inputs.push_back(static_cast<double>(kernels[1]));
        layer_inputs.push_back(static_cast<double>(strides[0]));
        layer_inputs.push_back(static_cast<double>(strides[1]));
        layer_inputs.push_back(static_cast<double>(dilations[0]));
        layer_inputs.push_back(static_cast<double>(dilations[1]));
        layer_inputs.push_back(static_cast<double>(padding_type));
        layer_inputs.push_back(static_cast<double>(part_dim));
        layer_inputs.push_back(static_cast<double>(part_ratio));
      } else if (task_type == FC_CPU_GPU_TEST_TASK) {
        op_types.push_back(CODL_OP_TYPE_FULLY_CONNECTED);
        CodlFullyConnectedCpuGpuTestTask *task
            = reinterpret_cast<CodlFullyConnectedCpuGpuTestTask *>(chain.GetTask(i).get());

        Tensor *input = task->input();
        Tensor *weight = task->weight();
        const float part_ratio = task->part_plan()->ratio();
        
        layer_inputs.push_back(static_cast<double>(input->dim(1)));
        layer_inputs.push_back(static_cast<double>(input->dim(2)));
        layer_inputs.push_back(static_cast<double>(input->dim(3)));
        layer_inputs.push_back(static_cast<double>(weight->dim(0)));
        layer_inputs.push_back(static_cast<double>(weight->dim(2)));
        layer_inputs.push_back(static_cast<double>(weight->dim(3)));
        layer_inputs.push_back(static_cast<double>(part_ratio));
      } else {
        MACE_NOT_IMPLEMENTED;
      }
      
      if (debug_level >= 2) {
        LOG(INFO) << "op_type " << CodlOpTypeToString(op_types[op_types.size() - 1])
                  << ", layer_inputs " << VectorToString<double>(layer_inputs);
      }

      inputs.push_back(layer_inputs);
    }

    lat += Predict(chain.GetTask(0)->op_context(),
                   op_types,
                   inputs,
                   part_plans,
                   debug_level);

    op_types.clear();
    inputs.clear();
    part_plans.clear();
  }
  
  *lat_ptr = lat;

  return 0;
}

int OpChainLatencyPredictor::Predict(
    CodlOpTaskChain &op_chain,
    const int debug_level,
    double *lat_ptr) {
  std::vector<CodlOpTaskChain> chains = {op_chain};
  Predict(chains, debug_level, lat_ptr);
  return 0;
}

}  // namespace mace
