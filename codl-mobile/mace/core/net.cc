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

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>
#include <utility>

#include "mace/core/future.h"
#include "mace/core/memory_optimizer.h"
#include "mace/core/net.h"
#include "mace/core/op_context.h"
#include "mace/public/mace.h"
#include "mace/port/env.h"
#include "mace/utils/conf_util.h"
#include "mace/utils/logging.h"
#include "mace/utils/macros.h"
#include "mace/utils/math.h"
#include "mace/utils/memory.h"
#include "mace/utils/timer.h"
#include "mace/utils/op_delay_tool.h"
#include "mace/utils/statistics.h"
#include "mace/utils/soc_devfreq.h"

namespace mace {

bool IsTuneOpWorkloadEnabled() {
  const char *op_workload_tuning = getenv("MACE_OP_WORKLOAD_TUNING");
  if (op_workload_tuning != nullptr
      && strlen(op_workload_tuning) == 1 && op_workload_tuning[0] == '1') {
    return true;
  } else {
    return false;
  }
}

std::string TimeMillisToString(const int64_t time_millis) {
  const int64_t time_sec = time_millis / 1000000;
  const int64_t min = time_sec / 60;
  const int64_t sec = time_sec % 60;
  std::stringstream stream;
  if (min > 0) {
    stream << min << "min";
  }
  if (sec > 0) {
    stream << sec << "s";
  }
  return stream.str();
}

SerialNet::SerialNet(const OpRegistryBase *op_registry,
                     const NetDef *net_def,
                     Workspace *ws,
                     Device *target_device,
                     MemoryOptimizer *mem_optimizer)
    : NetBase(),
      ws_(ws),
      target_device_(target_device),
      cpu_device_(
          make_unique<CPUDevice>(
              target_device->cpu_runtime()->num_threads(),
              target_device->cpu_runtime()->policy(),
              &target_device->cpu_runtime()->thread_pool())) {
  MACE_LATENCY_LOGGER(1, "Constructing SerialNet");

#ifdef MACE_ENABLE_OPENCL
  // used for memory optimization
  std::unordered_map<std::string, MemoryType> output_mem_map;
#endif  // MACE_ENABLE_OPENCL

  OpConstructContext construct_context(ws_);
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    std::shared_ptr<OperatorDef> op_def(new OperatorDef(net_def->op(idx)));
    // Create operation
    auto op_device_type = static_cast<DeviceType>(op_def->device_type());
    if (op_device_type == target_device_->device_type()) {
      construct_context.set_device(target_device_);
    } else if (op_device_type == DeviceType::CPU) {
      construct_context.set_device(cpu_device_.get());
    } else {
      LOG(FATAL) << "Encounter unexpected error: "
                 << op_device_type << " vs " << target_device_->device_type();
    }
    construct_context.set_operator_def(op_def);

    auto op = op_registry->CreateOperation(&construct_context,
                                           op_device_type);
    operators_.emplace_back(std::move(op));
    // where to do graph reference count.
    mem_optimizer->UpdateTensorRef(op_def.get());

#ifdef MACE_ENABLE_OPENCL
    if (target_device_->device_type() == DeviceType::GPU) {
      // update the map : output_tensor -> MemoryType
      MemoryType out_mem_type =
          static_cast<MemoryType>(
              ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                  net_def->op(idx), OutputMemoryTypeTagName(),
                  static_cast<int>(MemoryType::CPU_BUFFER)));
      for (int out_idx = 0; out_idx < op_def->output_size(); ++out_idx) {
        output_mem_map[op_def->output(out_idx)] = out_mem_type;
      }
    }
#endif  // MACE_ENABLE_OPENCL
  }
  // Update output tensor reference
  for (auto &output_info : net_def->output_info()) {
    mem_optimizer->UpdateTensorRef(output_info.name());
  }

#if 0
  // Do memory optimization
  for (auto &op : operators_) {
    VLOG(2) << "Operator " << op->debug_def().name() << "<" << op->device_type()
            << ", " << op->debug_def().type() << ">";
#ifdef MACE_ENABLE_OPENCL
    mem_optimizer->Optimize(op->operator_def().get(), &output_mem_map);
#else
    mem_optimizer->Optimize(op->operator_def().get());
#endif  // MACE_ENABLE_OPENCL
  }
  VLOG(1) << mem_optimizer->DebugInfo();
  // NOTE(fucheng): See information of memory optimizer.
  //LOG(INFO) << mem_optimizer->DebugInfo();
#endif

  const char *op_chain_env = getenv("CODL_OP_CHAIN");
  const bool op_chain_enabled = (op_chain_env != nullptr &&
                                 strlen(op_chain_env) == 1 &&
                                 op_chain_env[0] == '1');
  if (op_chain_enabled) {
    // NOTE(fucheng): Build op chain.
    BuildOpChain();
  }

  is_first_time_run_ = true;

  const char *run_conv2d_only = getenv("MACE_RUN_CONV2D_ONLY");
  run_conv2d_only_enabled_ = (run_conv2d_only != nullptr &&
                              strlen(run_conv2d_only) == 1 &&
                              run_conv2d_only[0] == '1');
}

SerialNet::~SerialNet() {}

MaceStatus SerialNet::Init() {
  MACE_LATENCY_LOGGER(1, "Initializing SerialNet");
  OpInitContext init_context(ws_);
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    auto &op = *iter;
    DeviceType device_type = op->device_type();
    if (device_type == target_device_->device_type()) {
      init_context.set_device(target_device_);
    } else {
      init_context.set_device(cpu_device_.get());
    }
    // Initialize the operation
    MACE_RETURN_IF_ERROR(op->Init(&init_context));
  }
  // Show operators count.
  LOG(INFO) << "Operations count: " << operators_.size();
  
  return MaceStatus::MACE_SUCCESS;
}

//===============================================
//#define CODL_ENABLE_GPU_SYNC
#define CODL_ENABLE_OP_PARTITION_CONFIGER
//#define CODL_ENABLE_CONV_CHAIN
//#define CODL_ENABLE_OP_PARTITION_TUNING
//#define CODL_ENABLE_DEBUG_INFO_TUNING
//#define CODL_ENABLE_NO_FC
//#define CODL_ENABLE_SHOW_TOTAL_CONV2D_DELAY
//===============================================

MaceStatus SerialNet::GetPartRatioPredictInputData(float **in_data_ptr) {
  const bool is_tz_enabled = true;
  const bool is_devfreq_enabled = true;
  float *in_data = *in_data_ptr;

  for (auto iter = operators_.begin();
      iter != operators_.end(); ++iter) {
    auto &op = *iter;
    if (op->debug_def().type().compare("Conv2D") == 0) {
      LOG(INFO) << "Get PR input data in a layer";
      // Get information from op debug definition
      std::vector<index_t> input_shape = op->Input(0)->shape();
      std::vector<index_t> filter_shape = op->Input(1)->shape();
      std::vector<index_t> output_shape = op->Output(0)->shape();
      std::vector<int> strides = op->GetRepeatedArgs<int>("strides");
      //int padding_type = op->GetOptionalArg<int>("padding", -1);
      //std::vector<int> paddings = op->GetRepeatedArgs<int>("padding_values");
      std::vector<int> dilations = op->GetRepeatedArgs<int>("dilations");

      MACE_CHECK(input_shape.size() > 0);
      MACE_CHECK(filter_shape.size() > 0);
      MACE_CHECK(output_shape.size() > 0);
      MACE_CHECK(strides.size() > 0);
      if (dilations.size() == 0) {
        dilations = {0, 0};
      }
      
      // Get thermal zone information.
      std::vector<int> cpu_cs; int gpu_cs;
      if (is_tz_enabled) {
        const std::unique_ptr<utils::SocThermalZone> tz(utils::SocThermalZone::CreateLocal());
        cpu_cs = tz->cpu_cur_cooling_status();
        gpu_cs = tz->gpu_cur_cooling_status();
      } else {
        cpu_cs = std::vector<int>(2);
        gpu_cs = 0;
      }

      // Get device frequency information.
      std::vector<int> cpu_freq; int gpu_freq;
      if (is_devfreq_enabled) {
        const std::unique_ptr<utils::SocDevfreq> devfreq(utils::SocDevfreq::CreateLocal());
        //cpu_freq = devfreq->cpu_freq(cpu_cs);
        //gpu_freq = devfreq->gpu_freq(gpu_cs);
        cpu_freq = devfreq->cpu_cur_freq();
        gpu_freq = devfreq->gpu_cur_freq();
      } else {
        cpu_freq = std::vector<int>(2);
        gpu_freq = 0;
      }

      in_data[0] = (float) input_shape[0];
      in_data[1] = (float) input_shape[1];
      in_data[2] = (float) input_shape[3];
      in_data[3] = (float) filter_shape[0];
      in_data[4] = (float) filter_shape[2];
      in_data[5] = (float) strides[0];
      in_data[6] = (float) dilations[0];
      in_data[7] = (float) cpu_cs[0];
      in_data[8] = (float) cpu_cs[1];
      in_data[9] = (float) gpu_cs;
      in_data[10] = (float) cpu_freq[0];
      in_data[11] = (float) cpu_freq[1];
      in_data[12] = (float) gpu_freq;

      in_data += 13;
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

constexpr double kCapabilityThreshold = 30;

MaceStatus SerialNet::PredictPartRatio(OpContext *context) {
  utils::SocDevfreq *devfreq = utils::GetSocDevfreq();
  const double last_capability = devfreq->capability();
  const double capability = devfreq->cpu_capability() + devfreq->gpu_capability();
  if (fabs(capability - last_capability) > kCapabilityThreshold) {
    LOG(INFO) << "Devfreq capability: " << capability;
    devfreq->set_capability(capability);
    partition_configer_.Clear();

    std::unordered_map<std::string,
                       std::unique_ptr<PartitionRatioPredictor>> pr_predictor_map;

    for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
      auto &op = *iter;
      const std::string op_type = op->debug_def().type();
      if (op_type.compare("Conv2D") == 0) {
        // Initialize partition ratio predictor.
        if (pr_predictor_map.count(op_type) == 0) {
          pr_predictor_map[op_type] = pr_predictor_registry_->CreatePartRatioPredictor(op_type);
          MACE_RETURN_IF_ERROR(pr_predictor_map[op_type]->Init());
        }

        // Get shapes.
        std::vector<index_t> input_shape = op->Input(0)->shape();
        std::vector<index_t> kernel_shape = op->Input(1)->shape();
        std::vector<int> strides = op->GetRepeatedArgs<int>("strides");
        int padding_type = op->GetOptionalArg<int>("padding", -1);
        std::vector<int> dilations = op->GetRepeatedArgs<int>("dilations");

        MACE_CHECK(input_shape.size() == 4);
        MACE_CHECK(kernel_shape.size() == 4);
        MACE_CHECK(strides.size() == 2);
        if (padding_type == -1) {
          padding_type = 0;
        }
        if (dilations.size() == 0) {
          dilations = {1, 1};
        }

        // Build input vector.
        std::vector<double> inputs;
        inputs.push_back(input_shape[1]);
        inputs.push_back(input_shape[2]);
        inputs.push_back(input_shape[3]);
        inputs.push_back(kernel_shape[0]);
        inputs.push_back(kernel_shape[2]);
        inputs.push_back(kernel_shape[3]);
        inputs.push_back(strides[0]);
        inputs.push_back(strides[1]);
        inputs.push_back(dilations[0]);
        inputs.push_back(dilations[1]);
        inputs.push_back(padding_type);

        // Predict.
        std::vector<double> outputs(2);
        PartitionRatioPredictor *pr_predictor = pr_predictor_map.at(op_type).get();
        pr_predictor->Predict(context, inputs, outputs);

        // Get output from vector.
        const int dim = static_cast<int>(outputs[0]);
        const float ratio = static_cast<float>(outputs[1]);
        partition_configer_.Append(dim, ratio);

        // Show logs.
        LOG(INFO) << "Predicted dimension " << dim << " ratio " << ratio;
      }
    }

    partition_configer_.Save();
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialNet::ShowOpFeatures(const std::string &op_type) {
  LOG(INFO) << "Print operator features, operator type: " << op_type;
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    auto &op = *iter;
    if (op->debug_def().type().compare(op_type) == 0) {
      std::vector<index_t> input_shape = op->Input(0)->shape();
      std::vector<index_t> filter_shape = op->Input(1)->shape();
      std::vector<index_t> output_shape = op->Output(0)->shape();
      std::vector<int> strides = op->GetRepeatedArgs<int>("strides");
      int padding_type = op->GetOptionalArg<int>("padding", -1);
      std::vector<int> paddings = op->GetRepeatedArgs<int>("padding_values");
      std::vector<int> dilations = op->GetRepeatedArgs<int>("dilations");

      MACE_CHECK(input_shape.size() == 4);
      MACE_CHECK(filter_shape.size() == 4);
      MACE_CHECK(output_shape.size() == 4);
      MACE_CHECK(strides.size() == 2);
      enum Padding {VALID = 0, SAME = 1, FULL = 2};
      if (padding_type == -1) {
        padding_type = 0;
      }
      if (padding_type == Padding::SAME || paddings.size() != 2) {
        paddings = {0, 0};
      }
      if (dilations.size() == 0) {
        dilations = {0, 0};
      }
      MACE_CHECK(dilations.size() == 2);

      std::vector<int> out_param;
      out_param.push_back(static_cast<int>(input_shape[1]));
      out_param.push_back(static_cast<int>(input_shape[2]));
      out_param.push_back(static_cast<int>(input_shape[3]));
      out_param.push_back(static_cast<int>(filter_shape[0]));
      out_param.push_back(static_cast<int>(filter_shape[2]));
      out_param.push_back(static_cast<int>(filter_shape[3]));
      out_param.push_back(strides[0]);
      out_param.push_back(strides[1]);
      out_param.push_back(padding_type);
      out_param.push_back(paddings[0]);
      out_param.push_back(paddings[1]);
      out_param.push_back(dilations[0]);
      out_param.push_back(dilations[1]);

      LOG(INFO) << VectorToString<int>(out_param);
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialNet::BuildOpChain() {
  std::vector<std::string> target_op_types =
      {"Conv2D", "Pooling", "Deconv2D", "FullyConnected", "MatMul", "Eltwise"};

  // Build chain.
  for (auto &op_type : target_op_types) {
    for (auto &op : operators_) {
      if (!op->debug_def().type().compare(op_type)) {
        std::shared_ptr<OperatorChain> found_chain;
        for (auto &chain : op_chains_) {
          if (!chain->back()->debug_def().output(0).compare(op->debug_def().input(0))) {
            chain->push_back(op.get());
            found_chain = chain;
            break;
          }
        }
        if (!found_chain.get()) {
          op_chains_.emplace_back(new OperatorChain(op.get()));
        }
      }
    }
  }

  // Print chain info.
  int chain_idx = 0;
  for (auto &chain : op_chains_) {
    if (chain->size() > 1) {
      LOG(INFO) << "OP chain: idx " << chain_idx << ", " << chain->DebugInfo();
      chain_idx ++;
    } else if (chain->size() == 1) {
      // NOTE(fucheng): Set chain position to NONE for chain of which size is 1.
      chain->op(0)->chain_context()->set_position(OP_POSITION_NONE);
    }
  }

  if (!chain_idx) {
    LOG(INFO) << "No OP chain found";
  }

  utils::ThreadPool::Sleep(1 * 1000);

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialNet::Run(RunMetadata *run_metadata,
                          PartitionRunConfig *part_run_config,
                          MaceRuntimeStatistics *runtime_stat) {
  MACE_UNUSED(runtime_stat);
  const char *profiling = getenv("MACE_OPENCL_PROFILING");
  bool
  enable_opencl_profiling =
      profiling != nullptr && strlen(profiling) == 1 && profiling[0] == '1';

  MACE_MEMORY_LOGGING_GUARD();
  MACE_LATENCY_LOGGER(1, "Running net");
  StatsFuture future;
  OpContext context(ws_, cpu_device_.get());
  context.set_cpu_device(cpu_device_.get());
  context.set_future(&future);
  context.set_part_run_config(part_run_config);

#ifdef CODL_ENABLE_OP_PARTITION_CONFIGER
  if (part_run_config != nullptr) {
    if (part_run_config->ratio() == kRatioFromConfigFile) {
      //LOG(INFO) << "PartRunConfig:"
      //          << " do_pr_reload " << part_run_config->do_pr_configer_reload()
      //          << " ratio " << part_run_config->ratio()
      //          << " is_file_open " << partition_configer_.is_file_open();
      //if (part_run_config->do_pr_configer_reload() ||
      //    partition_configer_.is_file_open()) {
      if (part_run_config->do_pr_configer_reload()) {
        partition_configer_.Load();
      }
    }
  }
#endif  // CODL_ENABLE_OP_PARTITION_CONFIGER

  context.set_op_runtime_mode(OpRuntimeMode::RUNNING);
  context.set_partition_configer(&partition_configer_);

  uint64_t op_idx = 0;
  //size_t cpu_gpu_op_idx = 0;
  const size_t kMaxOpCount = 2048;
  for (auto iter = operators_.begin();
      iter != operators_.end() && op_idx < kMaxOpCount; ++iter, ++op_idx) {
    context.set_op_idx(op_idx);
    auto &op = *iter;
    DeviceType device_type = op->device_type();
    int dtype = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
        op->debug_def(), "T", static_cast<int>(DT_FLOAT));
    VLOG(2) << "Run operator:" << " idx " << op_idx
            << ", name " << op->debug_def().name()
            << ", type " << op->debug_def().type()
            << ", device " << device_type
            << ", dtype " << dtype;
    MACE_LATENCY_LOGGER(1, "Running operator ", op->debug_def().name(),
                       "<", device_type, ", ", op->debug_def().type(),
                       ", ",
                       ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                           op->debug_def(), "T", static_cast<int>(DT_FLOAT)),
                       ">");
    if (device_type == target_device_->device_type()) {
      context.set_device(target_device_);
    } else {
      context.set_device(cpu_device_.get());
    }

    CallStats call_stats;
    if (run_metadata == nullptr) {
      bool run_op = true;
#if 0
      LOG(INFO) << "run_conv2d_only_enabled_ " << run_conv2d_only_enabled_
               << " is_first_time_run_ " << is_first_time_run_
               << " op_type " << op->debug_def().type();
#endif
      if (run_conv2d_only_enabled_ &&
          !is_first_time_run_ &&
          op->debug_def().type().compare("Conv2D") != 0) {
        run_op = false;
      }
#if 0
      int64_t t0 = NowMicros();
#endif
      if (run_op) {
        const OpPosition op_pos = op->chain_context()->position();
        if (op_pos == OP_POSITION_NONE) {
          MACE_RETURN_IF_ERROR(op->Run(&context));
        } else {
          if (op_pos == OP_POSITION_HEAD) {
            OperatorChain *op_chain = reinterpret_cast<OperatorChain *>(
                                          op->chain_context()->chain());
            MACE_RETURN_IF_ERROR(op_chain->Run(&context));
          }
        }
      }
#if 0
      LOG(INFO) << "op_type " << op->debug_def().type()
                << ", run_op " << (NowMicros() - t0) / 1000.0 << " ms";
#endif
    } else {
      if (device_type == DeviceType::CPU
          || (device_type == DeviceType::GPU && !enable_opencl_profiling)) {
        call_stats.start_micros = NowMicros();
        MACE_RETURN_IF_ERROR(op->Run(&context));
        call_stats.end_micros = NowMicros();
      } else if (device_type == DeviceType::GPU) {
        StatsFuture future;
        context.set_future(&future);
        MACE_RETURN_IF_ERROR(op->Run(&context));
        future.wait_fn(&call_stats);
      }

      // Record run metadata.
      std::vector<int> strides;
      int padding_type = -1;
      std::vector<int> paddings;
      std::vector<int> dilations;
      std::vector<index_t> kernels;
      std::string type = op->debug_def().type();

      if (type.compare("Conv2D") == 0 ||
          type.compare("Deconv2D") == 0 ||
          type.compare("DepthwiseConv2d") == 0 ||
          type.compare("DepthwiseDeconv2d") == 0 ||
          type.compare("Pooling") == 0) {
        strides = op->GetRepeatedArgs<int>("strides");
        padding_type = op->GetOptionalArg<int>("padding", -1);
        paddings = op->GetRepeatedArgs<int>("padding_values");
        dilations = op->GetRepeatedArgs<int>("dilations");
        if (type.compare("Pooling") == 0) {
          kernels = op->GetRepeatedArgs<index_t>("kernels");
        } else {
          kernels = op->Input(1)->shape();
        }
      } else if (type.compare("MatMul") == 0) {
        bool transpose_a = op->GetOptionalArg<bool>("transpose_a", false);
        kernels = op->Input(0)->shape();
        if (transpose_a) {
          std::swap(kernels[kernels.size() - 2], kernels[kernels.size() - 1]);
        }
      } else if (type.compare("FullyConnected") == 0) {
        kernels = op->Input(1)->shape();
      }

      std::vector<std::vector<int64_t>> output_shapes;
      for (auto output : op->Outputs()) {
        output_shapes.push_back(output->shape());
      }
      OperatorStats op_stats = {op->debug_def().name(), op->debug_def().type(),
                                output_shapes,
                                {strides, padding_type, paddings, dilations,
                                 kernels}, call_stats};
      run_metadata->op_stats.emplace_back(op_stats);
    }

    VLOG(3) << "Operator " << op->debug_def().name()
            << " has shape: " << MakeString(op->Output(0)->shape());

    if (EnvConfEnabled("MACE_LOG_TENSOR_RANGE")) {
      for (int i = 0; i < op->OutputSize(); ++i) {
        if (op->debug_def().quantize_info_size() == 0) {
          int data_type = op->GetOptionalArg("T", static_cast<int>(DT_FLOAT));
          if (data_type == static_cast<int>(DT_FLOAT)) {
            float max_v = std::numeric_limits<float>::lowest();
            float min_v = std::numeric_limits<float>::max();
            Tensor::MappingGuard guard(op->Output(i));
            auto *output_data = op->Output(i)->data<float>();
            for (index_t j = 0; j < op->Output(i)->size(); ++j) {
              max_v = std::max(max_v, output_data[j]);
              min_v = std::min(min_v, output_data[j]);
            }
            LOG(INFO) << "Tensor range @@" << op->debug_def().output(i)
                      << "@@" << min_v << "," << max_v;
          }
        } else {
          const int bin_size = 2048;
          for (int ind = 0; ind < op->debug_def().quantize_info_size(); ++ind) {
            float min_v = op->debug_def().quantize_info(ind).minval();
            float max_v = op->debug_def().quantize_info(ind).maxval();
            std::vector<int> bin_distribution(bin_size, 0);
            float bin_v = (max_v - min_v) / bin_size;
            Tensor::MappingGuard guard(op->Output(i));
            auto *output_data = op->Output(i)->data<float>();
            for (index_t j = 0; j < op->Output(i)->size(); ++j) {
              int index = static_cast<int>((output_data[j] - min_v) / bin_v);
              if (index < 0)
                index = 0;
              else if (index > bin_size - 1)
                index = bin_size - 1;
              bin_distribution[index]++;
            }
            LOG(INFO) << "Tensor range @@" << op->debug_def().output(i)
                      << "@@" << min_v << "," << max_v << "@@"
                      << MakeString(bin_distribution);
          }
        }
      }
    }
  }

#ifdef CODL_ENABLE_OP_PARTITION_CONFIGER
  if (part_run_config != nullptr) {
    if (part_run_config->ratio() == kRatioFromPredictor) {
      MACE_RETURN_IF_ERROR(PredictPartRatio(&context));
      MACE_CHECK(partition_configer_.is_file_open());
    }
  }

  if (!partition_configer_.is_file_open()) {
    partition_configer_.Save();
  }
  partition_configer_.Reset();
#endif

  // Show operator features if enabled.
  std::string op_type;
  GetEnv("MACE_SHOW_OP_FEATURE_BY_OP_TYPE", &op_type);
  if (!op_type.empty()) {
    ShowOpFeatures(op_type);
  }

  if (is_first_time_run_) {
    is_first_time_run_ = false;
  }

  return MaceStatus::MACE_SUCCESS;
}
}  // namespace mace
