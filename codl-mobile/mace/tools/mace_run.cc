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

/**
 * Usage:
 * mace_run --model=mobi_mace.pb \
 *          --input=input_node  \
 *          --output=output_node  \
 *          --input_shape=1,224,224,3   \
 *          --output_shape=1,224,224,2   \
 *          --input_file=input_data \
 *          --output_file=mace.out  \
 *          --model_data_file=model_data.data \
 *          --device=GPU
 */
#include <sys/types.h>
#include <dirent.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <climits>

#include "gflags/gflags.h"
#include "mace/public/mace.h"
#include "mace/port/env.h"
#include "mace/port/file_system.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"
#include "mace/utils/string_util.h"
#include "mace/utils/statistics.h"

#ifdef MODEL_GRAPH_FORMAT_CODE
#include "mace/codegen/engine/mace_engine_factory.h"
#endif

namespace mace {
namespace tools {

template <typename T>
static std::string VectorToString(const std::vector<T> &vec) {
  if (vec.empty()) {
    return "";
  }

  std::stringstream stream;
  stream << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    stream << vec[i];
    if (i != vec.size() - 1) {
      stream << ",";
    }
  }
  stream << "]";

  return stream.str();
}

template <typename T>
std::string DataToString(const T *data,
                         const int64_t count,
                         const int64_t max_count) {
  std::stringstream stream;
  stream << "[";
  int64_t i;
  for (i = 0; i < count && i < max_count; ++i) {
  	stream << data[i];
  	if (i < (count - 1))
  		stream << ",";
  }
  if (i == max_count && i < count)
  	stream << "...";
  stream << "]";
  
  return stream.str();
}

template <typename T>
int CalcArgmax(const T *data, const int64_t count) {
  int max_index = -1;
  T max_value = INT_MIN;

  for (int i = 0; i < count; i ++) {
    if (data[i] > max_value) {
      max_value = data[i];
      max_index = i;
    }
  }

  return max_index;
}

void ParseShape(const std::string &str, std::vector<int64_t> *shape) {
  std::string tmp = str;
  while (!tmp.empty()) {
    int dim = atoi(tmp.data());
    shape->push_back(dim);
    size_t next_offset = tmp.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
}

std::string FormatName(const std::string input) {
  std::string res = input;
  for (size_t i = 0; i < input.size(); ++i) {
    if (!isalnum(res[i])) res[i] = '_';
  }
  return res;
}

DeviceType ParseDeviceType(const std::string &device_str) {
  if (device_str.compare("CPU") == 0) {
    return DeviceType::CPU;
  } else if (device_str.compare("GPU") == 0) {
    return DeviceType::GPU;
  } else if (device_str.compare("HEXAGON") == 0) {
    return DeviceType::HEXAGON;
  } else if (device_str.compare("HTA") == 0) {
    return DeviceType::HTA;
  } else if (device_str.compare("APU") == 0) {
    return DeviceType::APU;
  } else {
    return DeviceType::CPU;
  }
}

DataFormat ParseDataFormat(const std::string &data_format_str) {
  if (data_format_str == "NHWC") {
    return DataFormat::NHWC;
  } else if (data_format_str == "NCHW") {
    return DataFormat::NCHW;
  } else if (data_format_str == "OIHW") {
    return DataFormat::OIHW;
  } else {
    return DataFormat::NONE;
  }
}

DEFINE_string(model_name,
              "",
              "model name in yaml");
DEFINE_string(input_node,
              "",
              "input nodes, separated by comma");
DEFINE_string(input_shape,
              "",
              "input shapes, separated by colon and comma");
DEFINE_string(output_node,
              "",
              "output nodes, separated by comma");
DEFINE_string(output_shape,
              "",
              "output shapes, separated by colon and comma");
DEFINE_string(input_data_format,
              "NHWC",
              "input data formats, NONE|NHWC|NCHW");
DEFINE_string(output_data_format,
              "NHWC",
              "output data formats, NONE|NHWC|NCHW");
DEFINE_string(input_file,
              "",
              "input file name | input file prefix for multiple inputs.");
DEFINE_string(output_file,
              "",
              "output file name | output file prefix for multiple outputs");
DEFINE_string(input_dir,
              "",
              "input directory name");
DEFINE_string(output_dir,
              "output",
              "output directory name");
DEFINE_string(opencl_binary_file,
              "",
              "compiled opencl binary file path");
DEFINE_string(opencl_parameter_file,
              "",
              "tuned OpenCL parameter file path");
DEFINE_string(model_data_file,
              "",
              "model data file name, used when EMBED_MODEL_DATA set to 0 or 2");
DEFINE_string(model_file,
              "",
              "model file name, used when load mace model in pb");
DEFINE_string(device, "GPU", "CPU/GPU/HEXAGON/APU");
DEFINE_int32(round, 1, "round");
DEFINE_int32(restart_round, 1, "restart round");
DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");
DEFINE_int32(gpu_perf_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_priority_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_memory_type, 0, "0:IMAGE/1:BUFFER");
DEFINE_int32(omp_num_threads, -1, "num of openmp threads");
DEFINE_int32(cpu_affinity_policy, 1,
             "0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY");
DEFINE_bool(benchmark, false, "enable benchmark op");
DEFINE_int32(partition_dim, 1, "0:NONE/1:INPUT_HEIGHT/2:INPUT_WIDTH/3:INPUT_CHANNEL/4:OUTPUT_CHANNEL");
DEFINE_double(partition_ratio, 1.0f, "partition ratio (0:CPU/1:GPU)");

bool RunModel(const std::string &model_name,
              const std::vector<std::string> &input_names,
              const std::vector<std::vector<int64_t>> &input_shapes,
              const std::vector<DataFormat> &input_data_formats,
              const std::vector<std::string> &output_names,
              const std::vector<std::vector<int64_t>> &output_shapes,
              const std::vector<DataFormat> &output_data_formats,
              float cpu_capability) {
  LOG(INFO) << "Run model";

  DeviceType device_type = ParseDeviceType(FLAGS_device);

  int64_t t0 = NowMicros();
  // config runtime
  MaceStatus status;
  MaceEngineConfig config(device_type);
  status = config.SetCPUThreadPolicy(
          FLAGS_omp_num_threads,
          static_cast<CPUAffinityPolicy >(FLAGS_cpu_affinity_policy));
  if (status != MaceStatus::MACE_SUCCESS) {
    LOG(WARNING) << "Set openmp or cpu affinity failed.";
  }
#ifdef MACE_ENABLE_OPENCL
  std::shared_ptr<GPUContext> gpu_context;
  if (device_type == DeviceType::GPU) {
    const char *storage_path_ptr = getenv("MACE_INTERNAL_STORAGE_PATH");
    const std::string storage_path =
        std::string(storage_path_ptr == nullptr ?
                    "/data/local/tmp/mace_run/interior" : storage_path_ptr);
    std::vector<std::string> opencl_binary_paths = {FLAGS_opencl_binary_file};

    gpu_context = GPUContextBuilder()
        .SetStoragePath(storage_path)
        .SetOpenCLBinaryPaths(opencl_binary_paths)
        .SetOpenCLParameterPath(FLAGS_opencl_parameter_file)
        .Finalize();

    config.SetGPUContext(gpu_context);
    config.SetGPUHints(static_cast<GPUPerfHint>(FLAGS_gpu_perf_hint),
                       static_cast<GPUPriorityHint>(FLAGS_gpu_priority_hint));
    config.SetGPUMemoryType(static_cast<GPUMemoryType>(FLAGS_gpu_memory_type));
    // Set gpu precision from environment.
    const char *gpu_dprec = getenv("MACE_GPU_DATA_PRECISION");
    if (!(gpu_dprec != nullptr && strlen(gpu_dprec) == 2)) {
      gpu_dprec = "32";
    }
    if (!strcmp(gpu_dprec, "16")) {
      config.SetGPUPrecisionHint(1);
    } else if (!strcmp(gpu_dprec, "32")) {
      config.SetGPUPrecisionHint(2);
    } else {
      config.SetGPUPrecisionHint(1);
    }
  }
#endif  // MACE_ENABLE_OPENCL
#ifdef MACE_ENABLE_HEXAGON
  LOG(INFO) << "Set hexagon configure";
  config.SetHexagonToUnsignedPD();
  config.SetHexagonPower(HEXAGON_NN_CORNER_TURBO, true, 100);
#endif

  // Remember this setting, otherwise an unknown error will appear.
  config.SetRuntimeStatisticsEnabled(true);

  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_graph_data =
    make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (FLAGS_model_file != "") {
    auto fs = GetFileSystem();
    status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_file.c_str(),
        &model_graph_data);
    if (status != MaceStatus::MACE_SUCCESS) {
      LOG(FATAL) << "Failed to read file: " << FLAGS_model_file;
    }
  }

  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_weights_data =
    make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (FLAGS_model_data_file != "") {
    auto fs = GetFileSystem();
    status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_data_file.c_str(),
        &model_weights_data);
    if (status != MaceStatus::MACE_SUCCESS) {
      LOG(FATAL) << "Failed to read file: " << FLAGS_model_data_file;
    }
  }

  std::shared_ptr<mace::MaceEngine> engine;
  MaceStatus create_engine_status;

  while (true) {
    // Create Engine
    int64_t t0 = NowMicros();
#ifdef MODEL_GRAPH_FORMAT_CODE
    if (model_name.empty()) {
      LOG(INFO) << "Please specify model name you want to run";
      return false;
    }
    create_engine_status =
          CreateMaceEngineFromCode(model_name,
                                   reinterpret_cast<const unsigned char *>(
                                     model_weights_data->data()),
                                   model_weights_data->length(),
                                   input_names,
                                   output_names,
                                   config,
                                   &engine);
#else
    (void)(model_name);
    if (model_graph_data == nullptr || model_weights_data == nullptr) {
      LOG(INFO) << "Please specify model graph file and model data file";
      return false;
    }
    create_engine_status =
        CreateMaceEngineFromProto(reinterpret_cast<const unsigned char *>(
                                    model_graph_data->data()),
                                  model_graph_data->length(),
                                  reinterpret_cast<const unsigned char *>(
                                    model_weights_data->data()),
                                  model_weights_data->length(),
                                  input_names,
                                  output_names,
                                  config,
                                  &engine);
#endif
    int64_t t1 = NowMicros();

    if (create_engine_status != MaceStatus::MACE_SUCCESS) {
      LOG(ERROR) << "Create engine runtime error, retry ... errcode: "
                 << create_engine_status.information();
    } else {
      double create_engine_millis = (t1 - t0) / 1000.0;
      LOG(INFO) << "Create Mace Engine latency: " << create_engine_millis
                << " ms";
      break;
    }
  }
  int64_t t1 = NowMicros();
  double init_millis = (t1 - t0) / 1000.0;
  LOG(INFO) << "Total init latency: " << init_millis << " ms";

  const size_t input_count = input_names.size();
  const size_t output_count = output_names.size();

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
  std::map<std::string, int64_t> inputs_size;
  for (size_t i = 0; i < input_count; ++i) {
    // Allocate input and output
    // only support float and int32, use char for generalization
    // sizeof(int) == 4, sizeof(float) == 4
    int64_t input_size =
        std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 4,
                        std::multiplies<int64_t>());
    inputs_size[input_names[i]] = input_size;
    auto buffer_in = std::shared_ptr<char>(new char[input_size],
                                           std::default_delete<char[]>());
    // load input
#if 0
    const std::string in_filename
        = FLAGS_input_file + "_" + FormatName(input_names[i]);
#endif
    const std::string in_filename = FLAGS_input_file;
    std::ifstream in_file(in_filename, std::ios::in | std::ios::binary);
    if (in_file.is_open()) {
      in_file.read(buffer_in.get(), input_size);
      in_file.close();
    } else {
      LOG(INFO) << "Open input file failed";
      LOG(INFO) << "Input file should be " << in_filename;
      return -1;
    }
    inputs[input_names[i]] = mace::MaceTensor(input_shapes[i], buffer_in,
        input_data_formats[i]);
  }

  for (size_t i = 0; i < output_count; ++i) {
    // only support float and int32, use char for generalization
    int64_t output_size =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 4,
                        std::multiplies<int64_t>());
    auto buffer_out = std::shared_ptr<char>(new char[output_size],
                                            std::default_delete<char[]>());
    outputs[output_names[i]] = mace::MaceTensor(output_shapes[i], buffer_out,
        output_data_formats[i]);
  }

  //mace::PartitionRunConfig part_run_config(
  //    (float) FLAGS_partition_ratio, mace::PartitionDim::DIM_NONE);

  if (!FLAGS_input_dir.empty()) {
    DIR *dir_parent;
    //struct dirent *entry;
    struct dirent **namelist;
    dir_parent = opendir(FLAGS_input_dir.c_str());
    std::vector<int> class_indices;
    if (dir_parent) {
      int num_files = scandir(FLAGS_input_dir.c_str(), &namelist, 0, alphasort);
      //while ((entry = readdir(dir_parent))) {
      for (int fi = 0; fi < num_files; fi ++) {
        //std::string file_name = std::string(entry->d_name);
        std::string file_name = namelist[fi]->d_name;
        std::string prefix = FormatName(input_names[0]);
        LOG(INFO) << "File name: " << file_name;
        LOG(INFO) << "Prefix: " << prefix;
        //const bool is_contain_prefix = file_name.find(prefix) == 0;
        const bool is_contain_prefix = file_name.find(prefix) != file_name.npos;
        if (is_contain_prefix) {
          //std::string suffix = file_name.substr(prefix.size());
          //LOG(INFO) << "Suffix: " << suffix;
          for (size_t i = 0; i < input_count; ++i) {
            //std::string input_name = FLAGS_input_dir + "/" + FormatName(input_names[i])
            //    + suffix;
            std::string input_name = FLAGS_input_dir + "/" + file_name;
            std::ifstream in_file(input_name, std::ios::in | std::ios::binary);
            //std::cout << "Read " << input_name << std::endl;
            LOG(INFO) << "Read " << input_name;
            if (in_file.is_open()) {
              int64_t input_size = inputs_size[input_names[i]];
              LOG(INFO) << "Input size: " << input_size;
              in_file.read(reinterpret_cast<char *>(
                               inputs[input_names[i]].data().get()),
                           input_size);
              in_file.close();


              LOG(INFO) << "Input data: " << DataToString<float>(
                  reinterpret_cast<const float *>(inputs[input_names[i]].data().get()),
                  input_size / sizeof(float), 3 * 3);
            } else {
              std::cerr << "Open input file failed" << std::endl;
              return -1;
            }
          }
          
          engine->Run(inputs, &outputs);

          if (!FLAGS_output_dir.empty()) {
            for (size_t i = 0; i < output_count; ++i) {
              //std::string output_name = 
              //    FLAGS_output_dir + "/" + FormatName(output_names[i]) + suffix;
              std::string output_name = FLAGS_output_dir + "/mace_out_" + file_name;
              LOG(INFO) << "Output name: " << output_name;
              std::ofstream out_file(output_name, std::ios::binary);
              if (out_file.is_open()) {
                int64_t output_size =
                    std::accumulate(output_shapes[i].begin(),
                                    output_shapes[i].end(),
                                    1,
                                    std::multiplies<int64_t>());
                char *output_data = reinterpret_cast<char *>(
                    outputs[output_names[i]].data().get());
                out_file.write(output_data, output_size * sizeof(float));
                out_file.flush();
                out_file.close();

                // Print data.
                LOG(INFO) << "Output size: " << output_size;
                LOG(INFO) << "Output data: " << DataToString<float>(
                    reinterpret_cast<const float *>(output_data),
                    output_size,
                    8);

                // Compute argmax and append to vector.
                const int class_idx = CalcArgmax<float>(
                    reinterpret_cast<float *>(output_data), output_size);
                LOG(INFO) << "Class idx: " << class_idx;
                class_indices.push_back(class_idx);
              } else {
                std::cerr << "Open output file failed" << std::endl;
                return -1;
              }
            }
          }
        }
      }

      closedir(dir_parent);

      LOG(INFO) << "Argmax: " << VectorToString<int>(class_indices);
    } else {
      std::cerr << "Directory " << FLAGS_input_dir << " does not exist."
                << std::endl;
    }
  } else {
    LOG(INFO) << "Warm up run";
    double warmup_millis;
    while (true) {
      //float init_part_ratio = FLAGS_partition_ratio == 1.0f ? 1.0f : 0.0f;
      float init_part_ratio = FLAGS_partition_ratio;
      PartitionDim part_dim = static_cast<PartitionDim>(FLAGS_partition_dim);
      std::shared_ptr<mace::PartitionRunConfig> init_part_info;
      if (part_dim == DIM_INPUT_HEIGHT) {
        init_part_info.reset(
            new mace::PartitionRunConfig(init_part_ratio, part_dim));
      } else if (part_dim == DIM_OUTPUT_CHANNEL) {
        init_part_info.reset(
            new mace::PartitionRunConfig(init_part_ratio, part_dim));
      }

      int64_t t3 = NowMicros();
      MaceStatus warmup_status = engine->Run(inputs, &outputs, init_part_info.get());

      if (warmup_status != MaceStatus::MACE_SUCCESS) {
        LOG(ERROR) << "Warmup runtime error, retry ... errcode: "
                   << warmup_status.information();
        do {
#ifdef MODEL_GRAPH_FORMAT_CODE
          create_engine_status =
            CreateMaceEngineFromCode(model_name,
                                     reinterpret_cast<const unsigned char *>(
                                       model_weights_data->data()),
                                     model_weights_data->length(),
                                     input_names,
                                     output_names,
                                     config,
                                     &engine);
#else
          create_engine_status =
              CreateMaceEngineFromProto(reinterpret_cast<const unsigned char *>(
                                            model_graph_data->data()),
                                        model_graph_data->length(),
                                        reinterpret_cast<const unsigned char *>(
                                            model_weights_data->data()),
                                        model_weights_data->length(),
                                        input_names,
                                        output_names,
                                        config,
                                        &engine);
#endif
        } while (create_engine_status != MaceStatus::MACE_SUCCESS);
      } else {
        int64_t t4 = NowMicros();
        warmup_millis = (t4 - t3) / 1000.0;
        LOG(INFO) << "1st warm up run latency: " << warmup_millis << " ms";
        break;
      }
    }

    double model_run_millis = -1;
    benchmark::OpStat op_stat;
    if (FLAGS_round > 0) {
      LOG(INFO) << "Run model";
      int64_t total_run_duration = 0;
      // 10
      int pr_stride = 0;
      int pr_value = (int) (FLAGS_partition_ratio * 100.0f);
      if (pr_value <= 0) {
        // -10
        pr_stride = 0;
      }
      std::vector<double> run_duration_list;
      for (int i = 0; i < FLAGS_round; ++i) {
        std::unique_ptr<port::Logger> info_log;
        std::unique_ptr<port::MallocLogger> malloc_logger;
        if (FLAGS_malloc_check_cycle >= 1
            && i % FLAGS_malloc_check_cycle == 0) {
          info_log = LOG_PTR(INFO);
          malloc_logger = port::Env::Default()->NewMallocLogger(
              info_log.get(), MakeString(i));
        }
        MaceStatus run_status;
        RunMetadata metadata;
        RunMetadata *metadata_ptr = nullptr;
        if (FLAGS_benchmark) {
          metadata_ptr = &metadata;
        }

        while (true) {
          PartitionDim part_dim = static_cast<PartitionDim>(FLAGS_partition_dim);
          mace::PartitionRunConfig part_run_config((float) pr_value / 100.0f, part_dim);
          
          int64_t t0 = NowMicros();
          run_status = engine->Run(inputs, &outputs, metadata_ptr, &part_run_config);
          if (run_status != MaceStatus::MACE_SUCCESS) {
            LOG(ERROR) << "Mace run model runtime error, retry ... errcode: "
                       << run_status.information();
            do {
#ifdef MODEL_GRAPH_FORMAT_CODE
              create_engine_status =
                CreateMaceEngineFromCode(
                    model_name,
                    reinterpret_cast<const unsigned char *>(
                      model_weights_data->data()),
                    model_weights_data->length(),
                    input_names,
                    output_names,
                    config,
                    &engine);
#else
              create_engine_status =
                  CreateMaceEngineFromProto(
                      reinterpret_cast<const unsigned char *>(
                          model_graph_data->data()),
                      model_graph_data->length(),
                      reinterpret_cast<const unsigned char *>(
                          model_weights_data->data()),
                      model_weights_data->length(),
                      input_names,
                      output_names,
                      config,
                      &engine);
#endif
            } while (create_engine_status != MaceStatus::MACE_SUCCESS);
          } else {
            int64_t t1 = NowMicros();
            double run_duration = (t1 - t0) / 1000.0;
            total_run_duration += (t1 - t0);
            run_duration_list.push_back(run_duration);
            if (FLAGS_benchmark) {
              op_stat.StatMetadata(metadata);
            }
            double gflops = 0;
            if (engine->runtime_stat()->flops_stat() != nullptr) {
              gflops =
                ((double) engine->runtime_stat()->flops_stat()->value()) * 1e-9;
            }
            
            double pration = (pr_value * 1.0f / 100.0f);
            LOG(INFO) << "Round " << i
                      << ", latency " << run_duration << " ms"
                      << ", GFLOPs " << gflops
                      << ", pratio " << pration;
            break;
          }
        }

        if ((i + 1) % 20 == 0) {
          if (pr_value <= 0 || pr_value >= 100)
            pr_stride = pr_stride * (-1);
          pr_value += pr_stride;
        }
      }
      model_run_millis = total_run_duration / 1000.0 / FLAGS_round;
      LOG(INFO) << "Average latency: " << model_run_millis << " ms";
      const int start_idx = 2;
      const std::vector<double> stat =
          mace::benchmark::StatRunDuration(run_duration_list, start_idx);
      LOG(INFO) << "Stat: start_idx " << start_idx
                << ", latency " << VectorToString<double>(stat);
    }

    for (size_t i = 0; i < output_count; ++i) {
      std::string output_name =
          FLAGS_output_file + "_" + FormatName(output_names[i]);
      std::ofstream out_file(output_name, std::ios::binary);
      // only support float and int32
      int64_t output_size =
          std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 4,
                          std::multiplies<int64_t>());
      out_file.write(outputs[output_names[i]].data<char>().get(), output_size);
      out_file.flush();
      out_file.close();
      LOG(INFO) << "Write output file "
                << output_name << " with size "
                << output_size << " done.";
#if 0
      const int64_t max_count = output_size / sizeof(float);
      LOG(INFO) << DataToString<float>(outputs[output_names[i]].data<float>().get(),
                                       output_size / sizeof(float), max_count);
#endif
      LOG(INFO) << "Argmax: " << CalcArgmax<float>(
          outputs[output_names[i]].data<float>().get(), output_size / sizeof(float));
    }

    // Metrics reporting tools depends on the format, keep in consistent
    printf("========================================================\n");
    printf("     capability(CPU)        init      warmup     run_avg\n");
    printf("========================================================\n");
    printf("time %15.3f %11.3f %11.3f %11.3f\n",
           cpu_capability, init_millis, warmup_millis, model_run_millis);
    if (FLAGS_benchmark) {
      op_stat.PrintStat();
    }
  }

  return true;
}

int Main(int argc, char **argv) {
  std::string usage = "MACE run model tool, please specify proper arguments.\n"
                      "usage: " + std::string(argv[0])
                      + " --help";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> input_names = Split(FLAGS_input_node, ',');
  std::vector<std::string> output_names = Split(FLAGS_output_node, ',');
  if (input_names.empty() || output_names.empty()) {
    LOG(INFO) << gflags::ProgramUsage();
    return 0;
  }

  if (FLAGS_benchmark) {
    setenv("MACE_OPENCL_PROFILING", "1", 1);
  }

  LOG(INFO) << "model name: " << FLAGS_model_name;
  LOG(INFO) << "mace version: " << MaceVersion();
  LOG(INFO) << "input node: " << FLAGS_input_node;
  LOG(INFO) << "input shape: " << FLAGS_input_shape;
  LOG(INFO) << "output node: " << FLAGS_output_node;
  LOG(INFO) << "output shape: " << FLAGS_output_shape;
  LOG(INFO) << "input_file: " << FLAGS_input_file;
  LOG(INFO) << "output_file: " << FLAGS_output_file;
  LOG(INFO) << "model_data_file: " << FLAGS_model_data_file;
  LOG(INFO) << "model_file: " << FLAGS_model_file;
  LOG(INFO) << "device: " << FLAGS_device;
  LOG(INFO) << "round: " << FLAGS_round;
  LOG(INFO) << "restart_round: " << FLAGS_restart_round;
  LOG(INFO) << "gpu_perf_hint: " << FLAGS_gpu_perf_hint;
  LOG(INFO) << "gpu_priority_hint: " << FLAGS_gpu_priority_hint;
  LOG(INFO) << "gpu_memory_type: " << FLAGS_gpu_memory_type;
  LOG(INFO) << "omp_num_threads: " << FLAGS_omp_num_threads;
  LOG(INFO) << "cpu_affinity_policy: " << FLAGS_cpu_affinity_policy;
  LOG(INFO) << "partition_dim: " << FLAGS_partition_dim;
  LOG(INFO) << "partition_ratio: " << FLAGS_partition_ratio;

  auto limit_opencl_kernel_time = getenv("MACE_LIMIT_OPENCL_KERNEL_TIME");
  if (limit_opencl_kernel_time) {
    LOG(INFO) << "limit_opencl_kernel_time: "
              << limit_opencl_kernel_time;
  }
  auto opencl_queue_window_size = getenv("MACE_OPENCL_QUEUE_WINDOW_SIZE");
  if (opencl_queue_window_size) {
    LOG(INFO) << "opencl_queue_window_size: "
              << getenv("MACE_OPENCL_QUEUE_WINDOW_SIZE");
  }

  std::vector<std::string> input_shapes = Split(FLAGS_input_shape, ':');
  std::vector<std::string> output_shapes = Split(FLAGS_output_shape, ':');

  const size_t input_count = input_shapes.size();
  const size_t output_count = output_shapes.size();
  std::vector<std::vector<int64_t>> input_shape_vec(input_count);
  std::vector<std::vector<int64_t>> output_shape_vec(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    ParseShape(input_shapes[i], &input_shape_vec[i]);
  }
  for (size_t i = 0; i < output_count; ++i) {
    ParseShape(output_shapes[i], &output_shape_vec[i]);
  }
  if (input_names.size() != input_shape_vec.size()
      || output_names.size() != output_shape_vec.size()) {
    LOG(INFO) << "inputs' names do not match inputs' shapes "
                 "or outputs' names do not match outputs' shapes";
    return 0;
  }
  std::vector<std::string> raw_input_data_formats =
    Split(FLAGS_input_data_format, ',');
  std::vector<std::string> raw_output_data_formats =
    Split(FLAGS_output_data_format, ',');
  std::vector<DataFormat> input_data_formats(input_count);
  std::vector<DataFormat> output_data_formats(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    input_data_formats[i] = ParseDataFormat(raw_input_data_formats[i]);
  }
  for (size_t i = 0; i < output_count; ++i) {
    output_data_formats[i] = ParseDataFormat(raw_output_data_formats[i]);
  }
  const bool do_get_cpu_capability = false;
  float cpu_float32_performance = 0.0f;
  if (do_get_cpu_capability && FLAGS_input_dir.empty()) {
    // get cpu capability
    Capability cpu_capability = GetCapability(DeviceType::CPU);
    cpu_float32_performance = cpu_capability.float32_performance.exec_time;
  }
  bool ret = false;
  for (int i = 0; i < FLAGS_restart_round; ++i) {
    VLOG(0) << "restart round " << i;
    ret = RunModel(FLAGS_model_name,
        input_names, input_shape_vec, input_data_formats,
        output_names, output_shape_vec, output_data_formats,
        cpu_float32_performance);
  }
  if (ret) {
    return 0;
  }
  return -1;
}

}  // namespace tools
}  // namespace mace

int main(int argc, char **argv) {
  mace::tools::Main(argc, argv);
}
