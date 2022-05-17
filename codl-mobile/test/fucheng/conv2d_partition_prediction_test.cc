
#include "mace/public/mace.h"
#include "mace/core/workspace.h"
#include "mace/core/op_context.h"
#include "mace/ops/conv_2d.h"
#include "mace/utils/thread_pool.h"
#include "mace/utils/soc_devfreq.h"
#include "test/fucheng/device_util.h"

namespace mace {

class Conv2dLayer {
public:
  Conv2dLayer(const index_t ih,
              const index_t iw,
              const index_t ic,
              const index_t oc,
              const index_t kh,
              const index_t kw,
              const int sh = 1,
              const int sw = 1,
              const int dh = 1,
              const int dw = 1,
              const Padding padding_type = Padding::SAME)
    : input_height_(ih),
      input_width_(iw),
      input_channel_(ic),
      output_channel_(oc),
      kernel_height_(kh),
      kernel_width_(kw),
      stride_h_(sh),
      stride_w_(sw),
      dilation_h_(dh),
      dilation_w_(dw),
      padding_type_(padding_type) {}

  std::vector<double> as_input_vector() const {
    std::vector<double> inputs;
    inputs.push_back(static_cast<double>(input_height_));
    inputs.push_back(static_cast<double>(input_width_));
    inputs.push_back(static_cast<double>(input_channel_));
    inputs.push_back(static_cast<double>(output_channel_));
    inputs.push_back(static_cast<double>(kernel_height_));
    inputs.push_back(static_cast<double>(kernel_width_));
    inputs.push_back(static_cast<double>(stride_h_));
    inputs.push_back(static_cast<double>(stride_w_));
    inputs.push_back(static_cast<double>(dilation_h_));
    inputs.push_back(static_cast<double>(dilation_w_));
    inputs.push_back(static_cast<double>(static_cast<int>(padding_type_)));
    return inputs;
  }

private:
  index_t input_height_;
  index_t input_width_;
  index_t input_channel_;
  index_t output_channel_;
  index_t kernel_height_;
  index_t kernel_width_;
  int stride_h_;
  int stride_w_;
  int dilation_h_;
  int dilation_w_;
  Padding padding_type_;
};

void BuildYolov2(std::vector<Conv2dLayer> &layers) {
  layers.emplace_back(416, 416, 3, 32, 3, 3);
  layers.emplace_back(208, 208, 32, 64, 3, 3);
  layers.emplace_back(104, 104, 64, 128, 3, 3);
  layers.emplace_back(104, 104, 128, 64, 1, 1);
  layers.emplace_back(104, 104, 64, 128, 3, 3);
  layers.emplace_back(52, 52, 128, 256, 3, 3);
  layers.emplace_back(52, 52, 256, 128, 1, 1);
  layers.emplace_back(52, 52, 128, 256, 3, 3);
  layers.emplace_back(26, 26, 256, 512, 3, 3);
  layers.emplace_back(26, 26, 512, 256, 1, 1);
  layers.emplace_back(26, 26, 256, 512, 3, 3);
  layers.emplace_back(26, 26, 512, 256, 1, 1);
  layers.emplace_back(26, 26, 256, 512, 3, 3);
  layers.emplace_back(13, 13, 512, 1024, 3, 3);
  layers.emplace_back(13, 13, 1024, 512, 1, 1);
  layers.emplace_back(13, 13, 512, 1024, 3, 3);
  layers.emplace_back(13, 13, 1024, 512, 1, 1);
  layers.emplace_back(13, 13, 512, 1024, 3, 3);
  layers.emplace_back(13, 13, 1024, 1024, 3, 3);
  layers.emplace_back(13, 13, 1024, 1024, 3, 3);
  layers.emplace_back(13, 13, 1536, 1024, 3, 3);
  layers.emplace_back(13, 13, 1024, 1024, 1, 1);
}

MaceStatus Conv2dPartitionPredictionTest() {
  const int num_threads = 4;
  const CPUAffinityPolicy cpu_affinity = CPUAffinityPolicy::AFFINITY_BIG_ONLY;

  TestDeviceContext device_context(num_threads, cpu_affinity);
  device_context.InitGpuDevice();

  OpContext context(GetWorkspace(), device_context.GetGpuDevice());

  std::shared_ptr<utils::SocDevfreq> devfreq(utils::SocDevfreq::CreateLocal());
  float cpu_capability = devfreq->cpu_capability();
  float gpu_capability = devfreq->gpu_capability();
  LOG(INFO) << "CPU capability " << cpu_capability
            << " GPU capability " << gpu_capability;

  ops::Conv2dPartitionRatioPredictor predictor;
  predictor.Init();

  std::vector<Conv2dLayer> layers;
  BuildYolov2(layers);
  
  std::vector<double> inputs;
  std::vector<double> outputs;
  for (auto iter = layers.begin(); iter != layers.end(); ++iter) {
    auto layer = *iter;
    inputs = layer.as_input_vector();
    predictor.Predict(&context, inputs, outputs);

    LOG(INFO) << "Predicted dimension " << static_cast<int>(outputs[0])
              << " ratio " << outputs[1];
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace

int main(int argc, char *argv[]) {
  MACE_UNUSED(argc);
  MACE_UNUSED(argv);
  mace::Conv2dPartitionPredictionTest();
  return 0;
}
