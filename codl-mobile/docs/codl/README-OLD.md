
## Introduction

We implement a CPU+GPU execution pipeline for Conv2D operator based on MACE, which is used for the latency measurement. The key code is located in `test/codlconv2drun` directory. Furthermore, we intergrade our implmentation into the original code of MACE, which can be found in `mace/ops` directory.

## Requirements

|NAME|VERSION|REFERENCE|
|:-|:-|:-|
|Android Debug Bridge (ADB)|1.0.41 or other|https://developer.android.com/studio/releases/platform-tools|
|Android NDK|r17c|https://developer.android.com/ndk/downloads/older_releases|
|Bazel|0.13.0|https://github.com/bazelbuild/bazel/releases/tag/0.13.0|

## Contents

### Image/Buffer Tensor Allocation

|FILE|CONTENT|
|:-|:-|
|[tensor_manage_util.cc](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/core/tensor_manage_util.cc)|A util for image or buffer tensor allocation and management.|

### Image/Buffer Data Transform

|FILE|CONTENT|
|:-|:-|
|[buffer_transformer.cc](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/ops/opencl/buffer_transformer.cc)|Partitial data transforming along with height or output channel dimension.|
|[buffer_to_image.cc](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/ops/opencl/image/buffer_to_image.cc)|Partitial data transforming from buffer to image.|
|[image_to_buffer.cc](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/ops/opencl/image/image_to_buffer.cc)|Partitial data transforming from image to buffer.|
|[buffer_to_image.cl](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/ops/opencl/cl/buffer_to_image.cl)|OpenCL kernel that supports partital data transforming.|

### CPU+GPU Co-execution

|FILE|CONTENT|
|:-|:-|
|[conv_2d_part_plan.cc](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/ops/conv_2d_part_plan.cc)|Build partition plan with specific dimension (height or output channel) and ratio (from 0 to 1).|
|[conv2d_cpu_gpu_test_task.cc](https://github.com/JiaFucheng/codl-mobile/blob/main/test/codlconv2drun/conv2d_cpu_gpu_test_task.cc) or<br>[conv_2d_cpu_gpu_image.cc](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/ops/conv_2d_cpu_gpu_image_v2.cc)|Parallel CPU+GPU co-execution for Conv2D operator, including data transforming, synchronization and kernel computation.|

### CPU+GPU Partition Prediction

|FILE|CONTENT|
|:-|:-|
|[const_model.h](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/utils/const_model.h)|Const model to predict the latency of synchronization.|
|[linear_regression_model.cc](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/utils/linear_regression_model.cc)|Linear regression model to predict the latency of data transforming.|
|[random_forest_model.cc](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/utils/random_forest_model.cc)|Random forest model to predict the latency of CPU/GPU computation.|
|[conv_2d_part_predict.cc](https://github.com/JiaFucheng/codl-mobile/blob/main/mace/ops/conv_2d_part_predict.cc)|Exploit the prediction model to calcuate the final partition dimension and ratio for Conv2D operator.|

## Instruction

### Step 0: ADB Connection

Connect your smartphone to PC with USB cable and run `adb devices` to check the connection.

### Step 1: Export

```shell
export ANDROID_NDK_HOME=PATH/TO/NDK
```

### Step 2: Build

```shell
bash tools/codl/bazel-build-conv2d-run.sh
```

### Step 3: Prepare

```shell
bash tools/codl/prepare-conv2d-run.sh
```

### Step 4: Quick Test

```shell
bash tools/codl/quick-test-conv2d-run.sh
```
