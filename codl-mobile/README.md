
## Introduction

This repository is the core implementation of CoDL that contains our proposed techniques, i.e., hybrid-dimension partitioning, chain searching algorithm, and non-linear and concurrency-aware latency prediction. We build CoDL as binary files to evaluate its performance. To build the executable binary files, you should install the requirements and follow the instructions below. We also provide evaluation tools (named `codl-eval-tools`) to execute the binary files in an easy way.

## Requirements 

### For Personal Computer (PC)

|NAME|REQUIREMENT|REFERENCE|
|:-|:-|:-|
|Operation System (OS)|Ubuntu >=16.04|https://releases.ubuntu.com/16.04/|
|Android Debug Bridge (ADB)|>=1.0.41|https://developer.android.com/studio/releases/platform-tools|
|Android NDK|r17c|https://developer.android.com/ndk/downloads/older_releases|
|Bazel|0.13.0|https://github.com/bazelbuild/bazel/releases/tag/0.13.0|
|Python|>=3.6|`add-apt-repository ppa:jblgf0/python`<br>`apt-get update`<br>`apt-get install python3.6`<br>`ln -s /usr/bin/python3 /usr/bin/python`|
|Jinja2|2.10.1|`python -m pip install jinja2==2.10.1`|

If you use a docker container to setup, these requirements should be further installed.

|NAME|REQUIREMENT|REFERENCE|
|:-|:-|:-|
|Java|>=1.8.0|`apt-get install default-jre default-jdk`|
|Git|>=2.7.4|`apt-get install git`|
|Libtinfo5|>=6.0|`apt-get install libtinfo5`|
|MarkupSafe|2.0.1|`python -m pip install MarkupSafe==2.0.1`|
|CMake|>=3.22.4|`apt-get install cmake`|

To install all requirements on PC in an easy way, we provide a Dockerfile at `codl-mobisys2022-artifact-evaluation/Dockerfile`. You can build the Docker image as follow:

```shell
cd /path/to/codl-mobisys2022-artifact-evaluation

docker build -t codl/codl -f ./Dockerfile .
```

Then you can run a Docker container using the Docker image as follow:

```shell
docker run -it --name codl-u16 --privileged -v /dev/bus/usb:/dev/bus/usb codl/codl:latest /bin/bash
```

Note that the Docker version we use is 20.10.12 (build e91ed57). We use the `-v /dev/bus/usb:/dev/bus/usb` argument of Docker to enable the ADB connection with USB in a Docker container. We basically follow the instruction of [How to connect adb devices to linux container](http://learningbysimpleway.blogspot.com/2018/02/how-to-connect-adb-devices-to-linux.html). You can refer to it if you want more details about the setup of ADB connection with USB in a Docker container.

### For Mobile Devices

|NAME|REQUIREMENT|
|:-|:-|
|Device Type|Smartphone|
|System of Chip (SoC)|Snapdragon or Kirin (we recommend Snapdragon 855/865/888 and Kirin 990)|
|Application Binary Interface (ABI)|arm64-v8a|
|OS|Android|
|API Support|OpenCL 2.0 Full|

## Instruction

### Step 0: Set Environment Variables (on PC)

Export these variables in current environment. If you use the Dockerfile to setup, you can skip this step.

```shell
# Bazel.
export PATH="/path/to/bazel_bin:$PATH"
# Android NDK.
export ANDROID_NDK_HOME="/path/to/android_ndk"
```

### Step 1: Build Executable Files (on PC)

We build two executable files `codl_op_run` and `codl_run`. The `codl_op_run` works for the execution of several operators (i.e., data sharing, convolution, pooling and fully connected). The `codl_run` works for the model inference using CPU and GPU.

We provide a script to build the executable files as follows.

Note: If you use a docker container to build the executable files, please make sure that the directory `/path/to/codl-mobile` must be copied (not mapped using `-v /host/path/to/codl-mobile:/container/path/to/codl-mobile` of docker) to a directory in the docker container, otherwise an issue will occur (see Issue 3 of Trouble Shooting).

```shell
cd /path/to/codl-mobile/

bash tools/codl/build_executable_files.sh
```

This step will take serveral minutes. The built executable files can be found in the directory `build/bin/arm64-v8a/cpu_gpu`.

We also provide prebuilt executable files in the directory `prebuild/bin/arm64-v8a/cpu_gpu`. You can rename `prebuild` to `build` without building the executable files.

### Step 2: Push Executable Files to Mobile Device

Connect the mobile device to PC with a USB cable or ADB Wi-Fi. Then use `tools/codl/push_executable_files.sh` as below to push the executable files to the mobile device.

```shell
cd /path/to/codl-mobile/

# If only one mobile device is connected to PC, adb_device is not required.
# Otherwise, you should specify an adb-assigned serial number of device as adb_device.
bash tools/codl/push_executable_files.sh [adb_device]
```

Note that we create a directory at `/data/local/tmp/codl` on the mobile device as workspace.

## Trouble Shooting

### Issue 1: /bin/bash: cmake: command not found

This issue occurs if there is no CMake in your environment. The complete log text of this issue is as follows. The solution is installing CMake by `sudo apt-get install cmake`.

```shell
ERROR: /root/.cache/bazel/_bazel_root/e78289bffba4bcf497303d5b399049da/external/opencl_clhpp/BUILD.bazel:5:1: Executing genrule @opencl_clhpp//:gen_opencl_clhpp failed (Exit 127): bash failed: error executing command 
  (cd /root/.cache/bazel/_bazel_root/e78289bffba4bcf497303d5b399049da/execroot/mace && \
  exec env - \
    PATH=/home/ubuntu/Software/bazel:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /bin/bash -c 'source external/bazel_tools/tools/genrule/genrule-setup.sh; workdir=$(mktemp -d -t opencl-clhpp-build.XXXXXXXXXX); cp -aL $(dirname external/opencl_clhpp/CMakeLists.txt)/* $workdir; pushd $workdir; mkdir build; pushd build; cmake ../ -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF; make generate_clhpp generate_cl2hpp; popd; popd; cp -a $workdir/build/* bazel-out/arm64-v8a-opt/genfiles/external/opencl_clhpp; rm -rf $workdir; echo installing to  bazel-out/arm64-v8a-opt/genfiles/external/opencl_clhpp')

Use --sandbox_debug to see verbose messages from the sandbox
/tmp/opencl-clhpp-build.W5LOH8MiFq /root/.cache/bazel/_bazel_root/e78289bffba4bcf497303d5b399049da/sandbox/3316531805327121904/execroot/mace
/tmp/opencl-clhpp-build.W5LOH8MiFq/build /tmp/opencl-clhpp-build.W5LOH8MiFq /root/.cache/bazel/_bazel_root/e78289bffba4bcf497303d5b399049da/sandbox/3316531805327121904/execroot/mace
/bin/bash: cmake: command not found
Target //test/codl_run:codl_op_run failed to build
INFO: Elapsed time: 6.849s, Critical Path: 0.33s
INFO: 4 processes: 2 linux-sandbox, 2 local.
FAILED: Build did NOT complete successfully
```

### Issue 2: external/androidndk/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/clang: error while loading shared libraries: libtinfo.so.5: cannot open shared object file: No such file or directory

This issue occurs if there is no Libtinfo5 in your environment. The complete log text of this issue is as follows. The solution is installing Libtinfo5 by `apt install libtinfo5`.

```shell
ERROR: /root/.cache/bazel/_bazel_root/e78289bffba4bcf497303d5b399049da/external/com_google_protobuf/BUILD:70:1: C++ compilation of rule '@com_google_protobuf//:protobuf_lite' failed (Exit 127): clang failed: error executing command 
  (cd /root/.cache/bazel/_bazel_root/e78289bffba4bcf497303d5b399049da/execroot/mace && \
  exec env - \
    PWD=/proc/self/cwd \
  external/androidndk/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/clang -gcc-toolchain external/androidndk/ndk/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64 -target aarch64-none-linux-android -ffunction-sections -funwind-tables -fstack-protector-strong -fpic -Wno-invalid-command-line-argument -Wno-unused-command-line-argument -no-canonical-prefixes -isystemexternal/androidndk/ndk/sysroot/usr/include/aarch64-linux-android '-D__ANDROID_API__=21' -O2 -g -DNDEBUG -MD -MF bazel-out/arm64-v8a-opt/bin/external/com_google_protobuf/_objs/protobuf_lite/external/com_google_protobuf/src/google/protobuf/arena.d '-frandom-seed=bazel-out/arm64-v8a-opt/bin/external/com_google_protobuf/_objs/protobuf_lite/external/com_google_protobuf/src/google/protobuf/arena.o' -iquote external/com_google_protobuf -iquote bazel-out/arm64-v8a-opt/genfiles/external/com_google_protobuf -iquote external/bazel_tools -iquote bazel-out/arm64-v8a-opt/genfiles/external/bazel_tools -isystem external/com_google_protobuf/src -isystem bazel-out/arm64-v8a-opt/genfiles/external/com_google_protobuf/src -isystem bazel-out/arm64-v8a-opt/bin/external/com_google_protobuf/src '-std=c++17' -fPIC -D_GLIBCXX_USE_C99_MATH_TR1 -DMACE_OBFUSCATE_LITERALS -DGEMMLOWP_USE_MACE_THREAD_POOL -DGEMMLOWP_TEST_PROFILE -DMACE_DEPTHWISE_U8_USE_MULTI_THREAD -Wall -Wno-mismatched-tags -Wno-missing-braces -O3 -ffunction-sections -fdata-sections -DHAVE_PTHREAD -Wall -Wwrite-strings -Woverloaded-virtual -Wno-sign-compare -Wno-unused-function -Wno-writable-strings '--sysroot=external/androidndk/ndk/platforms/android-21/arch-arm64' -isystem external/androidndk/ndk/sources/android/support/include -isystem external/androidndk/ndk/sources/cxx-stl/llvm-libc++/include -isystem external/androidndk/ndk/sources/cxx-stl/llvm-libc++abi/include -isystemexternal/androidndk/ndk/sysroot/usr/include -c external/com_google_protobuf/src/google/protobuf/arena.cc -o bazel-out/arm64-v8a-opt/bin/external/com_google_protobuf/_objs/protobuf_lite/external/com_google_protobuf/src/google/protobuf/arena.o)

Use --sandbox_debug to see verbose messages from the sandbox
external/androidndk/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/clang: error while loading shared libraries: libtinfo.so.5: cannot open shared object file: No such file or directory
Target //test/codl_run:codl_op_run failed to build
INFO: Elapsed time: 25.224s, Critical Path: 14.50s
INFO: 161 processes: 159 linux-sandbox, 2 local.
FAILED: Build did NOT complete successfully
```

### Issue 3: cp: failed to preserve ownership for '/tmp/opencl-clhpp-build.XXXXXXXXXX/BUILD.bazel': Invalid argument

This issue occurs if you map the directory `/path/to/codl-mobile` to a directory in a docker container. The complete log text of this issue is as follows. The solution is copying the directory `/path/to/codl-mobile` to a directory in a docker container instead of mapping.

```shell
ERROR: /root/.cache/bazel/_bazel_root/05270bc57d83ec6806bb1cda0cfefe97/external/opencl_clhpp/BUILD.bazel:5:1: Executing genrule @opencl_clhpp//:gen_opencl_clhpp failed (Exit 1): bash failed: error executing command 
  (cd /root/.cache/bazel/_bazel_root/05270bc57d83ec6806bb1cda0cfefe97/execroot/mace && \
  exec env - \
    PATH=/home/ubuntu/Software/bazel:/usr/local/bin:/home/root/Software/bazel:/usr/local/bin:/home/ubuntu/Software/bazel:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /bin/bash -c 'source external/bazel_tools/tools/genrule/genrule-setup.sh; workdir=$(mktemp -d -t opencl-clhpp-build.XXXXXXXXXX); cp -aL $(dirname external/opencl_clhpp/CMakeLists.txt)/* $workdir; pushd $workdir; mkdir build; pushd build; cmake ../ -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF; make generate_clhpp generate_cl2hpp; popd; popd; cp -a $workdir/build/* bazel-out/arm64-v8a-opt/genfiles/external/opencl_clhpp; rm -rf $workdir; echo installing to  bazel-out/arm64-v8a-opt/genfiles/external/opencl_clhpp')

Use --sandbox_debug to see verbose messages from the sandbox
cp: failed to preserve ownership for '/tmp/opencl-clhpp-build.cwUUo7eRXC/BUILD.bazel': Invalid argument
Target //test/codl_run:codl_op_run failed to build
INFO: Elapsed time: 7.781s, Critical Path: 0.47s
INFO: 8 processes: 6 linux-sandbox, 2 local.
FAILED: Build did NOT complete successfully
```

## Technical Contents

The key techniques of CoDL include the hybrid-dimension partitioning, operator chain, chain searching algorithm, and non-linear and concurrency-aware latency prediction. We list the related files of each technique below:

|Technique|File(s)|Description|
|:-:|:-|:-|
|Hybrid-dimension Partitioning|mace/ops/opencl/image/image_to_buffer.cc|We implement the `PartImageToBuffer::Compute` function to transform data from image to buffer. In the function, `dim0` refers to the partitioning dimension that could be height (`H_NCHW`) or output channel (`C_NCHW`). `offset` and `part_length` refers to the offset and length on the partitioning dimension.|
||mace/ops/opencl/image/buffer_to_image.cc|We implement the `PartBufferToImage::Compute` function to transform data from buffer to image. In the function, `dim0` refers to the partitioning dimension that could be height (`H_NCHW`) or output channel (`C_NCHW`). `offset` and `part_length` refers to the offset and length on the partitioning dimension.|
|Operator Chain|test/codl_run/op_test_task_chain.h(.cc)|We implement the operator chain as `CodlOpTaskChain`.|
||test/codl_run/op_chain_executor.h(.cc)|We implement an executor `CodlOpTaskChainExecutor` to execute the operator chain.|
|Chain Searching Algorithm|test/codl_run/op_chain_search.h(.cc)|The chain searching algorithm is implemented in the `ChainSearch::Greedy` function.|
|Non-linear and Concurrency-aware Latency Prediction|mace/ops/gemm_latency_predict.cc<br>mace/ops/conv_2d_lr_latency_predict.cc<br>mace/ops/pooling_latency_predict.cc<br>mace/ops/fully_connected_latency_predict.cc|We implement the latency prediction for various kernels, such as direct convolution, GEMM and Winograd. The features we used for each prediction model can be found in the `XxxLatencyPredictor::Predict` function, where `Xxx` refers to the kernel name (e.g., `Conv2dCpuDirect` for direct concolution on CPU).|
