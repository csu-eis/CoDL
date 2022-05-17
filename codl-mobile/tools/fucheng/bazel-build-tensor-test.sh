#!/bin/bash

set -e

BIN_DIR=build/bin

mkdir -p $BIN_DIR

rm -rf $BIN_DIR/arm64-v8a

mkdir -p $BIN_DIR/arm64-v8a/cpu_gpu

build_tensor_buffer_test() {
  echo "build tensor buffer test for arm64-v8a + cpu_gpu"

  bazel build \
    --config android \
    --config optimization \
    test/fucheng:tensor_buffer_test \
    --define neon=true \
    --define opencl=true \
    --define quantize=true \
    --define rpcmem=false \
    --define codl=false \
    --define buildlib=false \
    --cpu=arm64-v8a
  
  cp bazel-bin/test/fucheng/tensor_buffer_test \
    $BIN_DIR/arm64-v8a/cpu_gpu/tensor_buffer_test

  echo "BIN PATH: $BIN_DIR"
}

build_tensor_buffer_gpu_test() {
  echo "build tensor buffer gpu test for arm64-v8a + cpu_gpu"

  bazel build \
    --config android \
    --config optimization \
    test/fucheng:tensor_buffer_gpu_test \
    --define neon=true \
    --define opencl=true \
    --define quantize=true \
    --define rpcmem=false \
    --define codl=false \
    --define buildlib=false \
    --cpu=arm64-v8a
  cp bazel-bin/test/fucheng/tensor_buffer_gpu_test \
    $BIN_DIR/arm64-v8a/cpu_gpu/tensor_buffer_gpu_test

  echo "BIN PATH: $BIN_DIR"
}

#build_tensor_buffer_test

#build_tensor_buffer_gpu_test
