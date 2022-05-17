#!/bin/bash

set -e

BIN_DIR=build/bin

mkdir -p $BIN_DIR

rm -rf $BIN_DIR/arm64-v8a

mkdir -p $BIN_DIR/arm64-v8a/cpu_gpu

build_cpp_json_example() {
  echo "build cpp json example for arm64-v8a + cpu_gpu"

  bazel build \
    --config android \
    --config optimization \
    test/fucheng:cpp_json_example \
    --define neon=true \
    --define opencl=true \
    --define quantize=true \
    --define rpcmem=false \
    --define buildlib=false \
    --cpu=arm64-v8a
  
  cp bazel-bin/test/fucheng/cpp_json_example \
    $BIN_DIR/arm64-v8a/cpu_gpu/cpp_json_example

  echo "BIN PATH: $BIN_DIR"
}

build_cpp_json_example
