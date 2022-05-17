#!/bin/bash

set -e

BIN_DIR=build/bin

mkdir -p $BIN_DIR

rm -rf $BIN_DIR/arm64-v8a

mkdir -p $BIN_DIR/arm64-v8a/cpu_gpu

build_conv2d_latency_prediction_test() {
  echo "build conv2d latency prediction test for arm64-v8a + cpu_gpu"

  bazel build \
    --config android \
    --config optimization \
    test/fucheng:conv2d_latency_prediction_test \
    --define neon=true \
    --define opencl=true \
    --define quantize=true \
    --define rpcmem=false \
    --define codl=true \
    --define buildlib=false \
    --cpu=arm64-v8a
  
  cp bazel-bin/test/fucheng/conv2d_latency_prediction_test \
    $BIN_DIR/arm64-v8a/cpu_gpu/conv2d_latency_prediction_test

  echo "BIN PATH: $BIN_DIR"
}

adb_push_conv2d_latency_prediction_test() {
  adb push $BIN_DIR/arm64-v8a/cpu_gpu/conv2d_latency_prediction_test \
    /data/local/tmp/mace_run_test
}

build_conv2d_latency_prediction_test

adb_push_conv2d_latency_prediction_test
