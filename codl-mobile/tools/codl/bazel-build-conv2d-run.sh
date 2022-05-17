#!/bin/bash

set -e

BIN_DIR=build/bin

mkdir -p $BIN_DIR

rm -rf $BIN_DIR/arm64-v8a

mkdir -p $BIN_DIR/arm64-v8a/cpu_gpu

build_conv2d_run() {
  echo "build conv2d run for arm64-v8a + cpu_gpu"

  bazel build --config android --config optimization test/codlconv2drun:conv2d_run \
    --define neon=true \
    --define opencl=true \
    --define quantize=true \
    --define rpcmem=false \
    --define codl=true \
    --define buildlib=false \
    --cpu=arm64-v8a
  cp bazel-bin/test/codlconv2drun/conv2d_run $BIN_DIR/arm64-v8a/cpu_gpu/conv2d_run

  echo "BIN PATH: $BIN_DIR"
}

build_conv2d_run
