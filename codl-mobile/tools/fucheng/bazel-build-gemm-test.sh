#!/bin/bash

set -e

BIN_DIR=build/bin

mkdir -p $BIN_DIR

rm -rf $BIN_DIR/arm64-v8a

mkdir -p $BIN_DIR/arm64-v8a/cpu_gpu

build_gemm_test() {
  echo "build gemm test for arm64-v8a + cpu_gpu"

  bazel build \
    --config android \
    --config optimization \
    test/fucheng:gemm_test \
    --define neon=true \
    --define opencl=true \
    --define quantize=true \
    --define rpcmem=false \
    --define buildlib=false \
    --cpu=arm64-v8a
  
  cp bazel-bin/test/fucheng/gemm_test $BIN_DIR/arm64-v8a/cpu_gpu/gemm_test

  echo "BIN PATH: $BIN_DIR"
}

build_gemm_test
