#!/bin/bash

set -e

BIN_DIR=build/bin

mkdir -p $BIN_DIR

rm -rf $BIN_DIR/arm64-v8a

mkdir -p $BIN_DIR/arm64-v8a/cpu_gpu

build_test() {
  target_parent_path=$1
  target_name=$2

  bazel_target=$target_parent_path:$target_name
  target_path=$target_parent_path/$target_name

  echo "build $target_name for arm64-v8a + cpu_gpu"

  bazel build \
    --config android \
    --config optimization \
    $bazel_target \
    --define neon=true \
    --define opencl=true \
    --define quantize=true \
    --define rpcmem=false \
    --define codl=false \
    --define buildlib=false \
    --cpu=arm64-v8a
  
  cp bazel-bin/$target_path $BIN_DIR/arm64-v8a/cpu_gpu/$target_name

  echo "BIN PATH: $BIN_DIR"
}

build_gemmlowp_test() {
  target_parent_path=third_party/gemmlowp
  target_name=gemmlowp_run

  build_test $target_parent_path $target_name
}

build_gemmlowp_benchmark() {
  target_parent_path=third_party/gemmlowp
  target_name=benchmark

  build_test $target_parent_path $target_name
}

build_gemmlowp_test

build_gemmlowp_benchmark
