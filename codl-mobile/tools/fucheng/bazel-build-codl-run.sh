#!/bin/bash

ABI=$1

set -e

BIN_DIR=build/bin

PREBUILD_BIN_DIR=prebuild/bin

mkdir -p $BIN_DIR
mkdir -p $PREBUILD_BIN_DIR

rm -rf $BIN_DIR/$ABI
rm -rf $PREBUILD_BIN_DIR/$ABI

mkdir -p $BIN_DIR/$ABI/cpu_gpu
mkdir -p $PREBUILD_BIN_DIR/$ABI/cpu_gpu

build_conv2d_test() {
  echo "build conv2d test for $ABI + cpu_gpu"

  bazel build \
    --config android \
    --config optimization \
    test/fucheng:conv2d_test \
    --define neon=true \
    --define opencl=true \
    --define quantize=true \
    --define rpcmem=false \
    --define codl=false \
    --define buildlib=false \
    --cpu=$ABI
  
  cp bazel-bin/test/fucheng/conv2d_test $BIN_DIR/$ABI/cpu_gpu/conv2d_test

  echo "BIN PATH: $BIN_DIR"
}

build_codl_op_run() {
  echo "build codl op run for $ABI + cpu_gpu"

  bazel build \
    --config android \
    --config optimization \
    test/codl_run:codl_op_run \
    --define neon=true \
    --define opencl=true \
    --define quantize=true \
    --define rpcmem=false \
    --define codl=true \
    --define buildlib=false \
    --cpu=$ABI
  
  cp bazel-bin/test/codl_run/codl_op_run $BIN_DIR/$ABI/cpu_gpu/codl_op_run
  cp bazel-bin/test/codl_run/codl_op_run $PREBUILD_BIN_DIR/$ABI/cpu_gpu/codl_op_run

  echo "BIN PATH: $BIN_DIR"
}

build_codl_run() {
  echo "build codl run for $ABI + cpu_gpu"

  bazel build \
    --config android \
    --config optimization \
    test/codl_run:codl_run \
    --define neon=true \
    --define opencl=true \
    --define quantize=true \
    --define rpcmem=false \
    --define codl=true \
    --define buildlib=false \
    --cpu=$ABI
  
  cp bazel-bin/test/codl_run/codl_run $BIN_DIR/$ABI/cpu_gpu/codl_run
  cp bazel-bin/test/codl_run/codl_run $PREBUILD_BIN_DIR/$ABI/cpu_gpu/codl_run

  echo "BIN PATH: $BIN_DIR"
}

#build_conv2d_test

build_codl_op_run

build_codl_run
