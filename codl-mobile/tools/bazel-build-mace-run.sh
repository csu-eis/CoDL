#!/bin/bash

set -e

BIN_DIR=build/bin

mkdir -p $BIN_DIR

rm -rf $BIN_DIR/arm64-v8a

mkdir -p $BIN_DIR/arm64-v8a/cpu_gpu

echo "build mace run for arm64-v8a + cpu_gpu_hexagon"

bazel build \
  --config android \
  --config optimization \
  mace/tools:mace_run_static \
  --define neon=true \
  --define opencl=true \
  --define quantize=true \
  --define hexagon=true \
  --define rpcmem=false \
  --define codl=true \
  --cpu=arm64-v8a

cp bazel-bin/mace/tools/mace_run_static $BIN_DIR/arm64-v8a/cpu_gpu/mace_run

echo "BIN PATH: $BIN_DIR"
