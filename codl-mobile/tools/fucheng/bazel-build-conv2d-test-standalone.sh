#!/bin/bash

set -e

LIB_DIR=build/lib
INC_DIR=build/include

mkdir -p $LIB_DIR
mkdir -p $INC_DIR

# Copy INC files.
cp test/fucheng/conv2d_test.h $INC_DIR/

# Remove and create LIB directory.
rm -rf $LIB_DIR/arm64-v8a
mkdir -p $LIB_DIR/arm64-v8a/cpu_gpu

echo "build conv2d gpu test library for arm64-v8a + cpu_gpu"

bazel build \
  --config android \
  --config optimization \
  test/fucheng:libconv2d_test_dynamic \
  --define neon=true \
  --define opencl=true \
  --define quantize=true \
  --define rpcmem=false \
  --define buildlib=true \
  --cpu=arm64-v8a

cp bazel-bin/test/fucheng/libconv2d_test.so $LIB_DIR/arm64-v8a/cpu_gpu/

echo "LIB PATH: $LIB_DIR"
echo "INC PACH: $INC_DIR"
