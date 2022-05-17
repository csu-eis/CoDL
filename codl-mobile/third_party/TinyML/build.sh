#!/bin/bash

bazel build \
  --config android \
  --config optimization \
  third_party/TinyML:LinearRegression_main \
  --cpu=arm64-v8a
