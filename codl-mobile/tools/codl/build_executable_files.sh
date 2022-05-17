#!/bin/bash

ABI=$1

if [ "$ABI" == "" ];then
  ABI="arm64-v8a"
fi

bash tools/fucheng/bazel-build-codl-run.sh $ABI
