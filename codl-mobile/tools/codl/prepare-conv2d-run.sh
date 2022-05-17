#!/bin/bash

CONV2D_RUN_BUILD_PATH=build/bin/arm64-v8a/cpu_gpu
CODL_RUN_TEST_PATH=/data/local/tmp/codl_run_test

# Create a directory for test.
adb shell mkdir -p $CODL_RUN_TEST_PATH
# Push.
adb push $CONV2D_RUN_BUILD_PATH/conv2d_run $CODL_RUN_TEST_PATH
# Change mode for execution.
adb shell chmod +x $CODL_RUN_TEST_PATH/conv2d_run

echo "Prepare OK"
