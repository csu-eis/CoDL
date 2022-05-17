#!/bin/bash

ADB_DEVICE=$1

MACE_RUN_TEST_PATH=/data/local/tmp/mace_run_test

adb -s $ADB_DEVICE shell mkdir -p /data/local/tmp/mace_run_test

adb -s $ADB_DEVICE push \
  bazel-bin/third_party/TinyML/LinearRegression_main \
  $MACE_RUN_TEST_PATH

cp third_party/TinyML/dataset/test_x_3_6000.csv third_party/TinyML/dataset/train_x.csv
cp third_party/TinyML/dataset/test_y_3_6000.csv third_party/TinyML/dataset/train_y.csv

adb -s $ADB_DEVICE push \
  third_party/TinyML/dataset/train_x.csv \
  $MACE_RUN_TEST_PATH

adb -s $ADB_DEVICE push \
  third_party/TinyML/dataset/train_y.csv \
  $MACE_RUN_TEST_PATH
