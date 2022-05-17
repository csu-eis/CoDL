#!/bin/sh

# bash tools/fucheng/mace-build-fucheng-script.sh 1 push c3aaae61

TAG_INFO="[INFO]"

PlayRingFunc() {
  #echo -e "\a"
  paplay /usr/share/sounds/ubuntu/notifications/Positive.ogg
}

CountDownFunc() {
  count_down=$1
  for((i=$count_down;i>=1;i--));
  do
    echo -e -n "\r$TAG_INFO $i"
    sleep 1s
  done
}

BUILD_PATH=build/bin/arm64-v8a/cpu_gpu

AdbPushAndChmod() {
  ADB_OPTION=$1
  DIRNAME=$2
  FILENAME=$3

  echo "$TAG_INFO adb $ADB_OPTION push $BUILD_PATH/$FILENAME to $DIRNAME/$FILENAME"
  adb $ADB_OPTION push $BUILD_PATH/$FILENAME $DIRNAME/$FILENAME
  echo "$TAG_INFO adb $ADB_OPTION shell chmod $DIRNAME/$FILENAME"
  adb $ADB_OPTION shell chmod +x $DIRNAME/$FILENAME
}

if [ $# == 0 ];then
  read -p "$TAG_INFO Target (1-mace_run/2-standalone_lib/3-codl_run/4-codl_run_standalone_lib): " target_idx
fi

target_idx="3"
enable_push="nopush"
adb_device="none"

if [ $# -ge 1 ];then
  target_idx=$1
fi

if [ $# -ge 2 ];then
  enable_push=$2
fi

if [ $# -ge 3 ];then
  adb_device=$3
fi

if [ "$target_idx" == "1" ]
then
  BUILD_TARGET="mace_run"
fi

if [ "$target_idx" == "2" ]
then
  BUILD_TARGET="standalone_lib"
fi

if [ "$target_idx" == "11" -o "$target_idx" == "codl_run" ]
then
  BUILD_TARGET="codl_run"
fi

if [ "$target_idx" == "12" ]
then
  BUILD_TARGET="codl_run_standalone_lib"
fi

if [ "$target_idx" == "13" ]
then
  BUILD_TARGET="conv2d_partition_prediction_test"
fi

if [ "$target_idx" == "21" ]
then
  BUILD_TARGET="gemm_test"
fi

if [ "$target_idx" == "22" ]
then
  BUILD_TARGET="gemmlowp_test"
fi

if [ "$target_idx" == "31" ]
then
  BUILD_TARGET="cpp_json_example"
fi

if [ "$target_idx" == "41" ]
then
  BUILD_TARGET="random_forest_model_test"
fi

#PlayRingFunc

# 1. mace_run
# 2. standalone_lib
# 3. codl_run
# 4. codl_run_standalone_lib
#BUILD_TARGET="standalone_lib"

echo "$TAG_INFO Build $BUILD_TARGET"

CountDownFunc 0

echo -e -n "\r"

DATA_LOCAL_TMP_PATH=/data/local/tmp
MACE_RUN_PATH=$DATA_LOCAL_TMP_PATH/mace_run
MACE_RUN_TEST_PATH=$DATA_LOCAL_TMP_PATH/mace_run_test
CODL_RUN_TEST_PATH=$DATA_LOCAL_TMP_PATH/codl_run_test
CODL_PATH=$DATA_LOCAL_TMP_PATH/codl

# Xiaomi 9 (SDM855): c3aaae61
# Hikey 960: 0123456789ABCDEF
# Honor 9 (Kirin 960): CSX4C17A26004675
ADB_DEVICE=$adb_device
ADB_OPTION="-s $ADB_DEVICE"

if [ "$ADB_DEVICE" == "none" ];then
  ADB_OPTION=""
fi

# MACE Run
if [ "$BUILD_TARGET" = "mace_run" ];then
  bash tools/bazel-build-mace-run.sh
  if [ "$enable_push" = "push" ];then
    AdbPushAndChmod "$ADB_OPTION" $MACE_RUN_TEST_PATH mace_run
  fi
fi

if [ "$BUILD_TARGET" = "standalone_lib" ];then
  bash tools/bazel-build-standalone-lib.sh
fi

if [ "$BUILD_TARGET" = "codl_run" ];then
  bash tools/fucheng/bazel-build-codl-run.sh
  if [ "$enable_push" = "push" ];then
    AdbPushAndChmod "$ADB_OPTION" $CODL_PATH codl_op_run
    AdbPushAndChmod "$ADB_OPTION" $CODL_PATH codl_run
  fi
fi

if [ "$BUILD_TARGET" = "codl_run_standalone_lib" ];then
  bash tools/fucheng/bazel-build-conv2d-test-standalone.sh
fi

if [ "$BUILD_TARGET" = "conv2d_partition_prediction_test" ];then
  bash tools/fucheng/bazel-build-conv2d-partition-prediction-test.sh
  
  AdbPushAndChmod "$ADB_OPTION" $MACE_RUN_TEST_PATH conv2d_partition_prediction_test
fi

if [ "$BUILD_TARGET" = "gemm_test" ];then
  bash tools/fucheng/bazel-build-gemm-test.sh
  
  AdbPushAndChmod "$ADB_OPTION" $MACE_RUN_TEST_PATH gemm_test
fi

if [ "$BUILD_TARGET" = "gemmlowp_test" ];then
  bash tools/fucheng/bazel-build-gemmlowp-test.sh

  AdbPushAndChmod "$ADB_OPTION" $MACE_RUN_TEST_PATH gemmlowp_run

  AdbPushAndChmod "$ADB_OPTION" $MACE_RUN_TEST_PATH benchmark
fi

if [ "$BUILD_TARGET" = "cpp_json_example" ];then
  bash tools/fucheng/bazel-build-cpp-json-example.sh
  
  AdbPushAndChmod "$ADB_OPTION" $MACE_RUN_TEST_PATH cpp_json_example
fi

if [ "$BUILD_TARGET" = "random_forest_model_test" ];then
  bash tools/fucheng/bazel-build-random-forest-model-test.sh

  AdbPushAndChmod "$ADB_OPTION" $MACE_RUN_TEST_PATH random_forest_model_test
fi

echo "$TAG_INFO ADB -s $ADB_DEVICE remove mace_cl_compiled_program.bin"
adb $ADB_OPTION shell \
  rm -f $MACE_RUN_PATH/interior/mace_cl_compiled_program.bin

#PlayRingFunc
