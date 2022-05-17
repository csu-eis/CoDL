#!/bin/bash

ADB_DEVICE=$1

if [ "$ADB_DEVICE" == "" ];then
  ADB_DEV_FLAG=""
else
  ADB_DEV_FLAG="-s $ADB_DEVICE"
fi

BIN_DIR=build/bin
CODL_MOBILE_DIR="/data/local/tmp/codl"

push_files_by_abi() {
  ABI=$1
  if [ -d "$BIN_DIR/$ABI" ];then
    adb $ADB_DEV_FLAG shell mkdir -p $CODL_MOBILE_DIR
    adb $ADB_DEV_FLAG push $BIN_DIR/$ABI/cpu_gpu/codl_op_run $CODL_MOBILE_DIR
    adb $ADB_DEV_FLAG push $BIN_DIR/$ABI/cpu_gpu/codl_run $CODL_MOBILE_DIR
  fi
}

push_files_by_abi "arm64-v8a"
