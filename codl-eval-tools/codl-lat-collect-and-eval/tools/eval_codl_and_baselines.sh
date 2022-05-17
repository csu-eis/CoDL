#!/bin/bash

ADB_DEVICE=$1

python test/op_chain_adb_run_test.py \
  --dl_model=all \
  --exec_type=all \
  --rounds=1 \
  --adb_device=$ADB_DEVICE
