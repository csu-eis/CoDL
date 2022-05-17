#!/bin/bash

ADB_SHELL="adb shell"

MOBILE_PATH="/data/local/tmp/mace_run_test"

OP_CHAIN_RUN="$MOBILE_PATH/conv2d_chain_run"

CMD="$OP_CHAIN_RUN \
        --test=yolo_v2_chain_search \
        --compute"

$ADB_SHELL $CMD
