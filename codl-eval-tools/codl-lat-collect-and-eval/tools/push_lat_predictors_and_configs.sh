#!/bin/bash

SOC_NAME=$1
SAVED_MODELS_PATH=$2
DEV_CONFIGS_PATH=$3
ADB_DEVICE=$4

CODL_MOBILE_PATH=/data/local/tmp/codl
SAVED_MODELS_MOBILE_PATH=$CODL_MOBILE_PATH/saved_prediction_models
CONFIG_MOBILE_PATH=$CODL_MOBILE_PATH/configs

push_prediction_models() {
  adb -s $ADB_DEVICE shell mkdir -p $SAVED_MODELS_MOBILE_PATH
  #adb -s $ADB_DEVICE shell rm $SAVED_MODELS_MOBILE_PATH/*.json
  adb -s $ADB_DEVICE push $SAVED_MODELS_PATH/$SOC_NAME/*.json $SAVED_MODELS_MOBILE_PATH
}

push_configs() {
  adb -s $ADB_DEVICE shell mkdir -p $CONFIG_MOBILE_PATH
  #adb -s $ADB_DEVICE shell rm $CONFIG_MOBILE_PATH/*.json
  adb -s $ADB_DEVICE push $DEV_CONFIGS_PATH/$SOC_NAME/config_codl.json $CONFIG_MOBILE_PATH
  adb -s $ADB_DEVICE push $DEV_CONFIGS_PATH/$SOC_NAME/config_mulayer.json $CONFIG_MOBILE_PATH
}

push_prediction_models

push_configs
