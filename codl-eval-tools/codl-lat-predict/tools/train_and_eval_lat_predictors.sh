#!/bin/bash

DATA_DIR=$1
SOC_NAME=$2

# CoDL.
bash tools/train_and_eval_codl_lat_predictors.sh $DATA_DIR $SOC_NAME

# muLayer.
bash tools/train_and_eval_mulayer_lat_predictors.sh $DATA_DIR $SOC_NAME
