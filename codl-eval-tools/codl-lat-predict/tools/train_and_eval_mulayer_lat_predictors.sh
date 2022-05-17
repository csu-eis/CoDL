#!/bin/bash

DATA_DIR=$1
SOC_NAME=$2

#OUTPUT_PATH_FLAG=""
OUTPUT_PATH_FLAG="--output_path=saved_models/$SOC_NAME"

echo "[INFO] muLayer, Conv2D"
python flops_lr_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_mulayer_conv2d_cpu.csv \
  --op_type=Conv2D \
  --device=CPU \
  $OUTPUT_PATH_FLAG

python flops_lr_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_mulayer_conv2d_gpu.csv \
  --op_type=Conv2D \
  --device=GPU \
  $OUTPUT_PATH_FLAG

echo "[INFO] muLayer, FullyConnected"
python flops_lr_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_mulayer_fc_cpu.csv \
  --op_type=FullyConnected \
  --device=CPU \
  $OUTPUT_PATH_FLAG

python flops_lr_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_mulayer_fc_gpu.csv \
  --op_type=FullyConnected \
  --device=GPU \
  $OUTPUT_PATH_FLAG

echo "[INFO] muLayer, Pooling"
python flops_lr_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_mulayer_pooling_cpu.csv \
  --op_type=Pooling \
  --device=CPU \
  $OUTPUT_PATH_FLAG

python flops_lr_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_mulayer_pooling_gpu.csv \
  --op_type=Pooling \
  --device=GPU \
  $OUTPUT_PATH_FLAG
