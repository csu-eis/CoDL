#!/bin/bash

DATA_DIR=$1
SOC_NAME=$2

#OUTPUT_PATH_FLAG=""
OUTPUT_PATH_FLAG="--output_path=saved_models/$SOC_NAME"

echo "[INFO] CoDL, DataSharing"
python ka_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_data_sharing.csv \
  --dataset=DATA-SHARING \
  --frac=0.7 \
  $OUTPUT_PATH_FLAG

echo "[INFO] CoDL, Conv2D"
python ka_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_conv2d_cpu_direct.csv \
  --dataset=CONV2D-DIRECT-CPU \
  --frac=0.7 \
  $OUTPUT_PATH_FLAG

python ka_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_conv2d_cpu_gemm.csv \
  --dataset=GEMM-TOTAL-CPU \
  --frac=0.7 \
  $OUTPUT_PATH_FLAG

python ka_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_conv2d_cpu_winograd_combined.csv \
  --dataset=WINOGRAD-TOTAL-CPU \
  --frac=0.7 \
  $OUTPUT_PATH_FLAG

python ka_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_conv2d_gpu_direct.csv \
  --dataset=CONV2D-DIRECT-GPU \
  --frac=0.7 \
  $OUTPUT_PATH_FLAG

echo "[INFO] CoDL, FullyConnected"
python ka_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_fc_cpu_gemv.csv \
  --dataset=GEMV-CPU \
  --frac=0.7 \
  $OUTPUT_PATH_FLAG

python ka_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_fc_gpu_direct.csv \
  --dataset=FC-DIRECT-GPU \
  --frac=0.7 \
  $OUTPUT_PATH_FLAG

echo "[INFO] CoDL, Pooling"
python ka_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_pooling_cpu_direct_max.csv \
  --dataset=POOLING-CPU \
  --frac=0.7 \
  $OUTPUT_PATH_FLAG

python ka_fit.py \
  --data=$DATA_DIR/$SOC_NAME/t_pooling_gpu_direct_max.csv \
  --dataset=POOLING-GPU \
  --frac=0.7 \
  $OUTPUT_PATH_FLAG
