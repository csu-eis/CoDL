#!/bin/bash

OP_TYPE=$1
SOC=$2
ADB_DEVICE=$3
DEVICE=$4
MEM_TYPE=$5

DATA_TRANSFORM_FLAG=""

if [ "$OP_TYPE" == "DataSharing" ];then
  OP_TYPE=Conv2D
  INPUT_DATASET=op_datasets/dataset_conv2d_large_cnn.csv
  OUTPUT_DATASET_TYPE=MLConv2d
  DATA_TRANSFORM_FLAG="--data_transform"
fi

if [ "$OP_TYPE" == "Conv2D" ];then
  INPUT_DATASET=op_datasets/dataset_conv2d_large_cnn.csv
  OUTPUT_DATASET_TYPE=MLConv2d
fi

if [ "$OP_TYPE" == "Conv2D-OP" ];then
  OP_TYPE=Conv2D
  INPUT_DATASET=op_datasets/dataset_conv2d_large_cnn.csv
  #OUTPUT_DATASET_TYPE=MLConv2dOp
  OUTPUT_DATASET_TYPE=MuLayerLRConv2d
fi

if [ "$OP_TYPE" == "FullyConnected" ];then
  INPUT_DATASET=op_datasets/dataset_fully_connected_cnn.csv
  OUTPUT_DATASET_TYPE=MLFullyConnected
fi

if [ "$OP_TYPE" == "FullyConnected-OP" ];then
  OP_TYPE=FullyConnected
  INPUT_DATASET=op_datasets/dataset_fully_connected_cnn.csv
  #OUTPUT_DATASET_TYPE=MLFullyConnectedOp
  OUTPUT_DATASET_TYPE=MuLayerLRFullyConnected
fi

if [ "$OP_TYPE" == "Pooling" ];then
  INPUT_DATASET=op_datasets/dataset_pooling_large_cnn.csv
  OUTPUT_DATASET_TYPE=MLPooling
fi

if [ "$OP_TYPE" == "Pooling-OP" ];then
  OP_TYPE=Pooling
  INPUT_DATASET=op_datasets/dataset_pooling_large_cnn.csv
  #OUTPUT_DATASET_TYPE=MLPooling
  OUTPUT_DATASET_TYPE=MuLayerLRPooling
fi

# --op_type: [Conv2D, FullyConnected, Pooling]
# --output_dataset_type: [MLConv2d, MLConv2dOp,
#                         MLFullyConnected, MLFullyConnectedOp,
#                         MLPooling,
#                         MLMatMul]
# --soc: [sdm855, sdm865, sdm888, kirin960, kirin990]
# --device: [CPU, GPU, CPU+GPU]
# --write_data

PYTHON=python3

$PYTHON measure.py \
  $DATA_TRANSFORM_FLAG \
  --op_type=$OP_TYPE \
  --input_dataset=$INPUT_DATASET \
  --output_dataset_type=$OUTPUT_DATASET_TYPE \
  --output_dataset_path=lat_datasets/$SOC \
  --soc=$SOC \
  --num_threads=4 \
  --gpu_mem_type=$MEM_TYPE \
  --adb_device=$ADB_DEVICE \
  --device=$DEVICE \
  --write_data
