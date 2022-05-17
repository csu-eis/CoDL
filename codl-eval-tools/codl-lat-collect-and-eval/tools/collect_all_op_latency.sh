#!/bin/bash

SOC=$1
ADB_DEVICE=$2
STEP=$3

# For muLayer.
if [ "$STEP" == "1.1" ] || [ "$STEP" == "" ];then
  echo "[INFO] 1.1 muLayer Conv2D"
  bash tools/collect_op_latency.sh Conv2D-OP $SOC $ADB_DEVICE CPU+GPU buffer
fi

if [ "$STEP" == "1.2" ] || [ "$STEP" == "" ];then
  echo "[INFO] 1.2 muLayer FullyConnected"
  bash tools/collect_op_latency.sh FullyConnected-OP $SOC $ADB_DEVICE CPU+GPU buffer
fi

if [ "$STEP" == "1.3" ] || [ "$STEP" == "" ];then
  echo "[INFO] 1.3 muLayer Pooling"
  bash tools/collect_op_latency.sh Pooling-OP $SOC $ADB_DEVICE CPU+GPU buffer
fi

# For CoDL.
if [ "$STEP" == "2.1" ] || [ "$STEP" == "" ];then
  echo "[INFO] 2.1 CoDL Data Sharing"
  bash tools/collect_op_latency.sh DataSharing $SOC $ADB_DEVICE CPU+GPU image
fi

if [ "$STEP" == "2.2" ] || [ "$STEP" == "" ];then
  echo "[INFO] 2.2 CoDL Conv2D"
  bash tools/collect_op_latency.sh Conv2D $SOC $ADB_DEVICE CPU+GPU image
fi

if [ "$STEP" == "2.3" ] || [ "$STEP" == "" ];then
  echo "[INFO] 2.3 CoDL FullyConnected"
  bash tools/collect_op_latency.sh FullyConnected $SOC $ADB_DEVICE CPU+GPU image
fi

if [ "$STEP" == "2.4" ] || [ "$STEP" == "" ];then
  echo "[INFO] 2.4 CoDL Pooling"
  bash tools/collect_op_latency.sh Pooling $SOC $ADB_DEVICE CPU+GPU image
fi

# For both.
if [ "$STEP" == "3.1" ] || [ "$STEP" == "" ];then
  echo "[INFO] 3.1 Remove timestamp from the file name"
  python tools/remove_timestamp.py --data_dir=lat_datasets/$SOC
fi

if [ "$STEP" == "3.2" ] || [ "$STEP" == "" ];then
  echo "[INFO] 3.2 Combine latency data of Winograd"
  python tools/combine_winograd_latency_data.py --data_dir=lat_datasets/$SOC
fi

echo "Done!"
