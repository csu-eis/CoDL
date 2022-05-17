#!/bin/bash

ADB_SHELL="adb shell"

MOBILE_PATH="/data/local/tmp/mace_run_test"

MACE_RUN="$MOBILE_PATH/mace_run"

CMD="LD_LIBRARY_PATH=/data/local/tmp/vendor/lib64/ \
     ADSP_LIBRARY_PATH=/data/local/tmp/vendor/lib/rfsa/adsp/ \
     $MACE_RUN \
        --model_file=/storage/emulated/0/mace_workspace/models/bert.pb \
        --model_data_file=/storage/emulated/0/mace_workspace/models/bert.data \
        --input_node=label_id,segment_ids,input_mask,input_ids,segment_ids_1,input_mask_1,input_ids_1 \
        --input_shape=1,1:1,256:1,256:1,256:1,256:1,256:1,256 \
        --input_file=$MOBILE_PATH/input_data/rdata_299x299x3 \
        --output_node=loss/Softmax \
        --output_shape=1,2 \
        --output_file=$MOBILE_PATH/mace_output \
        --device=GPU \
        --cpu_affinity_policy=1 \
        --omp_num_threads=4 \
        --gpu_perf_hint=0 \
        --gpu_priority_hint=0 \
        --gpu_memory_type=0 \
        --round=1 \
        --restart_round=1 \
        --partition_dim=4 \
        --partition_ratio=1.0"

$ADB_SHELL $CMD
