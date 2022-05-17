#!/bin/bash

ADB_SHELL="adb shell"
CODL_RUN=/data/local/tmp/codl_run_test/conv2d_run

run_matmul_on_mobile() {
  DEVICE=$1
  GPU_MEM_TYPE=$2
  TRANSPOSE_B=$3
  
  if [ "$DEVICE" == "CPU" ];then
    PART_RATIO=0.0
    CU_HINT=1
  fi

  if [ "$DEVICE" == "GPU" ];then
    PART_RATIO=1.0
    CU_HINT=2
  fi

  if [ "$GPU_MEM_TYPE" == "IMAGE" ];then
    GPU_MEM_TYPE=0
  fi

  if [ "$GPU_MEM_TYPE" == "BUFFER" ];then
    GPU_MEM_TYPE=1
  fi

  CMD="MACE_DURATION_COLLECT_GRANULARITY=Coarse \
       MACE_DO_COMPUTE=1 \
       $CODL_RUN \
       --compute \
       --cpu_affinity_policy=1 \
       --rounds=21 \
       --gpu_memory_type=$GPU_MEM_TYPE \
       --num_threads=4 \
       --part_dim=4 \
       --part_ratio=$PART_RATIO \
       --cu_hint=$CU_HINT \
       --op_type=MatMul \
       --input_shape=$LHS_SHAPE \
       --weight_shape=$RHS_SHAPE \
       --$TRANSPOSE_B"

  $ADB_SHELL $CMD
}

# rank = 2, no transpose b
run_eval_example1() {
  LHS_SHAPE=8,16
  RHS_SHAPE=16,16

  run_matmul_on_mobile "CPU" "BUFFER"
  run_matmul_on_mobile "GPU" "BUFFER"
  run_matmul_on_mobile "GPU" "IMAGE"
}

# rank = 4, no transpose b
run_eval_example2() {
  LHS_SHAPE=1,16,8,16
  RHS_SHAPE=1,16,16,16

  run_matmul_on_mobile "CPU" "BUFFER"
  run_matmul_on_mobile "GPU" "BUFFER"
  run_matmul_on_mobile "GPU" "IMAGE"
}

# rank = 4, transpose b
run_eval_example3() {
  LHS_SHAPE=1,16,8,16
  RHS_SHAPE=1,16,16,16

  run_matmul_on_mobile "CPU" "BUFFER" "transpose_b"
  run_matmul_on_mobile "GPU" "BUFFER" "transpose_b"
  run_matmul_on_mobile "GPU" "IMAGE" "transpose_b"
}

run_eval_example3
