#/bin/bash

CODL_RUN_TEST_PATH=/data/local/tmp/codl_run_test
CONV2D_RUN=$CODL_RUN_TEST_PATH/conv2d_run

adb shell "$CONV2D_RUN -c 1 -t 4 -m 0 -w 0 -u 0 -d 1 -r 1.0 -o 20 -s \"1,224,224,3;64,3,3,3;1,1\" -p 1"
