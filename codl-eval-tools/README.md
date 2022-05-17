
## Introduction

This repository is the evaluation tools of CoDL. The tools work on a PC for collecting the latency data from a mobile device, training the latency prediction models, and evaluating the performance of CoDL and baselines on the mobile device.

## Requirements

|NAME|REQUIREMENT|
|:-|:-|
|python|>=3.6|
|numpy|>=1.19.2|
|pandas|>=1.0.5|
|scipy|>=1.5.2|
|scikit-learn|>=0.24.2|
|rich|>=3.3.1|
|kiwisolver|>=1.3.1|
|matplotlib|>=3.1.1|
|confuse|>=1.5.0|

If you use the Dockerfile to setup, these requirements are already installed. Otherwise, you should install the requirements as follows:

```shell
# For pip.
pip install -r requirements.txt

# For anaconda.
bash conda_install_requirements.sh
```

## Instruction

### Step 0: Prepare

Connect a mobile device to a PC with a USB cable or ADB Wi-Fi.

### Step 1: Collect latency data

We provide a script to collect all required latency data from the mobile device. This step will take one or two hours.

```shell
cd /path/to/codl-eval-tools/codl-lat-collect-and-eval/

# soc_name is one of [sdm855, sdm865, sdm888, kirin990]. Other SoCs are not currently supported.
# adb_device is the adb-assigned serial number of mobile device obtained by adb devices.
bash tools/collect_all_op_latency.sh soc_name adb_device
```

Note that the collected data is stored as CSV files in the directory `codl-lat-collect-and-eval/lat_datasets/soc_name`.

### Step 2: Train and evaluate latency prediction models

We provide a script to train the latency prediciton models of both CoDL and $\mu$Layer, and evaluate the accuracy of them as follows:

```shell
cd /path/to/codl-eval-tools/codl-lat-predict/

bash tools/train_and_eval_lat_predictors.sh /path/to/lat_datasets soc_name
```

The output of this step is the accuracy of latency prediciton models as follows:

```shell
[INFO] CoDL, DataSharing
[INFO] model DATA-SHARING, acc 81.25
[INFO] CoDL, Conv2D
[INFO] model DIRECT-CPU, acc 78.12
...
```

Note that the trained latency prediciton models are stored as JSON files in the directory `codl-lat-predict/saved_models/soc_name`.

### Step 3: Push latency prediction models and configure files

We provide a script to push the trained latency prediction models and configure files to the mobile device as follows:

```shell
cd /path/to/codl-eval-tools/codl-lat-collect-and-eval/

bash tools/push_lat_predictors_and_configs.sh \
  soc_name \
  /path/to/saved_models \
  /path/to/configs \
  adb_device
```

We provide predifined configure files in the directory `codl-lat-collect-and-eval/configs/` for the SoC platforms we tested. We are working on building the configure files for other SoC paltforms in an easy way.

Note that we push the latency prediction models and configure files to `/data/local/tmp/codl/saved_prediction_models` and `/data/local/tmp/codl/configs`, respectively.

### Step 4: Evaluate performance of CoDL and baselines

We provide a script to evaluate the performance of CoDL and baselines as follows:

```shell
cd /path/to/codl-eval-tools/codl-lat-collect-and-eval/

bash tools/eval_codl_and_baselines.sh adb_device
```

The output of this step is the inference latency of deep learning models executed with CoDL and baselines as follows:

```shell
[VERBOSE] model yolo_v2, exec_type cpu, round 0, lat 213.00, ...
[VERBOSE] model yolo_v2, exec_type gpu, round 0, lat 169.31, ...
[VERBOSE] model yolo_v2, exec_type mulayer_buffer_based, round 0, lat 324.06, ...
[VERBOSE] model yolo_v2, exec_type codl_buffer_based, round 0, lat 272.37, ...
[VERBOSE] model yolo_v2, exec_type codl, round 0, lat 131.93, ...
...
```
