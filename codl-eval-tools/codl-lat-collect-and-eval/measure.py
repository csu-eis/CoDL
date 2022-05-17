
import time
import os
import sys
import argparse
import pandas as pd
from utils.common.log import *
from utils.common.time import *
from utils.common.devfreq import *
from utils.common.basic_type import *
from utils.soc.soc_registry import RegistrySoc
from utils.op.op_type import *
from utils.op.op_adb_utils import *
from utils.op.conv2d_common_calc import *
from utils.op.conv2d_param_utils import *
from utils.op.pooling_common_calc import *
from utils.op.pooling_param_utils import *
from utils.op.fully_connected_common_calc import *
from utils.op.fully_connected_param_utils import *
from utils.op.deconv2d_common_calc import *
from utils.op.deconv2d_param_utils import *
from utils.op.matmul_common_calc import *
from utils.op.matmul_param_utils import *
from utils.dataset.dataset_info import *
from utils.dataset.ml_dataset_utils import *
from core.measure_base import *
from core.measure_conv2d_base import *
#from config import SetGlobalTargetSoc, GetGlobalTargetSoc
from config import SetGlobalMobileSocName, GetGlobalMobileSocName
from config import SetGlobalThreadCount
from config import SetGlobalGpuMemoryType, GetGlobalGpuMemoryType

def StringToImplTypes(op_type, impl_types):
  if impl_types is None:
    return None
  
  out_impl_types = []

  if op_type == OpType.Conv2D:
    for t in impl_types:
      if t == 'DIRECT':
        out_impl_types.append(Conv2dCpuImplType.DIRECT)
      elif t == 'GEMM':
        out_impl_types.append(Conv2dCpuImplType.GEMM)
      elif t == 'WINOGRAD':
        out_impl_types.append(Conv2dCpuImplType.WINOGRAD)
      elif t == 'WINOGRAD_GEMM':
        out_impl_types.append(Conv2dCpuImplType.WINOGRAD_GEMM)
      else:
        raise ValueError('Unsupported implement type: ' + t)
  elif op_type == OpType.Pooling:
    for t in impl_types:
      if t == 'DIRECT_AVG':
        out_impl_types.append(PoolingCpuImplType.DIRECT_AVG)
      elif t == 'DIRECT_MAX':
        out_impl_types.append(PoolingCpuImplType.DIRECT_MAX)
      else:
        raise ValueError('Unsupported implement type: ' + t)
  elif op_type == OpType.FullyConnected:
    for t in impl_types:
      if t == 'GEMV':
        out_impl_types.append(FullyConnectedCpuImplType.GEMV)
      else:
        raise ValueError('Unsupported implement type: ' + t)
  elif op_type == OpType.Deconv2D:
    for t in impl_types:
      if t == 'DIRECT':
        out_impl_types.append(Deconv2dCpuImplType.DIRECT)
      else:
        raise ValueError('Unsupported implement type: ' + t)
  elif op_type == OpType.MatMul:
    for t in impl_types:
      if t == 'GEMM':
        out_impl_types.append(MatMulCpuImplType.GEMM)
      else:
        raise ValueError('Unsupported implement type: ' + t)

  return out_impl_types

def get_op_param_utils(op_type):
  if op_type == OpType.Conv2D:
    return Conv2dParamUtils()
  elif op_type == OpType.Pooling:
    return PoolingParamUtils()
  elif op_type == OpType.FullyConnected:
    return FullyConnectedParamUtils()
  elif op_type == OpType.Deconv2D:
    return Deconv2dParamUtils()
  elif op_type == OpType.MatMul:
    return MatMulParamUtils()
  else:
    raise ValueError('Unsupported OP type: ' + op_type)
    return None

CNN_MODEL_ALL = ['YOLO-V2', 'POSENET', 'VGG-16',
                 'RESNET-V1-50', 'MOBILENET-V1', 'MOBILENET-V2', 'SQUEEZENET',
                 'FAST_STYLE_TRANSFER', 'RETINAFACE']

CNN_MODEL_LARGE = ['YOLO-V2', 'POSENET', 'VGG-16',
                   'FAST_STYLE_TRANSFER', 'RETINAFACE']

def convert_cnn_model_names(names):
  if names is None:
    return None
  if names[0] == 'ALL':
    return None
  elif names[0] == 'LARGE':
    return CNN_MODEL_LARGE
  else:
    return names

def StringToDeviceList(device):
  if device == 'CPU':
    dev_list = [Device.CPU]
  elif device == 'GPU':
    dev_list = [Device.GPU]
  elif device == 'CPU+GPU':
    dev_list = [Device.CPU, Device.GPU]
  else:
    raise ValueError('Unsupported device configure: ' + device)
    dev_list = None

  return dev_list

def get_dataset_configure(dataset_name, dev_name_list,
                          do_data_transform, do_compute):
  PART_DIM_ALL = [PartitionDimension.HEIGHT, PartitionDimension.WIDTH,
                  PartitionDimension.IN_CHANNEL, PartitionDimension.OUT_CHANNEL]
  PART_DIM_H_OC = [PartitionDimension.HEIGHT, PartitionDimension.OUT_CHANNEL]
  PART_DIM_H = [PartitionDimension.HEIGHT]
  PART_DIM_OC = [PartitionDimension.OUT_CHANNEL]
  PART_RATIO_ALL = list(range(0, 100 // 10 + 1, 1))
  PART_RATIO_CPU_OR_GPU = [0, 1]

  if dataset_name == DatasetName.FLOPsLRConv2d:
    pdim_list = PART_DIM_OC
    pratio_list = PART_RATIO_CPU_OR_GPU
  elif dataset_name == DatasetName.MuLayerLRConv2d or \
       dataset_name == DatasetName.MuLayerLRPooling or \
       dataset_name == DatasetName.MuLayerLRFullyConnected:
    pdim_list = PART_DIM_OC
    pratio_list = PART_RATIO_CPU_OR_GPU
  elif dataset_name == DatasetName.MLConv2d or \
       dataset_name == DatasetName.MLConv2dOpImplType:
    if not do_data_transform:
      pdim_list = PART_DIM_H_OC
    else:
      dev_name_list = [Device.CPU]
      pdim_list = [PartitionDimension.HEIGHT]
      do_compute = True
    pratio_list = PART_RATIO_ALL
    pratio_list = [float (i) / (len(pratio_list) - 1) for i in pratio_list]
  elif dataset_name == DatasetName.MLConv2dOp:
    pdim_list = PART_DIM_H  # PART_DIM_H_OC
    pratio_list = PART_RATIO_CPU_OR_GPU  # PART_RATIO_ALL
    #pratio_list = [float (i) / (len(pratio_list) - 1) for i in pratio_list]
  elif dataset_name == dataset_name == DatasetName.MLPooling:
    pdim_list = PART_DIM_H
    pratio_list = PART_RATIO_ALL
    pratio_list = [float (i) / (len(pratio_list) - 1) for i in pratio_list]
  elif dataset_name == DatasetName.MLFullyConnected or \
       dataset_name == DatasetName.MLDeconv2d:
    pdim_list = PART_DIM_OC
    pratio_list = PART_RATIO_ALL
    pratio_list = [float (i) / (len(pratio_list) - 1) for i in pratio_list]
  elif dataset_name == DatasetName.MLFullyConnectedOp:
    pdim_list = PART_DIM_OC
    pratio_list = PART_RATIO_ALL
    pratio_list = [float (i) / (len(pratio_list) - 1) for i in pratio_list]
  elif dataset_name == DatasetName.MLMatMul:
    pdim_list = [PartitionDimension.M_OR_BATCH]
    pratio_list = PART_RATIO_ALL
    pratio_list = [float (i) / (len(pratio_list) - 1) for i in pratio_list]
  else:
    raise ValueError('Unknown dataset name: ' + dataset_name.name)

  return dev_name_list, pdim_list, pratio_list, do_data_transform, do_compute

def get_dataset_utils(dataset_name):
  if dataset_name == DatasetName.FLOPsLRConv2d:
    return FLOPsLRConv2dDataUtils()
  elif dataset_name == DatasetName.MuLayerLRConv2d:
    return MuLayerLRConv2dDataUtils()
  elif dataset_name == DatasetName.MuLayerLRPooling:
    return MuLayerLRPoolingDataUtils()
  elif dataset_name == DatasetName.MuLayerLRFullyConnected:
    return MuLayerLRFullyConnectedDataUtils()
  elif dataset_name == DatasetName.MLConv2d:
    return MLConv2dDataUtils()
  elif dataset_name == DatasetName.MLConv2dOp:
    return MLConv2dOpDataUtils()
  elif dataset_name == DatasetName.MLConv2dOpImplType:
    return MLConv2dOpImplTypeDataUtils()
  elif dataset_name == DatasetName.MLPooling:
    return MLPoolingDataUtils()
  elif dataset_name == DatasetName.MLFullyConnected:
    return MLFullyConnectedDataUtils()
  elif dataset_name == DatasetName.MLFullyConnectedOp:
    return MLFullyConnectedOpDataUtils()
  elif dataset_name == DatasetName.MLDeconv2d:
    return MLDeconv2dDataUtils()
  elif dataset_name == DatasetName.MLMatMul:
    return MLMatMulDataUtils()
  else:
    raise ValueError('ERROR: Unknown dataset name ' + dataset_name.name)
    return None

def get_op_impl_type(op_type, op_param, is_wino_gemm, device):
  if op_type == OpType.Conv2D:
    if device == Device.CPU:
      return GetConv2dCpuImplType(op_param['filter_shape'],
                                  op_param['strides'],
                                  is_wino_gemm)
    elif device == Device.GPU:
      return GetConv2dGpuImplType()
    else:
      raise ValueError('Unsupported device: ' + device)
      return None
  elif op_type == OpType.Pooling:
    return GetPoolingCpuImplType(op_param['pooling_type'])
  elif op_type == OpType.FullyConnected:
    return GetFullyConnectedCpuImplType()
  elif op_type == OpType.Deconv2D:
    return GetDeconv2dCpuImplType()
  elif op_type == OpType.MatMul:
    return GetMatMulCpuImplType()
  else:
    raise ValueError('Unsupported OP type: ' + op_type)
    return None

def scale_op_param(op_type, op_param, hw_scale, oc_scale):
  if op_type == OpType.Conv2D or op_type == OpType.Pooling:
    op_param['input_shape'][1] = int(op_param['input_shape'][1] * hw_scale)
    op_param['input_shape'][2] = int(op_param['input_shape'][2] * hw_scale)
    op_param['filter_shape'][0] = int(op_param['filter_shape'][0] * oc_scale)
  elif op_type == OpType.FullyConnected:
    op_param['input_shape'][1] = int(op_param['input_shape'][1] * hw_scale)
    op_param['input_shape'][2] = int(op_param['input_shape'][2] * hw_scale)
    op_param['weight_shape'][2] = op_param['input_shape'][1]
    op_param['weight_shape'][3] = op_param['input_shape'][2]
    op_param['weight_shape'][0] = int(op_param['weight_shape'][0] * oc_scale)
  elif op_type == OpType.Deconv2D:
    op_param['filter_shape'][0] = int(op_param['filter_shape'][0] * oc_scale)
  elif op_type == OpType.MatMul:
    return
  else:
    raise ValueError('Unsupported OP type: ' + op_type)

def build_devfreq(adb_device):
  '''
  target_soc = GetGlobalTargetSoc()
  if target_soc == TargetSoc.SDM855:
    return Sdm855DevfreqUtils(adb_device)
  elif target_soc == TargetSoc.SDM865:
    return Sdm865DevfreqUtils(adb_device)
  elif target_soc == TargetSoc.KIRIN960:
    return Kirin960DevfreqUtils(adb_device)
  else:
    raise ValueError('Unsupported target soc: ' + target_soc)
  '''

  soc_name = GetGlobalMobileSocName()
  if soc_name == 'sdm855':
    return Sdm855DevfreqUtils(adb_device)
  elif soc_name == 'sdm865':
    return Sdm865DevfreqUtils(adb_device)
  elif soc_name == 'sdm888':
    return Sdm888DevfreqUtils(adb_device)
  elif soc_name == 'kirin960':
    return Kirin960DevfreqUtils(adb_device)
  else:
    return None

def measure(op_type,
            ops_filename,
            output_dataset_type,
            output_dataset_path,
            dev_config,
            adb_device=None,
            cnn_model_names=None,
            impl_types=None,
            do_data_transform=False,
            do_write_data=True,
            start_op_idx=0,
            end_op_idx=sys.maxsize,
            sleep_time=0,
            do_pause=True):
  #PyCheck.check_not_none(cnn_model_names, 'Please identify at least one CNN model name')
  #PyCheck.check_not_none(impl_types, 'Please identify at least one implementation type')

  TerminalLogger.log(LogTag.INFO, 'Measure latency and extract model variables')

  # Setting list.
  op_type = StringToOpType(op_type)
  dataset_name = StringToDatasetName(output_dataset_type)
  is_measurement_enabled = True
  is_run_on_device_enabled = True
  do_compute = True
  gpu_mem_object = StringToMemoryObject('gpu_' + GetGlobalGpuMemoryType())
  cnn_model_names = convert_cnn_model_names(cnn_model_names)
  impl_types = StringToImplTypes(op_type, impl_types)
  is_wino_gemm = False
  if (impl_types is not None) and (Conv2dCpuImplType.WINOGRAD_GEMM in impl_types):
    is_wino_gemm = True
  scale_factors = {'HW': [1.0], 'OC': [1.0]}
  use_devfreq = False

  op_param_utils = get_op_param_utils(op_type)

  device_list = StringToDeviceList(dev_config)
  device_list, pdim_list, pratio_list, do_data_transform, do_compute = \
      get_dataset_configure(dataset_name, device_list, do_data_transform, do_compute)

  # Configures and confirm to start.
  TerminalLogger.log(LogTag.INFO, 'OP type: {}'.format(op_type.name))
  TerminalLogger.log(LogTag.INFO, 'OPs file name: {}'.format(ops_filename))
  TerminalLogger.log(LogTag.INFO, 'output dataset type: {}'.format(dataset_name))
  TerminalLogger.log(LogTag.INFO, 'start index: {}'.format(start_op_idx))
  TerminalLogger.log(LogTag.INFO, 'end index: {}'.format(end_op_idx))
  TerminalLogger.log(LogTag.INFO, 'sleep seconds: {}'.format(sleep_time))
  TerminalLogger.log(LogTag.INFO, 'is measurement enabled: {}'.format(is_measurement_enabled))
  TerminalLogger.log(LogTag.INFO, 'is run on device enabled: {}'.format(is_run_on_device_enabled))
  TerminalLogger.log(LogTag.INFO, 'is write data enabled: {}'.format(do_write_data))
  TerminalLogger.log(LogTag.INFO, 'do data transform: {}'.format(do_data_transform))
  TerminalLogger.log(LogTag.INFO, 'do compute: {}'.format(do_compute))
  TerminalLogger.log(LogTag.INFO, 'gpu memory object: {}'.format(gpu_mem_object))
  TerminalLogger.log(LogTag.INFO, 'cnn model name: {}'.format(cnn_model_names))
  TerminalLogger.log(LogTag.INFO, 'conv impl type: {}'.format(impl_types))
  TerminalLogger.log(LogTag.INFO, 'scale factors: {}'.format(scale_factors))
  TerminalLogger.log(LogTag.INFO, 'devices: {}'.format(device_list))
  TerminalLogger.log(LogTag.INFO, 'adb device: {}'.format(adb_device))
  TerminalLogger.log(LogTag.INFO, 'pdim: {}'.format(pdim_list))
  TerminalLogger.log(LogTag.INFO, 'pratio: {}'.format(pratio_list))
  TerminalLogger.log(LogTag.INFO, '')
  TerminalLogger.log(LogTag.INFO, 'Is smartphone configured (such as CPU/GPU frequency)?')
  if do_pause:
    TerminalUtils.pause('Press Enter to start')
  else:
    wait_sec = 3
    TerminalLogger.log(LogTag.INFO, 'We will start after {} seconds'.format(wait_sec))
    #time.sleep(wait_sec)
    TerminalUtils.count_down(wait_sec)
  
  op_data_utils = get_dataset_utils(dataset_name)

  devfreq = build_devfreq(adb_device) if use_devfreq else None

  if is_measurement_enabled:
    if do_write_data:
      # Create output dataset files.
      op_data_utils.create_all_files(output_dataset_path)
    #else:
    #  op_data_utils.print_title()

  # Load parameters from file.
  start_op_idx = 0
  op_count = 0
  done_op_param_list = []
  df = pd.read_csv(ops_filename)
  # Run.
  start_time = time.time()
  for hw_scale in scale_factors['HW']:
    for oc_scale in scale_factors['OC']:
      for index, row in df.iterrows():
        # Skip.
        if index < start_op_idx or index >= end_op_idx:
          continue
        # Load shape.
        op_param = op_param_utils.extract_from_row(row)
        # Scaling for HW and OC.
        scale_op_param(op_type, op_param, hw_scale, oc_scale)
        # Check if the shape has been done.
        if op_param_utils.is_in_list(op_param, done_op_param_list):
          TerminalLogger.log(LogTag.DEBUG, 'Skip an op parameter which has been done.')
          continue
        else:
          done_op_param_list.append(op_param)
        # Check if the CNN model should be measured.
        cnn_model_name = row['CNN']
        if (cnn_model_names is not None and cnn_model_name not in cnn_model_names) \
            or (cnn_model_name.startswith('NONE')):
          continue
        op_param['nn_name'] = cnn_model_name
        # Calculate output shape.
        op_param = op_param_utils.calc_output_shape(op_param)
        # Build ML (training/testing) data.
        for device in device_list:
          for pdim in pdim_list:
            for pratio in pratio_list:
              if pratio == 0.0 and device == Device.GPU or \
                 pratio == 1.0 and device == Device.CPU:
                continue
              target_op_param = op_param_utils.calc_partition_shape(
                  op_param, pdim, pratio, device)
              if target_op_param is None:
                continue
              #TerminalLogger.log(LogTag.INFO, 'target_op_param {}'.format(
              #      target_op_param))

              # Skip by implementation.
              impl_type = get_op_impl_type(op_type, target_op_param,
                                           is_wino_gemm, device)
              if (impl_types is not None) and (impl_type not in impl_types):
                continue

              #TerminalLogger.log(LogTag.INFO, 'conv shape: {}'.format(target_op_param))
              #TerminalLogger.log(LogTag.INFO, 'conv h {} oc {} flops {}'.format(
              #    target_op_param['output_shape'][1],
              #    target_op_param['output_shape'][3],
              #    op_param_utils.calc_flops(target_op_param)))

              if is_measurement_enabled:
                # Measture latency, and extract and write data to files.
                if is_run_on_device_enabled:
                  lat_list, sd_list = OpAdbUtils.run_op_on_device(
                      op_type,
                      target_op_param,
                      device,
                      op_data_utils.collect_granularity(),
                      adb_device=adb_device,
                      pdim=pdim,
                      do_data_transform=do_data_transform,
                      do_compute=do_compute,
                      gpu_mem_object=gpu_mem_object)
                else:
                  lat_list, sd_list = None, None
                #TerminalLogger.log(LogTag.INFO, 'lat list: {}'.format(lat_list))
                #TerminalUtils.pause('Press Enter to continue')
                
                target_op_param['is_wino_gemm'] = is_wino_gemm
                data_dict = op_data_utils.extract(target_op_param,
                                                  device,
                                                  do_data_transform,
                                                  lat_list)
                #TerminalLogger.log(LogTag.INFO, 'data list: {}'.format(data_dict))
                #TerminalUtils.pause('Press Enter to continue')

                data_dict['sd_vars'] = [sd_list]
                data_dict['partition_vars'] = [[pdim.value, pratio]]

                if do_write_data:
                  op_data_utils.write_data(output_dataset_path, data_dict)
                else:
                  op_data_utils.print_data(data_dict, cnn_model_name)

                if impl_type == Conv2dCpuImplType.WINOGRAD and \
                   dataset_name == DatasetName.MLConv2d:
                  target_op_param['is_wino_gemm'] = True

                  data_dict = op_data_utils.extract(target_op_param,
                                                    device,
                                                    do_data_transform,
                                                    lat_list)

                  data_dict['sd_vars'] = [sd_list]
                  data_dict['partition_vars'] = [[pdim.value, pratio]]

                  if do_write_data:
                    op_data_utils.write_data(output_dataset_path, data_dict)
                  else:
                    op_data_utils.print_data(data_dict, cnn_model_name)

                op_count = op_count + 1

              if do_write_data:
                time_now = round(time.time() - start_time)
                TerminalLogger.log(LogTag.INFO, 'time: {}'.format(
                    TimeUtils.sec_to_format_text(time_now)))
                TerminalLogger.log(LogTag.INFO, 'time per op: {:.3f}'.format(
                    float(time_now) / op_count))
                TerminalLogger.log(LogTag.INFO, 'df index: {}/{}'.format(index, len(df) - 1))
                TerminalLogger.log(LogTag.INFO, 'op count: {}'.format(op_count))
                TerminalLogger.log(LogTag.INFO, 'dataset: {}'.format(dataset_name))
                TerminalLogger.log(LogTag.INFO, 'cnn: {}'.format(cnn_model_name))
                TerminalLogger.log(LogTag.INFO, 'scale: HW {}'.format(hw_scale))
                TerminalLogger.log(LogTag.INFO, 'param: dev {}, pdim {}, pratio {}'.format(
                    device.name, PartitionDimensionValueToText(pdim.value), pratio))
                if devfreq is not None:
                  TerminalLogger.log(LogTag.INFO, 'freq: cpu {}, gpu {}'.format(
                      devfreq.get_cur_cpu_freq(), devfreq.get_cur_gpu_freq()))
                TerminalLogger.log(LogTag.INFO, '')

                if sleep_time > 0:
                  time.sleep(sleep_time)

  if is_measurement_enabled and do_write_data:
    op_data_utils.remove_unused_files(output_dataset_path)

  TerminalLogger.log(LogTag.INFO, '')
  TerminalLogger.log(LogTag.INFO, 'Total op count: ' + str(op_count))
  TerminalLogger.log(LogTag.INFO, '')

def multi_measure(op_type, ops_filename, start_idx=0, end_idx=-1):
  measure_params = []
  measure_params.append({'output_dataset_type': 'MLConv2d',
                         'output_dataset_path': 'dataset',
                         'device': 'CPU+GPU', 'data_transform': True})
  measure_params.append({'output_dataset_type': 'MLConv2d',
                         'output_dataset_path': 'dataset',
                         'device': 'CPU+GPU', 'data_transform': False})
  measure_params.append({'output_dataset_type': 'MLConv2dOp',
                         'output_dataset_path': 'dataset',
                         'device': 'CPU+GPU', 'data_transform': False})
  measure_params.append({'output_dataset_type': 'MuLayerLRConv2d',
                         'output_dataset_path': 'dataset',
                         'device': 'CPU+GPU', 'data_transform': False})
  
  end_idx = len(measure_params) if end_idx == -1 else end_idx
  for i in range(start_idx, end_idx):
    measure(op_type,
            ops_filename,
            measure_params[i]['output_dataset_type'],
            measure_params[i]['output_dataset_path'],
            measure_params[i]['device'],
            measure_params[i]['data_transform'],
            do_pause=False)

SUPPRTED_OUTPUT_DATASET_TYPE = ['FLOPsLRConv2d',
                                'MuLayerLRConv2d',
                                'MuLayerLRPooling',
                                'MuLayerLRFullyConnected',
                                'MLConv2d', 'MLConv2dOp', 'MLConv2dOpImplType',
                                'MLPooling',
                                'MLFullyConnected', 'MLFullyConnectedOp',
                                'MLDeconv2d', 'MLMatMul']

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--op_type', type=str, required=True, help='OP type')
  parser.add_argument('--input_dataset', type=str, required=True, help='Input dataset')
  parser.add_argument('--output_dataset_type', type=str, required=True,
                      choices=SUPPRTED_OUTPUT_DATASET_TYPE, help='Output dataset type')
  parser.add_argument('--output_dataset_path', type=str, required=False,
                      default='dataset', help='Output dataset path')
  parser.add_argument('--adb_device', type=str, required=False, default=None,
                      help='ADB device')
  parser.add_argument('--soc', type=str, required=True,
                      choices=['sdm855', 'sdm865', 'sdm888', 'kirin960', 'kirin990'],
                      default='sdm855', help='SoC')
  parser.add_argument('--device', type=str, required=True,
                      choices=['CPU', 'GPU', 'CPU+GPU'], default='CPU+GPU', help='Devices')
  parser.add_argument('--num_threads', type=int, required=False, default=1, help='Number of threads')
  parser.add_argument('--gpu_mem_type', type=str, required=False,
                      choices=['buffer', 'image'], default='image',
                      help='GPU memory type')
  parser.add_argument('--cnn_models', nargs='+', default=None, help='CNN models')
  parser.add_argument('--impl_types', nargs='+', default=None, help='Implement types')
  parser.add_argument('--multi_measure', action='store_true', help='Multiple measurement')
  parser.add_argument('--data_transform', action='store_true', help='Data transform')
  parser.add_argument('--write_data', action='store_true', help='Write data to file')
  parser.add_argument('--pause', action='store_true', help='Pause')
  args = parser.parse_args()
  return args

def main():
  RegistrySoc()

  args = parse_args()

  op_type = args.op_type
  input_dataset = args.input_dataset
  multi_measure = args.multi_measure
  soc_name = args.soc
  num_threads = args.num_threads
  gpu_mem_type = args.gpu_mem_type

  #SetGlobalTargetSoc(soc_name)
  SetGlobalMobileSocName(soc_name)
  SetGlobalThreadCount(num_threads)
  SetGlobalGpuMemoryType(gpu_mem_type)

  if not multi_measure:
    output_dataset_type = args.output_dataset_type
    output_dataset_path = args.output_dataset_path
    device = args.device
    adb_device = args.adb_device
    cnn_models = args.cnn_models
    impl_types = args.impl_types
    do_data_transform = args.data_transform
    do_write_data = args.write_data
    do_pause = args.pause
    measure(op_type, input_dataset, output_dataset_type, output_dataset_path,
            device, adb_device, cnn_models, impl_types,
            do_data_transform, do_write_data,
            do_pause=do_pause)
  else:
    multi_measure(op_type, input_dataset)

if __name__ == '__main__':
  main()
