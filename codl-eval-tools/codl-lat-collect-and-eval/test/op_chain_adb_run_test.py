
import os
import sys

def import_parent_path():
  sys.path.append('.')
  path_list = []
  for root, dirs, _ in os.walk('.'):
    for name in dirs:
      path_list.append(os.path.join(root, name))
  sys.path.extend(path_list)

import_parent_path()

import time
import argparse
from utils.common.log import LogTag, TerminalLogger
from utils.dl_model.dl_model import *
from utils.op.op_chain_adb_utils import *
from utils.power.power_utils import *

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dl_model', type=str, required=True, help='DL model')
  parser.add_argument('--exec_type', type=str, required=True,
                      help='Execution type')
  parser.add_argument('--rounds', type=int, required=False,
                      default=10, help='Rounds')
  parser.add_argument('--adb_device', type=str, required=False,
                      default=None, help='ADB device')
  parser.add_argument('--enable_power_and_energy', action='store_true')
  parser.add_argument('--enable_log', action='store_true')
  args = parser.parse_args()
  return args

def StringToDLModel(model):
  if model == 'yolo_v2': return DLModel.YOLO_V2
  elif model == 'vgg16': return DLModel.VGG16
  elif model == 'posenet': return DLModel.POSENET
  elif model == 'fast_style_transfer': return DLModel.FAST_STYLE_TRANSFER
  elif model == 'retinaface': return DLModel.RETINAFACE
  else:
    raise ValueError('Unsupported model name: ' + model)

def StringToExecutionType(exec_type):
  if exec_type == 'CPU': return ExecutionType.CPU
  elif exec_type == 'GPU_IMAGE': return ExecutionType.GPU_IMAGE
  elif exec_type == 'GPU_BUFFER': return ExecutionType.GPU_BUFFER
  elif exec_type == 'MULAYER_BUFFER_BASED': return ExecutionType.MULAYER_BUFFER_BASED
  elif exec_type == 'CODL_BUFFER_BASED': return ExecutionType.CODL_BUFFER_BASED
  elif exec_type == 'CODL': return ExecutionType.CODL
  else:
    return ValueError('Unsupported execution type: ' + exec_type)

def BuildExecutionParam(exec_type):
  params = {}
  params['op_count'] = -1
  params['exec_type'] = exec_type
  params['chain_param_hint'] = 1
  params['gpu_mtype'] = 2
  params['data_transform'] = 1
  params['make_partition_plan'] = 0
  params['latency_acq'] = 1
  params['lp_backend'] = 1
  params['search_method'] = 'serial'
  params['search_baseline'] = 0
  params['pratio_hint'] = 0
  params['config_file'] = '/data/local/tmp/codl/configs/config_codl.json'
  params['debug_level'] = 0
  params['min_power_limit'] = 0.0
  if exec_type == ExecutionType.CPU:
    params['chain_param_hint'] = 0
    params['data_transform'] = 0
    params['min_power_limit'] = 6.0
  elif exec_type == ExecutionType.GPU_IMAGE:
    params['chain_param_hint'] = 1
    params['min_power_limit'] = 3.0
  elif exec_type == ExecutionType.GPU_BUFFER:
    params['gpu_mtype'] = 1
    params['chain_param_hint'] = 1
    params['min_power_limit'] = 2.5
  elif exec_type == ExecutionType.MULAYER_BUFFER_BASED:
    params['gpu_mtype'] = 1
    params['make_partition_plan'] = 1
    params['lp_backend'] = 1
    params['config_file'] = '/data/local/tmp/codl/configs/config_mulayer.json'
    params['min_power_limit'] = 7.0
  elif exec_type == ExecutionType.CODL_BUFFER_BASED or \
       exec_type == ExecutionType.CODL:
    if exec_type == ExecutionType.CODL_BUFFER_BASED:
      params['gpu_mtype'] = 1
    params['lp_backend'] = 0
    params['chain_param_hint'] = 11
    params['search_method'] = 'heuristic'
    params['min_power_limit'] = 7.75
  else:
    raise ValueError('Unsupported execution type')

  return params

def SetExecutionParamForDLModel(model, exec_params):
  exec_params['internal_rounds'] = 50
  if model == DLModel.FAST_STYLE_TRANSFER:
    exec_params['internal_rounds'] = 30
  elif model == DLModel.RETINAFACE:
    exec_params['internal_rounds'] = 20
  return exec_params

def SetParamForAdbDevice(adb_devname, exec_params):
  if adb_devname == 'PKT0220320002864':
    exec_params['chain_param_hint'] = 10
  return exec_params

def SummarizeListValues(val_list):
  if len(val_list) == 0:
    return 0, 0, 0
  val_arr = np.array(val_list)
  avg = np.mean(val_arr)
  perr = np.max(val_arr) - np.mean(val_arr)
  merr = np.mean(val_arr) - np.min(val_arr)
  return avg, perr, merr

def run_dl_model(model, exec_type, rounds,
                 adb_devname=None,
                 enable_power_and_energy=False,
                 enable_log=False):
  exec_params = BuildExecutionParam(exec_type)
  exec_params = SetExecutionParamForDLModel(model, exec_params)
  #exec_params = SetParamForAdbDevice(adb_devname, exec_params)
  bp_sampler = None
  if enable_power_and_energy:
    bp_sampler = BatteryPowerSampler(0.1, adb_devname, enable_debug=False)

  lat_list = []
  power_list = []
  energy_list = []
  kEnergyRounds = 1
  for i in range(rounds):
    start_time = time.time()
    if bp_sampler is not None:
      bp_sampler.start()
    lat = OpChainAdbUtils.run_on_device(model, exec_params, True, adb_devname)
    if bp_sampler is not None:
      bp_sampler.stop()
    duration = time.time() - start_time
    if bp_sampler is not None:
      power = bp_sampler.get_avg_power(exec_params['min_power_limit'])
    else:
      power = 0
    energy = (lat  / 1000.0) * power * kEnergyRounds
    if bp_sampler is not None:
      bat_cap = BatteyAdbUtils.read_capacity(adb_devname)
    else:
      bat_cap = 100
    result_text = 'model %s, exec_type %s, round %d, lat %.2f, power %.2f, energy %.2f, bat_cap %d, duration %.2f' % (
        ModelToModelName(model), ExecutionTypeToName(exec_type), i,
        lat, power, energy, bat_cap, duration)
    TerminalLogger.log(LogTag.VERBOSE, result_text)
    if lat > 0:
      lat_list.append(lat)
    if power > 0:
      power_list.append(power)
    if energy > 0:
      energy_list.append(energy)
    if bat_cap < 20:
      raise ValueError('Battery capacity is lower than 20%, please charge')
    time.sleep(5)

    if enable_log and (i > 0) and ((i + 1) % 5) == 0:
      avg_lat, lat_perr, lat_merr = SummarizeListValues(lat_list)
      avg_power, power_perr, power_merr = SummarizeListValues(power_list)
      avg_energy, energy_perr, energy_merr = SummarizeListValues(energy_list)
      output_text = ('model %s, exec_type %s,' + \
                     ' avg_lat %.2f, lat_perr %.2f, lat_merr %.2f,' + \
                     ' avg_power %.2f, power_perr %.2f, power_merr %.2f,' + \
                     ' avg_energy %.2f, energy_perr %.2f, energy_merr %.2f\n') % (
                          model.name, exec_type.name,
                          avg_lat, lat_perr, lat_merr,
                          avg_power, power_perr, power_merr,
                          avg_energy, energy_perr, energy_merr)
      TerminalLogger.log(LogTag.INFO, output_text)

  if enable_log:
    if adb_devname is not None:
      output_log_file = 'dl_model_e2e_eval_%s.log' % adb_devname
    else:
      output_log_file = 'dl_model_e2e_eval.log'
    output_log_filepath = os.path.join('logs', output_log_file)

    with open(output_log_filepath, 'a') as f:
      f.write(output_text)

def run_dl_models_and_exec_types(dl_models, exec_types, rounds, adb_devname=None,
                                 enable_power_and_energy=False, enable_log=False):
  for dl_model in dl_models:
    for et in exec_types:
      run_dl_model(dl_model, et, rounds, adb_devname,
                   enable_power_and_energy, enable_log)

ALL_DL_MODELS = [DLModel.YOLO_V2,
                 DLModel.VGG16,
                 DLModel.POSENET,
                 DLModel.FAST_STYLE_TRANSFER,
                 DLModel.RETINAFACE]

ALL_EXEC_TYPES = [ExecutionType.CPU,
                  ExecutionType.GPU_IMAGE,
                  #ExecutionType.GPU_BUFFER,
                  ExecutionType.MULAYER_BUFFER_BASED,
                  ExecutionType.CODL_BUFFER_BASED,
                  ExecutionType.CODL]

def run_all_exec_types(dl_model, rounds, adb_devname=None):
  exec_types = ALL_EXEC_TYPES
  for et in exec_types:
    run_dl_model(dl_model, et, rounds, adb_devname)

def run_all_dl_models(rounds, adb_devname=None):
  dl_models = ALL_DL_MODELS
  for dl_model in dl_models:
    run_all_exec_types(dl_model, rounds, adb_devname)

if __name__ == '__main__':
  args = parse_args()
  dl_model = args.dl_model
  exec_type = args.exec_type
  rounds = args.rounds
  adb_devname = args.adb_device
  enable_log = args.enable_log
  enable_power_and_energy = args.enable_power_and_energy

  if dl_model == 'all':
    dl_models = ALL_DL_MODELS
  else:
    dl_models = [StringToDLModel(dl_model)]
  
  if exec_type == 'all':
    exec_types = ALL_EXEC_TYPES
  else:
    exec_types = [StringToExecutionType(args.exec_type)]
  
  run_dl_models_and_exec_types(dl_models, exec_types,
                               rounds, adb_devname, enable_power_and_energy,
                               enable_log)
