
import numpy as np
from enum import Enum
from utils.dl_model.dl_model import *
from utils.common.adb import *

kDebug = False

class ExecutionType(Enum):
  NONE = 0
  CPU = 1
  GPU_IMAGE = 2
  GPU_BUFFER = 3
  MULAYER_BUFFER_BASED = 4
  CODL_BUFFER_BASED = 5
  CODL = 6
  IDEAL = 7

def ExecutionTypeToName(exec_type):
  if exec_type == ExecutionType.NONE: return 'none'
  elif exec_type == ExecutionType.CPU: return 'cpu'
  elif exec_type == ExecutionType.GPU_IMAGE: return 'gpu'
  elif exec_type == ExecutionType.MULAYER_BUFFER_BASED: return 'mulayer_buffer_based'
  elif exec_type == ExecutionType.CODL_BUFFER_BASED: return 'codl_buffer_based'
  elif exec_type == ExecutionType.CODL: return 'codl'
  else: raise ValueError('Unsupported execution type')

def BuildTestName(model, exec_params):
  model_name = ModelToModelName(model)
  if exec_params['exec_type'] == ExecutionType.CODL_BUFFER_BASED or \
     exec_params['exec_type'] == ExecutionType.CODL:
    return model_name + '_real_chain_search'
  else:
    return model_name + '_chain_search'

def ExtractLatencyValue(text):
  total = 0
  text_items = text.split(' ')
  for i in range(len(text_items)):
    if text_items[i] == 'total':
      total = float(text_items[i + 1])
  return total

def ExtractLatencyValues(text):
  avg, perr, merr = 0, 0, 0
  text_items = text.split(' ')
  for i in range(len(text_items)):
    if text_items[i] == 'avg_lat':
      avg = float(text_items[i + 1])
    elif text_items[i] == 'lat_perr':
      perr = float(text_items[i + 1])
    elif text_items[i] == 'lat_merr':
      merr = float(text_items[i + 1])
  return avg, perr, merr

class OpChainAdbUtils(object):
  @staticmethod
  def run_on_device(model, exec_params, enable_grep=True, adb_device=None):
    if kDebug:
      print(exec_params)

    workspace_path = '/data/local/tmp/codl'
    execute_name = 'codl_run'

    env_var = ''
    env_var = add_env_var_if_true(
        env_var,
        build_env(MaceEnv.MACE_OPENCL_PROFILING.name, 1),
        True)
    env_var = add_env_var_if_not_none(
        env_var,
        build_env(MaceEnv.CODL_CONFIG_PATH.name, exec_params['config_file']),
        exec_params['config_file'])

    cmd = 'adb'
    if adb_device is not None:
      cmd = cmd + ' -s ' + adb_device
    cmd = cmd + ' shell \"' + env_var + ' ' \
        + "/".join((workspace_path, execute_name))

    cmd = cmd + ' --test=%s' % BuildTestName(model, exec_params)
    cmd = cmd + ' --op_idx=0'
    cmd = cmd + ' --op_count=%d' % exec_params['op_count']
    cmd = cmd + ' --chain_idx=-1'
    cmd = cmd + ' --chain_count=-1'
    cmd = cmd + ' --num_threads=4'
    cmd = cmd + ' --chain_param_hint=%d' % exec_params['chain_param_hint']
    cmd = cmd + ' --gpu_mtype=%d' % exec_params['gpu_mtype']
    if exec_params['make_partition_plan'] == 1:
      cmd = cmd + ' --make_partition_plan'
      cmd = cmd + ' --profile_data_transform'
      cmd = cmd + ' --profile_compute'
    if exec_params['data_transform'] == 1:
      cmd = cmd + ' --data_transform'
    cmd = cmd + ' --compute'
    cmd = cmd + ' --latency_acq=%d' % exec_params['latency_acq']
    cmd = cmd + ' --lp_backend=%d' % exec_params['lp_backend']
    cmd = cmd + ' --search_method=%s' % exec_params['search_method']
    cmd = cmd + ' --search_baseline=%d' % exec_params['search_baseline']
    cmd = cmd + ' --pratio_hint=%d' % exec_params['pratio_hint']
    cmd = cmd + ' --rounds=%d' % exec_params['internal_rounds']
    cmd = cmd + ' --debug_level=%d' % exec_params['debug_level']
    cmd = cmd + '\"'

    if enable_grep:
      enable_debug = False
      grep_keyword = 'total'
      cmd = cmd + ' | grep \"{:s}\"'.format(grep_keyword)
    else:
      enable_debug = True

    if kDebug:
      print(cmd)

    ret_text = ShellUtils.run_cmd(cmd, enable_debug)

    if kDebug:
      print(ret_text)

    if enable_grep:
      lat = ExtractLatencyValue(ret_text)
      return lat
    else:
      return -1
