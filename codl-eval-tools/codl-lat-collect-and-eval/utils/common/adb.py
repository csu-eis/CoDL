
import os
import sys
from enum import Enum

kDebug = False
kUseRoot = False

class ShellUtils:
  @staticmethod
  def run_cmd(cmd, enable_debug=False):
    with os.popen(cmd) as pipe:
      #return pipe.read()
      text = ''
      for line in pipe:
        text = text + line
        if enable_debug:
          print(line, end='')
      return text

class AdbUtils:
  @staticmethod
  def cat(filename, adb_devname=None):
    cmd = 'adb'
    if adb_devname is not None:
      cmd = cmd + ' -s %s' % adb_devname
    if kUseRoot:
      cmd = cmd + ' shell \"su -c \\\"cat {}\\\"\"'.format(filename)
    else:
      cmd = cmd + ' shell \"cat {}\"'.format(filename)
    if kDebug:
      print('cmd: ' + cmd)
    ret_text = ShellUtils.run_cmd(cmd)
    if ret_text[-1] == '\n':
      ret_text = ret_text[:-1]
    if kDebug:
      print('ret: ' + ret_text)
    return ret_text

  @staticmethod
  def echo(filename, value, adb_devname=None):
    cmd = 'adb'
    if adb_devname is not None:
      cmd = cmd + ' -s %s' % adb_devname
    if kUseRoot:
      cmd = cmd + ' shell \"su -c \\\"echo {} > {}\\\"\"'.format(value, filename)
    else:
      cmd = cmd + ' shell \"echo {} > {}\"'.format(value, filename)
    if kDebug:
      print('cmd: ' + cmd)
    ret_text = ShellUtils.run_cmd(cmd)
    if kDebug:
      print('ret: ' + ret_text)

class MaceEnv(Enum):
  MACE_CPP_MIN_VLOG_LEVEL = 'MACE_CPP_MIN_VLOG_LEVEL'
  MACE_OPENCL_PROFILING = 'MACE_OPENCL_PROFILING'
  MACE_OPENCL_WORKGROUP_SIZE = 'MACE_OPENCL_WORKGROUP_SIZE'
  MACE_DURATION_COLLECT_GRANULARITY = 'MACE_DURATION_COLLECT_GRANULARITY'
  MACE_DO_DATA_TRANSFORM = 'MACE_DO_DATA_TRANSFORM'
  MACE_DO_COMPUTE = 'MACE_DO_COMPUTE'
  CODL_CONFIG_PATH = 'CODL_CONFIG_PATH'

def add_env_var(env_var, var):
  return env_var + ' ' + var

def add_env_var_if_true(env_var, var, cond):
  if cond:
    return env_var + ' ' + var
  else:
    return env_var

def add_env_var_if_equal(env_var, var, cond, tar_cond):
  if cond == tar_cond:
    return env_var + ' ' + var
  else:
    return env_var

def add_env_var_if_not_none(env_var, var, cond):
  if cond is not None:
    return env_var + ' ' + var
  else:
    return env_var

def build_env(name, value):
  return '{}={}'.format(name, value)
