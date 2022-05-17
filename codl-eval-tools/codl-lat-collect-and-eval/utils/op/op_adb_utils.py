
from enum import Enum
from utils.common.adb import *
from utils.common.log import *
from utils.common.string import *
from utils.common.basic_type import *
from utils.op.op_type import OpType
from utils.op.op_common_calc import *
from utils.op.conv2d_param_utils import Conv2dParamUtils
from utils.dataset.ml_dataset_utils import GetGlobalThreadCount

kEnableDebug = False
kEnableGrep = True

class AdbOutputDataType(Enum):
  NONE = 0
  OP_LATENCY = 1
  GPU_INFO_BASE = 2
  GPU_INFO_KERNEL = 3

def extract_data(text, data_type):
  if text[-1] == ',':
    text = text[:-1]
  
  if data_type == 'int':
    return int(text)
  elif data_type == 'float':
    return float(text)
  else:
    raise ValueError('Unsupported data type: ' + data_type)
    return 0

class OpAdbUtils(object):
  @staticmethod
  def run_op_on_device(op_type,
                       op_param,
                       dev_name,
                       collect_granularity,
                       adb_device=None,
                       pdim=PartitionDimension.HEIGHT,
                       pratio=1.0,
                       do_data_transform=False,
                       do_compute=True,
                       gpu_mem_object=MemoryObject.GPU_IMAGE,
                       gpu_lws=None,
                       output_data_type=AdbOutputDataType.OP_LATENCY,
                       vlog=0,
                       debug=False):
    def build_gpu_lws(lws):
      if lws is None:
        return
      
      text = '['
      for v in lws:
        text = text + ('%d,' % v)
      text = text + '0]'
      return text

    env_var = ''

    env_var = add_env_var(env_var,
                          build_env(MaceEnv.MACE_CPP_MIN_VLOG_LEVEL.name, vlog))
    env_var = add_env_var_if_equal(
        env_var,
        build_env(MaceEnv.MACE_DURATION_COLLECT_GRANULARITY.name, 'Fine'),
        collect_granularity,
        'fine')
    env_var = add_env_var_if_equal(
        env_var,
        build_env(MaceEnv.MACE_DURATION_COLLECT_GRANULARITY.name, 'Coarse'),
        collect_granularity,
        'coarse')
    env_var = add_env_var_if_equal(
        env_var,
        build_env(MaceEnv.MACE_DURATION_COLLECT_GRANULARITY.name, 'None'),
        collect_granularity,
        'none')
    env_var = add_env_var_if_not_none(
        env_var,
        build_env(MaceEnv.MACE_OPENCL_WORKGROUP_SIZE.name, build_gpu_lws(gpu_lws)),
        gpu_lws)

    if dev_name == Device.CPU:
      num_threads = GetGlobalThreadCount()
      #pdim = PartitionDimension.HEIGHT
      pratio = 0.0
      cu_hint = 1
    elif dev_name == Device.GPU:
      num_threads = 1
      #pdim = PartitionDimension.HEIGHT
      pratio = 1.0
      cu_hint = 2
      env_var = add_env_var_if_true(
          env_var,
          build_env(MaceEnv.MACE_OPENCL_PROFILING.name, 1),
          True)
    elif dev_name == Device.CPU_GPU:
      num_threads = GetGlobalThreadCount()
      cu_hint = 0

    if pdim == PartitionDimension.WIDTH or \
       pdim == PartitionDimension.IN_CHANNEL:
      pdim = PartitionDimension.HEIGHT

    env_var = add_env_var_if_true(
        env_var,
        build_env(MaceEnv.MACE_DO_DATA_TRANSFORM.name, 1),
        do_data_transform)
    env_var = add_env_var_if_true(
        env_var,
        build_env(MaceEnv.MACE_OPENCL_PROFILING.name, 1),
        do_data_transform)
    env_var = add_env_var_if_true(
        env_var,
        build_env(MaceEnv.MACE_DO_COMPUTE.name, 1),
        do_compute)

    codl_mobile_path = '/data/local/tmp/codl'
    codl_op_run_name = 'codl_op_run'

    warmup_rounds = 1 + 10
    rounds = 10

    cmd = 'adb'
    if adb_device is not None:
      cmd = cmd + ' -s ' + adb_device
    cmd = cmd + ' shell \"' + env_var + ' ' \
        + "/".join((codl_mobile_path, codl_op_run_name))
    '''
    cmd = cmd + ' -p 1 -c 1 -w 0'
    cmd = cmd + ' -o ' + str(warmup_rounds + rounds)
    cmd = cmd + ' -m ' + str(gpu_mem_object.value)
    cmd = cmd + ' -t ' + str(num_threads) + ' -d ' + str(pdim.value) \
              + ' -r ' + str(pratio) + ' -u ' + str(cu_hint)
    
    conv2d_shape_text = "{},{},{},{};{},{},{},{};{},{}".format(
        conv2d_param['input_shape'][0],
        conv2d_param['input_shape'][1],
        conv2d_param['input_shape'][2],
        conv2d_param['input_shape'][3],
        conv2d_param['filter_shape'][0],
        conv2d_param['filter_shape'][1],
        conv2d_param['filter_shape'][2],
        conv2d_param['filter_shape'][3],
        conv2d_param['strides'][0],
        conv2d_param['strides'][1])
    cmd = cmd + ' -s \\\"' + conv2d_shape_text + '\\\"\"'
    '''

    if do_data_transform:
      cmd = cmd + ' --data_transform'
      cmd = cmd + ' --debug'
    if do_compute:
      cmd = cmd + ' --compute'
    cmd = cmd + ' --cpu_affinity_policy=1'
    cmd = cmd + ' --rounds=%d' % (warmup_rounds + rounds)
    cmd = cmd + ' --gpu_memory_type=%d' % gpu_mem_object.value
    cmd = cmd + ' --num_threads=%d' % num_threads
    cmd = cmd + ' --part_dim=%d' % pdim.value
    cmd = cmd + ' --part_ratio=%f' % pratio
    cmd = cmd + ' --cu_hint=%d' % cu_hint

    def build_op_args(cmd, op_type, op_param):
      cmd = cmd + ' --op_type=%s' % op_type.name
      if len(op_param['input_shape']) == 4:
        cmd = cmd + ' --input_shape=\\\"%d,%d,%d,%d\\\"' % (
            op_param['input_shape'][0],
            op_param['input_shape'][1],
            op_param['input_shape'][2],
            op_param['input_shape'][3])
      elif len(op_param['input_shape']) == 2:
        cmd = cmd + ' --input_shape=\\\"%d,%d\\\"' % (
            op_param['input_shape'][0],
            op_param['input_shape'][1])
      
      if op_type == OpType.Conv2D or \
         op_type == OpType.Pooling or \
         op_type == OpType.Deconv2D:
        cmd = cmd + ' --weight_shape=\\\"%d,%d,%d,%d\\\"' % (
            op_param['filter_shape'][0],
            op_param['filter_shape'][1],
            op_param['filter_shape'][2],
            op_param['filter_shape'][3])
        cmd = cmd + ' --strides=\\\"%d,%d\\\"' % (
            op_param['strides'][0],
            op_param['strides'][1])
        if op_type == OpType.Pooling:
          cmd = cmd + ' --pooling_type=%d' % op_param['pooling_type']
      elif op_type == OpType.FullyConnected:
        cmd = cmd + ' --weight_shape=\\\"%d,%d,%d,%d\\\"' % (
            op_param['weight_shape'][0],
            op_param['weight_shape'][1],
            op_param['weight_shape'][2],
            op_param['weight_shape'][3])
      elif op_type == OpType.MatMul:
        rank = len(op_param['input_shape'])
        if rank == 2:
          cmd = cmd + ' --weight_shape=\\\"%d,%d\\\"' % (
              op_param['rhs_shape'][0],
              op_param['rhs_shape'][1])
        elif rank == 4:
          cmd = cmd + ' --weight_shape=\\\"%d,%d,%d,%d\\\"' % (
              op_param['rhs_shape'][0],
              op_param['rhs_shape'][1],
              op_param['rhs_shape'][2],
              op_param['rhs_shape'][3])

        if op_param['transpose_a']:
          cmd = cmd + ' --transpose_a'
        if op_param['transpose_b']:
          cmd = cmd + ' --transpose_b'
      else:
        raise ValueError('Unsupported op type: ' + op_type.name)

      return cmd

    cmd = build_op_args(cmd, op_type, op_param)
    
    if output_data_type == AdbOutputDataType.OP_LATENCY:
      grep_keyword = 'Stat:'
    elif output_data_type == AdbOutputDataType.GPU_INFO_BASE:
      grep_keyword = 'global_mem_cacheline_size'
    elif output_data_type == AdbOutputDataType.GPU_INFO_KERNEL:
      grep_keyword = '] kwg_size'
    else:
      raise ValueError('Unsupported ADB output data type: ' + output_data_type)

    if kEnableGrep:
      cmd = cmd + '\" | grep \"{:s}\"'.format(grep_keyword)

    if kEnableDebug:
      print('cmd: ' + cmd)
      TerminalUtils.pause('Press ENTER to run cmd')
    
    ret_text = ShellUtils.run_cmd(cmd)
    
    if kEnableDebug:
      print('ret: ' + ret_text)

    if output_data_type == AdbOutputDataType.OP_LATENCY:
      if not kEnableGrep:
        return [-1, -1, -1]

      def extract_lat_list(text):
        text = text.split(', ')[-2]
        text = text.split(' ')[-1]
        return StringUtils.str_to_float_list(text)

      def extract_sd_list(text):
        text = text.split(', ')[-1]
        text = text.split(' ')[-1]
        return StringUtils.str_to_float_list(text)

      if ret_text != None and len(ret_text) > 0:
        # Extract latency text.
        #print(ret_text)
        lat_list = extract_lat_list(ret_text)
        #print(lat_list)
        sd_list = extract_sd_list(ret_text)
        #print(sd_list)
      else:
        lat_list = [-1]
        sd_list = [0]

      return lat_list, sd_list
    elif output_data_type == AdbOutputDataType.GPU_INFO_BASE:
      global_mem_cache_size = 0
      compute_units = 0

      text_items = ret_text.split(' ')
      for i in range(len(text_items)):
        if text_items[i] == 'global_mem_cache_size':
          global_mem_cache_size = extract_data(text_items[i + 1], 'int')
        elif text_items[i] == 'compute_units':
          compute_units = extract_data(text_items[i + 1], 'int')

      return global_mem_cache_size, compute_units
    elif output_data_type == AdbOutputDataType.GPU_INFO_KERNEL:
      kwg_size = 0

      text_items = ret_text.split(' ')
      for i in range(len(text_items)):
        if text_items[i] == 'kwg_size':
          kwg_size = extract_data(text_items[i + 1], 'int')

      return kwg_size

  @staticmethod
  def get_gpu_info(adb_device=None):
    conv2d_param_utils = Conv2dParamUtils()
    conv2d_param = conv2d_param_utils.example()
    global_mem_cache_size, compute_units = OpAdbUtils.run_op_on_device(
        OpType.Conv2D, conv2d_param, Device.GPU, 'coarse', adb_device,
        PartitionDimension.HEIGHT, 1.0,
        False, False, MemoryObject.GPU_IMAGE, None,
        AdbOutputDataType.GPU_INFO_BASE)
    kwg_size = OpAdbUtils.run_op_on_device(
        OpType.Conv2D, conv2d_param, Device.GPU, 'coarse', adb_device,
        PartitionDimension.HEIGHT, 1.0,
        False, True, MemoryObject.GPU_IMAGE, None,
        AdbOutputDataType.GPU_INFO_KERNEL, 1)
    if kEnableDebug:
      print('Global memory cache size: %d' % global_mem_cache_size)
      print('Compute units: %d' % compute_units)
      print('Max work group size: %d' % kwg_size)
    return global_mem_cache_size, compute_units, kwg_size
