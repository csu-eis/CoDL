
import os
import numpy as np
from utils.common.log import *
from utils.common.file import *
from utils.common.math import *
from utils.common.cpu_threadpool_calc import *
from utils.common.basic_type import Device
from utils.soc.soc import *
from utils.op.conv2d_common_calc import *
from utils.op.conv2d_gpu_calc import *
from utils.op.fully_connected_common_calc import *
from utils.op.fully_connected_gpu_calc import *
from utils.op.deconv2d_common_calc import *
from utils.op.deconv2d_gpu_calc import *
from utils.op.matmul_common_calc import *
from utils.op.matmul_param_utils import *
from utils.op.matmul_gpu_calc import *
from utils.dataset.dataset_info import *
from config import *

'''
def GetMobileSocName():
  target_soc = GetGlobalTargetSoc()
  if target_soc == TargetSoc.SDM855:
    return SocName.Snapdragon855.name
  elif target_soc == TargetSoc.SDM865:
    return SocName.Snapdragon865.name
  elif target_soc == TargetSoc.KIRIN960:
    return SocName.Kirin960.name
  else:
    raise ValueError('Unsupported target soc: ' + target_soc)
'''

def GetMobileSocName():
  return GetGlobalMobileSocName()

class OpDataUtils(object):
  def __init__(self):
    self._model_names = []
    self._col_names = {}
    self._file_date_texts = []

  def _build_filename(self, path, model_name, date_text):
    return os.path.join(path, model_name + '_' + date_text + '.csv')

  def _check_label_len(self, len_labels, len_req):
    if len_labels != len_req:
      raise ValueError('Require {} labels, but we got {} labels'.format(
          len_req, len_labels))

  def _build_data_text(self, data, i):
    def append_data_text(text, data_list):
      if len(data_list) == 0:
        return text
      for j in range(len(data_list) - 1):
        text = text + str(data_list[j]) + ','
      text = text + str(data_list[-1])
      return text
    data_text = ''
    data_text = append_data_text(data_text, data['train_vars'][i]) + ','
    data_text = append_data_text(data_text, data['sd_vars'][i]) + ','
    data_text = append_data_text(data_text, data['partition_vars'][i]) + ','
    data_text = append_data_text(data_text, data['feature_vars'][i])

    return data_text

  def collect_granularity(self):
    return None

  def create_file(self, path, model_name, interact=True):
    date_text = FileLogger.current_date_text()
    out_filename = self._build_filename(path, model_name, date_text)
    if os.path.exists(out_filename):
      if interact:
        answer = input('File {} exists, delete it? (y/n): '.format(out_filename))
      else:
        answer = 'n'
    else:
      answer = 'y'
    if answer == 'y':
      FileUtils.create_dir(path)
      with open(out_filename, 'w') as f:
        TerminalLogger.log(LogTag.INFO, 'Create file ' + out_filename)
        f.write(self._col_names[model_name] + '\n')
        self._file_date_texts.append(date_text)

  def create_all_files(self, path):
    for name in self._model_names:
      self.create_file(path, name)

  def remove_unused_files(self, path):
    for i in range(len(self._model_names)):
      model_name = self._model_names[i]
      date_text = self._file_date_texts[i]
      out_filename = self._build_filename(path, model_name, date_text)
      if FileUtils.count_line(out_filename) == 1:
        TerminalLogger.log(LogTag.INFO, 'Remove unused file ' + out_filename)
        os.remove(out_filename)

  def print_title(self):
    col_text = self._col_names[self._model_names[0]]
    TerminalLogger.log(LogTag.INFO, col_text)

  def print_data(self, data, cnn_name=None):
    for i in range(len(data['model_names'])):
      model_name = data['model_names'][i]
      cnn_name = None
      model_name = None
      data_text = ''
      if cnn_name is not None:
        data_text = data_text + cnn_name + ','
      if model_name is not None:
        data_text = data_text + model_name + ','
      data_text = data_text + self._build_data_text(data, i)
      TerminalLogger.log(LogTag.INFO, data_text)

  def write_data(self, path, data):
    for i in range(len(data['model_names'])):
      date_text = self._file_date_texts[i]
      model_name = data['model_names'][i]
      out_filename = self._build_filename(path, model_name, date_text)
      with open(out_filename, 'a+') as f:
        data_text = self._build_data_text(data, i)
        f.write(data_text + '\n')

  def extract(self, op_param, device, do_data_transform=False, lat_values=None):
    return None

''' Conv2D '''
class Conv2dDataUtils(OpDataUtils):
  def extract(self, conv2d_param, device,
              do_data_transform=False, lat_values=None):
    return OpDataUtils.extract(self, conv2d_param, device,
                               do_data_transform, lat_values)

class LRConv2dDataUtils(Conv2dDataUtils):
  def __init__(self):
    Conv2dDataUtils.__init__(self)

  def collect_granularity(self):
    return 'coarse'

  def _extract_all(self, conv2d_param, lat_values):
    return None

  def _extract_cpu(self, conv2d_param, lat_values):
    return None

  def _extract_gpu(self, conv2d_param, lat_values):
    return None

  def extract(self, conv2d_param, device,
              do_data_transform=False, lat_values=None):
    if device == Device.CPU:
      return self._extract_cpu(conv2d_param, lat_values)
    elif device == Device.GPU:
      return self._extract_gpu(conv2d_param, lat_values)
    else:
      raise ValueError('Unsupported device: ' + device)
      return None

class FLOPsLRConv2dDataUtils(LRConv2dDataUtils):
  def __init__(self):
    LRConv2dDataUtils.__init__(self)
    conv2d_dataset_info = FLOPsLRConv2dDatasetInfo()
    self._model_names = conv2d_dataset_info.model_names()
    self._col_names = conv2d_dataset_info.col_names()

  def _extract_all(self, conv2d_param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      t_comp = lat_values[0]
    else:
      t_comp = 0

    # FLOPs is used as the feature.
    flops = CalcConvFlops(conv2d_param['filter_shape'],
                          conv2d_param['output_shape'])

    feature_list = [flops]

    ml_data = {}
    ml_data['model_names'] = ['t_flops_cpu_gpu']
    ml_data['train_vars'] = [[t_comp]]
    ml_data['feature_vars'] = [feature_list]

    return ml_data

  def _extract_cpu(self, conv2d_param, lat_values=None):
    ml_data = self._extract_all(conv2d_param, lat_values)
    ml_data['model_names'] = ['t_flops_cpu']
    return ml_data

  def _extract_gpu(self, conv2d_param, lat_values=None):
    ml_data = self._extract_all(conv2d_param, lat_values)
    ml_data['model_names'] = ['t_flops_gpu']
    return ml_data

class MuLayerLRConv2dDataUtils(LRConv2dDataUtils):
  def __init__(self):
    LRConv2dDataUtils.__init__(self)
    conv2d_dataset_info = MuLayerLRConv2dDatasetInfo()
    self._model_names = conv2d_dataset_info.model_names()
    self._col_names = conv2d_dataset_info.col_names()

  def _extract_all(self, conv2d_param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      t_comp = lat_values[0]
    else:
      t_comp = 0

    h = conv2d_param['input_shape'][1]
    w = conv2d_param['input_shape'][2]
    # Input channel, i.e. the number of feature in the input feature maps.
    ic = conv2d_param['input_shape'][3]
    # Filter size.
    fs = conv2d_param['filter_shape'][2]
    # Output channel, i.e. the number of filters.
    oc = conv2d_param['filter_shape'][0]
    # Stride.
    stride = conv2d_param['strides'][0]

    # FLOPs.
    flops = CalcConvFlops(conv2d_param['filter_shape'],
                          conv2d_param['output_shape'])

    # Two features used in muLayer.
    #feature_a = ic
    feature_a = h * w * ic
    feature_b = np.square(float(fs) / stride) * oc
    feature_list = [flops, feature_a, feature_b]

    ml_data = {}
    ml_data['model_names'] = ['t_mulayer_conv2d_cpu_gpu']
    ml_data['train_vars'] = [[t_comp]]
    ml_data['feature_vars'] = [feature_list]

    return ml_data

  def _extract_cpu(self, conv2d_param, lat_values=None):
    ml_data = self._extract_all(conv2d_param, lat_values)
    ml_data['model_names'] = ['t_mulayer_conv2d_cpu']
    return ml_data

  def _extract_gpu(self, conv2d_param, lat_values=None):
    ml_data = self._extract_all(conv2d_param, lat_values)
    ml_data['model_names'] = ['t_mulayer_conv2d_gpu']
    return ml_data

class MLConv2dDataBaseUtils():
  @staticmethod
  def GetConv2dKeyParam(conv2d_param):
    flops = CalcConvFlops(conv2d_param['filter_shape'],
                          conv2d_param['output_shape'])
    params = CalcConvParams(conv2d_param['filter_shape'])
    key_param = {'flops': flops, 'params': params}
    return key_param

  @staticmethod
  def GetDataTransformKeyParam(conv2d_param):
    key_param = {'ih': conv2d_param['input_shape'][1],
                 'iw': conv2d_param['input_shape'][2],
                 'ic': conv2d_param['input_shape'][3],
                 'oh': conv2d_param['output_shape'][1],
                 'ow': conv2d_param['output_shape'][2],
                 'oc': conv2d_param['output_shape'][3],
                 'data_in': conv2d_param['input_shape'][1] \
                    * conv2d_param['input_shape'][2] \
                    * conv2d_param['input_shape'][3],
                 'data_out': conv2d_param['output_shape'][1] \
                    * conv2d_param['output_shape'][2] \
                    * conv2d_param['output_shape'][3]}

    return key_param

  @staticmethod
  def GetCpuDirectKeyParamInternal(ih, iw, ic, oc, kh, kw, sh, sw):
    #print('ih {} iw {} ic {} oc {} kh {} kw {} sh {} sw {}'.format(
    #    ih, iw, ic, oc, kh, kw, sh, sw))

    b = 1
    pbh, pbw = GetConv2dCpuDirectPaddingBlockSize(kh, sh)
    ih, oh = ConvPoolCalcUtils.CalcOutSize(ih, kh, sh, pbh)
    iw, ow = ConvPoolCalcUtils.CalcOutSize(iw, kw, sw, pbw)
    filter_shape = [oc, ic, kh, kw]
    output_shape = [b, oh, ow, oc]

    # FLOPs per output channel.
    flops = CalcConvFlops(filter_shape, output_shape)
    flops_per_oc = flops // oc
    # Output channels per item.
    oc_per_item = GetConv2dCpuDirectOcPerItem(kh)
    # Tile of thread.
    num_threads = GetGlobalThreadCount()
    threadpool = ThreadPool(num_threads)
    threadpool_info = threadpool.compute_2d(
        0, b, 1,
        0, oc, oc_per_item)
    tile_size = threadpool_info['tile_size']
    max_thread_tiles = threadpool_info['max_tile_in_threads']
    if num_threads == 1:
      oc_per_item = 1
    # FLOPs per tile.
    flops_per_tile = tile_size * oc_per_item * flops_per_oc

    key_param = {'flops': flops,
                 'tile_size': tile_size,
                 'max_thread_tiles': max_thread_tiles,
                 'flops_per_tile': flops_per_tile}

    return key_param

  @staticmethod
  def GetCpuDirectKeyParam(conv2d_param):
    ih = conv2d_param['input_shape'][1]
    iw = conv2d_param['input_shape'][2]
    ic = conv2d_param['input_shape'][3]
    oc = conv2d_param['filter_shape'][0]
    kh = conv2d_param['filter_shape'][2]
    kw = conv2d_param['filter_shape'][3]
    sh = conv2d_param['strides'][0]
    sw = conv2d_param['strides'][1]
    return MLConv2dDataBaseUtils.GetCpuDirectKeyParamInternal(
        ih, iw, ic, oc, kh, kw, sh, sw)

  @staticmethod
  def GetCpuGemmKeyParamInternal(m, k, n):
    MB = 8; KB = 4; NB = 8

    mb = RoundUpDiv(m, MB); kb = RoundUpDiv(k, KB); nb = RoundUpDiv(n, NB)

    # Thread pool.
    num_threads = GetGlobalThreadCount()
    threadpool = ThreadPool(num_threads)
    
    threadpool_info = threadpool.compute_1d(0, mb, 1)
    tile_size = threadpool_info['tile_size']
    max_thread_mb = threadpool_info['max_tile_in_threads'] * tile_size
    
    threadpool_info = threadpool.compute_1d(0, nb, 1)
    tile_size = threadpool_info['tile_size']
    max_thread_nb = threadpool_info['max_tile_in_threads'] * tile_size

    key_param = {'m': m, 'k': k, 'n': n,
                 'MB': MB, 'KB': KB, 'NB': NB,
                 'mb': mb, 'kb': kb, 'nb': nb,
                 'max_thread_mb': max_thread_mb,
                 'max_thread_nb': max_thread_nb}

    return key_param

  @staticmethod
  def GetCpuGemmKeyParam(conv2d_param):
    # Block size of GEMM.
    m = conv2d_param['filter_shape'][0]
    k = conv2d_param['filter_shape'][1]
    n = conv2d_param['input_shape'][1] * conv2d_param['input_shape'][2]
    return MLConv2dDataBaseUtils.GetCpuGemmKeyParamInternal(m, k, n)

  @staticmethod
  def BuildGemmParamFromWinogradParam(conv2d_param, wino_key_param):
    gemm_param = {}
    gemm_param['input_shape'] = [1, 1, wino_key_param['total_out_tile_count'],
                                 conv2d_param['input_shape'][3]]
    gemm_param['filter_shape'] = [conv2d_param['filter_shape'][0],
                                  conv2d_param['filter_shape'][1], 1, 1]
    gemm_param['strides'] = [1, 1]
    gemm_param['padding_type'] = 0
    gemm_param['paddings'] = [0, 0]
    gemm_param['dilations'] = [1, 1]
    return gemm_param

  @staticmethod
  def GetCpuWinogradKeyParamInternal(ot_size, pad_ih, pad_iw,
                                     pad_oh, pad_ow, ic, oc):
    out_tile_counts = [pad_oh // ot_size, pad_ow // ot_size]
    total_out_tile_count = out_tile_counts[0] * out_tile_counts[1]

    # Thread pool.
    num_threads = GetGlobalThreadCount()
    threadpool = ThreadPool(num_threads)

    threadpool_info = threadpool.compute_2d(0, 1, 1, 0, ic, 1)
    tile_size = threadpool_info['tile_size']
    max_thread_ic = threadpool_info['max_tile_in_threads'] * tile_size
    
    threadpool_info = threadpool.compute_2d(0, 1, 1, 0, oc, 1)
    tile_size = threadpool_info['tile_size']
    max_thread_oc = threadpool_info['max_tile_in_threads'] * tile_size
    
    key_param = {'out_tile_size': ot_size,
                 'total_out_tile_count': total_out_tile_count,
                 'max_thread_ic': max_thread_ic,
                 'max_thread_oc': max_thread_oc,
                 'pad_ih': pad_ih,
                 'pad_iw': pad_iw,
                 'pad_oh': pad_oh,
                 'pad_ow': pad_ow}

    return key_param

  @staticmethod
  def GetCpuWinogradKeyParam(conv2d_param):
    # Winograd-related size.
    out_tile_size = CalcConv2dWinogradOutTileSize(conv2d_param['input_shape'],
                                                  conv2d_param['output_shape'],
                                                  conv2d_param['padding_type'])
    pad_in_shape, pad_out_shape = CalcConv2dWinogradShape(
        conv2d_param['input_shape'], conv2d_param['output_shape'], out_tile_size)
    
    return MLConv2dDataBaseUtils.GetCpuWinogradKeyParamInternal(
        out_tile_size, pad_in_shape[1], pad_in_shape[2], pad_out_shape[1], pad_out_shape[2],
        conv2d_param['input_shape'][3], conv2d_param['filter_shape'][0])

  @staticmethod
  def GetGpuDirectKeyParamInternalWarpMode(ic, oh, ow, oc, ks, stride):
    # Block size.
    kChannelBlockSize = Conv2dOpenCLUtils.GetChannelBlockSize()
    kWidthBlockSize = Conv2dOpenCLUtils.GetWidthBlockSize(ks)

    # Workgroup size.
    ocb = RoundUpDiv(oc, kChannelBlockSize)
    owb = RoundUpDiv(ow, kWidthBlockSize)
    gws = [ocb, owb, oh]
    lws = Conv2dOpenCLUtils.DefaultLocalWS(
              gws, kWidthBlockSize, ks, GetMobileSocName())

    # Number of warps.
    n_warp = CalcWarpNumber(gws, lws, GetMobileSocName())

    # Other features.
    icb = RoundUpDiv(ic, kChannelBlockSize)

    key_param = {'n_warp': n_warp, 'oh': oh, 'owb': owb, 'icb': icb, 'ocb': ocb,
                 'ks': ks, 'stride': stride}

    return key_param

  @staticmethod
  def GetGpuDirectKeyParamInternal(ic, oh, ow, oc, ks, stride):
    return MLConv2dDataBaseUtils.GetGpuDirectKeyParamInternalWarpMode(
        ic, oh, ow, oc, ks, stride)

  @staticmethod
  def GetGpuDirectKeyParam(conv2d_param):
    return MLConv2dDataBaseUtils.GetGpuDirectKeyParamInternal(
        conv2d_param['input_shape'][3], conv2d_param['output_shape'][1],
        conv2d_param['output_shape'][2], conv2d_param['output_shape'][3],
        conv2d_param['filter_shape'][2], conv2d_param['strides'][0])

class MLConv2dDataUtils(Conv2dDataUtils):
  def __init__(self):
    Conv2dDataUtils.__init__(self)
    conv2d_dataset_info = MLConv2dDatasetInfoV2()
    self._model_names = conv2d_dataset_info.model_names()
    self._col_names = conv2d_dataset_info.col_names()

  def collect_granularity(self):
    return 'fine'

  def _extract_data_transform(self, conv2d_param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 8)
      t_dt_in = lat_values[0]
      t_map_in = lat_values[1]
      t_map_out = lat_values[2]
      t_sync = lat_values[3]
      t_unmap_in = lat_values[4]
      t_unmap_out = lat_values[5]
      t_dt_out = lat_values[6]
      t_sync_v2 = lat_values[7]
    else:
      t_dt_in = 0
      t_map_in = 0
      t_map_out = 0
      t_sync = 0
      t_unmap_in = 0
      t_unmap_out = 0
      t_dt_out = 0
      t_sync_v2 = 0

    key_param = MLConv2dDataBaseUtils.GetDataTransformKeyParam(conv2d_param)

    ml_data = {}
    ml_data['model_names'] = ['t_data_sharing']
    ml_data['train_vars'] = [[t_dt_in, t_map_in, t_map_out, t_sync,
                              t_unmap_in, t_unmap_out, t_dt_out, t_sync_v2]]
    ml_data['feature_vars'] = [[key_param['ih'], key_param['iw'], key_param['ic'],
                                key_param['oh'], key_param['ow'], key_param['oc'],
                                key_param['data_in'], key_param['data_out']]]

    return ml_data

  def _extract_cpu_direct(self, conv2d_param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 3)
      cost_comp = lat_values[1]
    else:
      cost_comp = 0

    key_param = MLConv2dDataBaseUtils.GetCpuDirectKeyParam(conv2d_param)

    # Cost of computing a FLOPs.
    t_flops = cost_comp / (key_param['max_thread_tiles'] \
        * key_param['flops_per_tile'])

    ml_data = {}
    ml_data['model_names'] = ['t_conv2d_cpu_direct']
    ml_data['train_vars'] = [[conv2d_param['nn_name'], cost_comp, t_flops]]
    ml_data['feature_vars'] = [[conv2d_param['input_shape'][1],
                                conv2d_param['input_shape'][2],
                                conv2d_param['input_shape'][3],
                                conv2d_param['filter_shape'][0],
                                conv2d_param['filter_shape'][2],
                                conv2d_param['filter_shape'][3],
                                conv2d_param['strides'][0],
                                conv2d_param['strides'][1],
                                key_param['max_thread_tiles'],
                                key_param['flops_per_tile']]]

    return ml_data

  @staticmethod
  def extract_cpu_gemm(key_param, lat_values=None):
    if lat_values is not None:
      cost_pack_lb = lat_values[0]
      cost_pack_rb = lat_values[1]
      cost_comp = lat_values[2]
    else:
      cost_pack_lb = 0
      cost_pack_rb = 0
      cost_comp = 0
    
    # Time of packing a left block data.
    t_pack_lb = cost_pack_lb / (key_param['max_thread_mb'] * key_param['kb'])
    # Time of packing a right block data.
    t_pack_rb = cost_pack_rb / (key_param['kb'] * key_param['max_thread_nb'])
    # Time of computing a block data.
    t_comp = cost_comp / (key_param['max_thread_mb'] * key_param['kb'] * key_param['nb'])

    ml_data = {}
    ml_data['model_names'] = ['t_conv2d_cpu_gemm']
    ml_data['train_vars'] = [[key_param['nn_name'],
                              cost_pack_lb, cost_pack_rb, cost_comp,
                              t_pack_lb, t_pack_rb, t_comp]]
    ml_data['feature_vars'] = [[key_param['m'], key_param['k'], key_param['n'],
                                key_param['MB'], key_param['KB'], key_param['NB'],
                                key_param['max_thread_mb'], key_param['max_thread_nb']]]

    return ml_data

  def _extract_cpu_gemm(self, conv2d_param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 3)

    key_param = MLConv2dDataBaseUtils.GetCpuGemmKeyParam(conv2d_param)
    key_param['nn_name'] = conv2d_param['nn_name']

    return MLConv2dDataUtils.extract_cpu_gemm(key_param, lat_values)

  def _extract_cpu_winograd(self, conv2d_param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 5)
      cost_pad = lat_values[0]
      cost_tr_in = lat_values[1]
      cost_tr_out = lat_values[3]
      cost_unpad = lat_values[4]
    else:
      cost_pad = 0
      cost_tr_in = 0
      cost_tr_out = 0
      cost_unpad = 0

    key_param = MLConv2dDataBaseUtils.GetCpuWinogradKeyParam(conv2d_param)

    # Time of pad/unpad a data.
    t_pad = cost_pad / (conv2d_param['input_shape'][1] * \
        conv2d_param['input_shape'][2] * conv2d_param['input_shape'][3])
    t_unpad = cost_unpad / (conv2d_param['output_shape'][1] * \
        conv2d_param['output_shape'][2] * conv2d_param['output_shape'][3])

    # Time of transforming a input block data.
    t_tr_in = cost_tr_in / (key_param['max_thread_ic'] * \
        key_param['total_out_tile_count'])
    # Time of transforming a output block data.
    t_tr_out = cost_tr_out / (key_param['max_thread_oc'] * \
        key_param['total_out_tile_count'])

    ml_data = {}
    ml_data['model_names'] = ['t_conv2d_cpu_winograd']
    ml_data['train_vars'] = [[conv2d_param['nn_name'],
                              cost_pad, cost_tr_in, cost_tr_out, cost_unpad,
                              t_pad, t_tr_in, t_tr_out, t_unpad]]
    ml_data['feature_vars'] = [[conv2d_param['input_shape'][1],
                                conv2d_param['input_shape'][2],
                                conv2d_param['input_shape'][3],
                                conv2d_param['output_shape'][1],
                                conv2d_param['output_shape'][2],
                                conv2d_param['output_shape'][3],
                                key_param['pad_ih'], key_param['pad_iw'],
                                key_param['pad_oh'], key_param['pad_ow'],
                                key_param['out_tile_size'],
                                key_param['max_thread_ic'], key_param['max_thread_oc'],
                                key_param['total_out_tile_count']]]

    return ml_data

  def _extract_cpu_winograd_gemm(self, conv2d_param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 5)
      cost_gemm = lat_values[2]
    else:
      cost_gemm = 0

    key_param = MLConv2dDataBaseUtils.GetCpuWinogradKeyParam(conv2d_param)
    out_tile_size = key_param['out_tile_size']
    total_in_tile_size = pow((out_tile_size + 2), 2)
    cost_gemm = cost_gemm / total_in_tile_size

    gemm_param = MLConv2dDataBaseUtils.BuildGemmParamFromWinogradParam(
                    conv2d_param, key_param)
    gemm_param['nn_name'] = conv2d_param['nn_name']

    ml_data = self._extract_cpu_gemm(gemm_param, [0, 0, cost_gemm])
    ml_data['model_names'] = ['t_conv2d_cpu_winograd_gemm']
    ml_data['feature_vars'][0].append(out_tile_size)

    return ml_data

  def _extract_gpu_direct_warp(self, conv2d_param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost_warp = lat_values[0]
    else:
      cost_warp = 0

    key_param = MLConv2dDataBaseUtils.GetGpuDirectKeyParam(conv2d_param)

    # Time of computing a warp.
    t_warp = cost_warp / key_param['n_warp']
    t_warp_icb = t_warp / key_param['icb']

    ml_data = {}
    ml_data['model_names'] = ['t_conv2d_gpu_direct']
    ml_data['train_vars'] = [[conv2d_param['nn_name'], cost_warp, t_warp, t_warp_icb]]
    ml_data['feature_vars'] = [[key_param['oh'], key_param['owb'], key_param['icb'],
                                key_param['ocb'], key_param['ks'], key_param['stride'],
                                key_param['n_warp']]]

    return ml_data

  def _extract_gpu_direct(self, conv2d_param, lat_values=None):
    return self._extract_gpu_direct_warp(conv2d_param, lat_values)

  def extract(self, conv2d_param, device, do_data_transform=False, lat_values=None):
    if device == Device.CPU:
      if do_data_transform:
        return self._extract_data_transform(conv2d_param, lat_values)
      filter_shape = conv2d_param['filter_shape']
      strides = conv2d_param['strides']
      is_wino_gemm = conv2d_param['is_wino_gemm']
      impl_type = GetConv2dCpuImplType(filter_shape, strides, is_wino_gemm)
      if impl_type == Conv2dCpuImplType.GEMM:
        return self._extract_cpu_gemm(conv2d_param, lat_values)
      elif impl_type == Conv2dCpuImplType.WINOGRAD:
        return self._extract_cpu_winograd(conv2d_param, lat_values)
      elif impl_type == Conv2dCpuImplType.WINOGRAD_GEMM:
        return self._extract_cpu_winograd_gemm(conv2d_param, lat_values)
      else:
        return self._extract_cpu_direct(conv2d_param, lat_values)
    elif device == Device.GPU:
      return self._extract_gpu_direct(conv2d_param, lat_values)
    else:
      raise ValueError('Unsupported device: ' + device)
      return None

class MLConv2dOpDataUtilsBase(Conv2dDataUtils):
  def __init__(self):
    Conv2dDataUtils.__init__(self)

  def collect_granularity(self):
    return 'coarse'

  def _extract_general(self, conv2d_param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost_op = lat_values[0]
    else:
      cost_op = 0

    key_param = MLConv2dDataBaseUtils.GetConv2dKeyParam(conv2d_param)

    ml_data = {}
    ml_data['model_names'] = None
    ml_data['train_vars'] = [[conv2d_param['nn_name'], cost_op]]
    ml_data['feature_vars'] = [[conv2d_param['input_shape'][1],
                                conv2d_param['input_shape'][2],
                                conv2d_param['input_shape'][3],
                                conv2d_param['filter_shape'][0],
                                conv2d_param['filter_shape'][2],
                                conv2d_param['filter_shape'][3],
                                conv2d_param['strides'][0],
                                conv2d_param['strides'][1],
                                conv2d_param['dilations'][0],
                                conv2d_param['dilations'][0],
                                key_param['flops'],
                                key_param['params']]]

    return ml_data

class MLConv2dOpDataUtils(MLConv2dOpDataUtilsBase):
  def __init__(self):
    MLConv2dOpDataUtilsBase.__init__(self)
    conv2d_dataset_info = MLConv2dOpDatasetInfo()
    self._model_names = conv2d_dataset_info.model_names()
    self._col_names = conv2d_dataset_info.col_names()

  def _extract_cpu(self, conv2d_param, lat_values=None):
    ml_data = self._extract_general(conv2d_param, lat_values)
    ml_data['model_names'] = ['t_conv2d_op_cpu']
    return ml_data

  def _extract_gpu(self, conv2d_param, lat_values=None):
    ml_data = self._extract_general(conv2d_param, lat_values)
    ml_data['model_names'] = ['t_conv2d_op_gpu']
    return ml_data

  def extract(self, conv2d_param, device,
              do_data_transform=False, lat_values=None):
    if device == Device.CPU:
      return self._extract_cpu(conv2d_param, lat_values)
    elif device == Device.GPU:
      return self._extract_gpu(conv2d_param, lat_values)
    else:
      raise ValueError('Unsupported device: ' + device)
      return None

class MLConv2dOpImplTypeDataUtils(MLConv2dOpDataUtilsBase):
  def __init__(self):
    MLConv2dOpDataUtilsBase.__init__(self)
    conv2d_dataset_info = MLConv2dOpImplTypeDatasetInfo()
    self._model_names = conv2d_dataset_info.model_names()
    self._col_names = conv2d_dataset_info.col_names()

  def _extract_cpu_direct(self, conv2d_param, lat_values=None):
    ml_data = self._extract_general(conv2d_param, lat_values)
    ml_data['model_names'] = ['t_op_cpu_direct']
    return ml_data

  def _extract_cpu_gemm(self, conv2d_param, lat_values=None):
    ml_data = self._extract_general(conv2d_param, lat_values)
    ml_data['model_names'] = ['t_op_cpu_gemm']
    return ml_data

  def _extract_cpu_winograd(self, conv2d_param, lat_values=None):
    ml_data = self._extract_general(conv2d_param, lat_values)
    ml_data['model_names'] = ['t_op_cpu_winograd']
    return ml_data

  def _extract_gpu_direct(self, conv2d_param, lat_values=None):
    ml_data = self._extract_general(conv2d_param, lat_values)
    ml_data['model_names'] = ['t_op_gpu_direct']
    return ml_data

  def extract(self, conv2d_param, device, do_data_transform=False, lat_values=None):
    if device == Device.CPU:
      filter_shape = conv2d_param['filter_shape']
      strides = conv2d_param['strides']
      if GetConv2dCpuImplType(filter_shape, strides) == Conv2dCpuImplType.GEMM:
        return self._extract_cpu_gemm(conv2d_param, lat_values)
      elif GetConv2dCpuImplType(filter_shape, strides) == Conv2dCpuImplType.WINOGRAD:
        return self._extract_cpu_winograd(conv2d_param, lat_values)
      else:
        return self._extract_cpu_direct(conv2d_param, lat_values)
    elif device == Device.GPU:
      return self._extract_gpu_direct(conv2d_param, lat_values)
    else:
      raise ValueError('Unsupported device: ' + device)
      return None

''' Pooling '''
class PoolingDataUtils(OpDataUtils):
  def extract(self, param, device, do_data_transform=False, lat_values=None):
    return OpDataUtils.extract(self, param, device, do_data_transform, lat_values)

class MuLayerLRPoolingDataUtils(PoolingDataUtils):
  def __init__(self):
    PoolingDataUtils.__init__(self)
    dataset_info = MuLayerLRPoolingDatasetInfo()
    self._model_names = dataset_info.model_names()
    self._col_names = dataset_info.col_names()

  def collect_granularity(self):
    return 'coarse'

  def _extract_all(self, param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      t_comp = lat_values[0]
    else:
      t_comp = 0

    # Input size and output size are used as the features.
    in_size = param['input_shape'][1] * param['input_shape'][2] * param['input_shape'][3]
    out_size = param['output_shape'][1] * param['output_shape'][2] * param['output_shape'][3]
    feature_list = [in_size, out_size]

    ml_data = {}
    ml_data['model_names'] = ['t_mulayer_pooling_cpu_gpu']
    ml_data['train_vars'] = [[t_comp]]
    ml_data['feature_vars'] = [feature_list]

    return ml_data

  def _extract_cpu(self, param, lat_values=None):
    ml_data = self._extract_all(param, lat_values)
    ml_data['model_names'] = ['t_mulayer_pooling_cpu']
    return ml_data

  def _extract_gpu(self, param, lat_values=None):
    ml_data = self._extract_all(param, lat_values)
    ml_data['model_names'] = ['t_mulayer_pooling_gpu']
    return ml_data

  def extract(self, param, device, do_data_transform=False, lat_values=None):
    if device == Device.CPU:
      return self._extract_cpu(param, lat_values)
    elif device == Device.GPU:
      return self._extract_gpu(param, lat_values)
    else:
      raise ValueError('Unsupported device: ' + device)
      return None

class MLPoolingDataUtils(PoolingDataUtils):
  def __init__(self):
    PoolingDataUtils.__init__(self)
    dataset_info = MLPoolingDatasetInfo()
    self._model_names = dataset_info.model_names()
    self._col_names = dataset_info.col_names()

  def collect_granularity(self):
    return 'fine'

  def _extract_cpu_direct_avg(self, param, lat_values):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost = lat_values[0]
    else:
      cost = 0

    t = 0

    ml_data = {}
    ml_data['model_names'] = ['t_pooling_cpu_direct_avg']
    ml_data['train_vars'] = [[param['nn_name'], cost, t]]
    ml_data['feature_vars'] = [[param['input_shape'][1], param['input_shape'][2],
                                param['input_shape'][3], param['input_shape'][3],
                                param['filter_shape'][2], param['filter_shape'][3],
                                param['strides'][0], param['strides'][1]]]

    return ml_data

  def _extract_cpu_direct_max(self, param, lat_values):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost = lat_values[0]
    else:
      cost = 0

    t = 0

    ml_data = {}
    ml_data['model_names'] = ['t_pooling_cpu_direct_max']
    ml_data['train_vars'] = [[param['nn_name'], cost, t]]
    ml_data['feature_vars'] = [[param['input_shape'][1], param['input_shape'][2],
                                param['input_shape'][3], param['input_shape'][3],
                                param['filter_shape'][2], param['filter_shape'][3],
                                param['strides'][0], param['strides'][1]]]

    return ml_data

  def _extract_gpu_direct_avg(self, param, lat_values):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost = lat_values[0]
    else:
      cost = 0

    t = 0

    ml_data = {}
    ml_data['model_names'] = ['t_pooling_gpu_direct_avg']
    ml_data['train_vars'] = [[param['nn_name'], cost, t]]
    ml_data['feature_vars'] = [[param['input_shape'][1], param['input_shape'][2],
                                param['input_shape'][3], param['input_shape'][3],
                                param['filter_shape'][2], param['filter_shape'][3],
                                param['strides'][0], param['strides'][1]]]

    return ml_data

  def _extract_gpu_direct_max(self, param, lat_values):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost = lat_values[0]
    else:
      cost = 0

    t = 0

    ml_data = {}
    ml_data['model_names'] = ['t_pooling_gpu_direct_max']
    ml_data['train_vars'] = [[param['nn_name'], cost, t]]
    ml_data['feature_vars'] = [[param['input_shape'][1], param['input_shape'][2],
                                param['input_shape'][3], param['input_shape'][3],
                                param['filter_shape'][2], param['filter_shape'][3],
                                param['strides'][0], param['strides'][1]]]

    return ml_data

  def extract(self, param, device, do_data_transform=False, lat_values=None):
    pooling_type = param['pooling_type']
    if device == Device.CPU:
      if do_data_transform:
        return None
      if pooling_type == PoolingType.AVG.value:
        return self._extract_cpu_direct_avg(param, lat_values)
      elif pooling_type == PoolingType.MAX.value:
        return self._extract_cpu_direct_max(param, lat_values)
      else:
        raise ValueError('Unsupported pooling type: ' + pooling_type)
        return None
    elif device == Device.GPU:
      if pooling_type == PoolingType.AVG.value:
        return self._extract_gpu_direct_avg(param, lat_values)
      elif pooling_type == PoolingType.MAX.value:
        return self._extract_gpu_direct_max(param, lat_values)
      else:
        raise ValueError('Unsupported pooling type: ' + pooling_type)
        return None
    else:
      raise ValueError('Unsupported device: ' + device)
      return None

''' FullyConnected '''
class MLFullyConnectedDataBaseUtils():
  @staticmethod
  def GetCpuGemvKeyParamInternal(m, k):
    MB = 4; KB = 8

    mb = RoundUpDiv(m, MB); kb = RoundUpDiv(k, KB)

    # Thread pool.
    num_threads = GetGlobalThreadCount()
    threadpool = ThreadPool(num_threads)
    
    threadpool_info = threadpool.compute_2d(0, 1, 1, 0, mb, 1)
    tile_size = threadpool_info['tile_size']
    max_thread_mb = threadpool_info['max_tile_in_threads'] * tile_size

    key_param = {'m': m, 'k': k,
                 'MB': MB, 'KB': KB,
                 'mb': mb, 'kb': kb,
                 'max_thread_mb': max_thread_mb}

    return key_param

  @staticmethod
  def GetCpuGemvKeyParam(param):
    m = param['weight_shape'][0]
    k = param['weight_shape'][1] * param['input_shape'][1] * param['input_shape'][2]
    return MLFullyConnectedDataBaseUtils.GetCpuGemvKeyParamInternal(m, k)

  @staticmethod
  def GetGpuDirectKeyParamInternalWarpMode(ih, iw, ic, oc):
    # Block size.
    kChannelBlockSize = FullyConnectedOpenCLUtils.GetChannelBlockSize()

    # Workgroup size.
    ocb = RoundUpDiv(oc, kChannelBlockSize)
    gws, lws = FullyConnectedOpenCLUtils.DefaultLocalWS(ocb, GetMobileSocName())

    # Number of warps.
    n_warp = CalcWarpNumber(gws, lws, GetMobileSocName())

    # Other features.
    iwb = gws[1]
    iwb_size = RoundUpDiv(iw, iwb)
    icb = RoundUpDiv(ic, kChannelBlockSize)
    ocb = RoundUpDiv(oc, kChannelBlockSize)

    key_param = {'n_warp': n_warp, 'ih': ih, 'iw': iw, 'iwb_size': iwb_size,
                 'ic': ic, 'icb': icb, 'oc': oc, 'ocb': ocb}

    return key_param

  @staticmethod
  def GetGpuDirectKeyParamInternal(ih, iw, ic, oc):
    return MLFullyConnectedDataBaseUtils.GetGpuDirectKeyParamInternalWarpMode(
        ih, iw, ic, oc)

  @staticmethod
  def GetGpuDirectKeyParam(param):
    return MLFullyConnectedDataBaseUtils.GetGpuDirectKeyParamInternal(
        param['input_shape'][1], param['input_shape'][2],
        param['input_shape'][3], param['weight_shape'][0])

  @staticmethod
  def GetOpKeyParam(param):
    flops = CalcFullyConnectedFlops(param['weight_shape'], param['output_shape'])
    params = CalcFullyConnectedParams(param['weight_shape'])
    key_param = {'flops': flops, 'params': params}
    return key_param

class FullyConnectedDataUtils(OpDataUtils):
  def extract(self, param, device, do_data_transform=False, lat_values=None):
    return OpDataUtils.extract(self, param, device, do_data_transform, lat_values)

class MuLayerLRFullyConnectedDataUtils(FullyConnectedDataUtils):
  def __init__(self):
    FullyConnectedDataUtils.__init__(self)
    dataset_info = MuLayerLRFullyConnectedDatasetInfo()
    self._model_names = dataset_info.model_names()
    self._col_names = dataset_info.col_names()

  def collect_granularity(self):
    return 'coarse'

  def _extract_all(self, param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      t_comp = lat_values[0]
    else:
      t_comp = 0

    # Input size and output size are used as the features.
    flops = param['input_shape'][1] * param['input_shape'][2] * param['input_shape'][3] * param['weight_shape'][0]
    in_size = param['input_shape'][1] * param['input_shape'][2] * param['input_shape'][3]
    out_size = param['weight_shape'][0]
    feature_list = [flops, in_size, out_size]

    ml_data = {}
    ml_data['model_names'] = ['t_mulayer_fc_cpu_gpu']
    ml_data['train_vars'] = [[t_comp]]
    ml_data['feature_vars'] = [feature_list]

    return ml_data

  def _extract_cpu(self, param, lat_values=None):
    ml_data = self._extract_all(param, lat_values)
    ml_data['model_names'] = ['t_mulayer_fc_cpu']
    return ml_data

  def _extract_gpu(self, param, lat_values=None):
    ml_data = self._extract_all(param, lat_values)
    ml_data['model_names'] = ['t_mulayer_fc_gpu']
    return ml_data

  def extract(self, param, device, do_data_transform=False, lat_values=None):
    if device == Device.CPU:
      return self._extract_cpu(param, lat_values)
    elif device == Device.GPU:
      return self._extract_gpu(param, lat_values)
    else:
      raise ValueError('Unsupported device: ' + device)
      return None

class MLFullyConnectedDataUtils(FullyConnectedDataUtils):
  def __init__(self):
    FullyConnectedDataUtils.__init__(self)
    dataset_info = MLFullyConnectedDatasetInfo()
    self._model_names = dataset_info.model_names()
    self._col_names = dataset_info.col_names()

  def collect_granularity(self):
    return 'fine'

  def _extract_cpu_gemv(self, param, lat_values):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost_comp = lat_values[0]
    else:
      cost_comp = 0

    key_param = MLFullyConnectedDataBaseUtils.GetCpuGemvKeyParam(param)

    # Time of computing a block data.
    t_comp = cost_comp / (key_param['max_thread_mb'] * key_param['kb'])

    ml_data = {}
    ml_data['model_names'] = ['t_fc_cpu_gemv']
    ml_data['train_vars'] = [[param['nn_name'],
                              cost_comp,
                              t_comp]]
    ml_data['feature_vars'] = [[key_param['m'], key_param['k'],
                                key_param['MB'], key_param['KB'],
                                key_param['max_thread_mb']]]

    return ml_data

  def _extract_gpu_direct(self, param, lat_values):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost_warp = lat_values[0]
    else:
      cost_warp = 0

    key_param = MLFullyConnectedDataBaseUtils.GetGpuDirectKeyParam(param)

    # Time of computing a warp.
    t_warp = cost_warp / key_param['n_warp']
    if key_param['iwb_size'] > 1:
      t_warp_blk = t_warp / (key_param['ih'] * key_param['iwb_size'] * key_param['icb'])
    else:
      t_warp_blk = t_warp / (key_param['ih'] * key_param['iw'] * key_param['icb'])

    ml_data = {}
    ml_data['model_names'] = ['t_fc_gpu_direct']
    ml_data['train_vars'] = [[param['nn_name'], cost_warp, t_warp, t_warp_blk]]
    ml_data['feature_vars'] = [[key_param['ih'],
                                key_param['iw'], key_param['iwb_size'],
                                key_param['ic'], key_param['icb'],
                                key_param['oc'], key_param['ocb'],
                                key_param['n_warp']]]

    return ml_data

  def extract(self, param, device, do_data_transform=False, lat_values=None):
    if device == Device.CPU:
      if do_data_transform:
        return None
      return self._extract_cpu_gemv(param, lat_values)
    elif device == Device.GPU:
      return self._extract_gpu_direct(param, lat_values)
    else:
      raise ValueError('Unsupported device: ' + device)
      return None

class MLFullyConnectedOpDataUtilsBase(FullyConnectedDataUtils):
  def __init__(self):
    FullyConnectedDataUtils.__init__(self)

  def collect_granularity(self):
    return 'coarse'

  def _extract_general(self, param, lat_values=None):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost_op = lat_values[0]
    else:
      cost_op = 0

    key_param = MLFullyConnectedDataBaseUtils.GetOpKeyParam(param)

    ml_data = {}
    ml_data['model_names'] = None
    ml_data['train_vars'] = [[param['nn_name'], cost_op]]
    ml_data['feature_vars'] = [[param['input_shape'][1],
                                param['input_shape'][2],
                                param['input_shape'][3],
                                param['weight_shape'][0],
                                param['weight_shape'][2],
                                param['weight_shape'][3],
                                key_param['flops'], key_param['params']]]

    return ml_data

class MLFullyConnectedOpDataUtils(MLFullyConnectedOpDataUtilsBase):
  def __init__(self):
    MLFullyConnectedOpDataUtilsBase.__init__(self)
    dataset_info = MLFullyConnectedOpDatasetInfo()
    self._model_names = dataset_info.model_names()
    self._col_names = dataset_info.col_names()

  def _extract_cpu(self, param, lat_values=None):
    ml_data = self._extract_general(param, lat_values)
    ml_data['model_names'] = ['t_fc_op_cpu']
    return ml_data

  def _extract_gpu(self, param, lat_values=None):
    ml_data = self._extract_general(param, lat_values)
    ml_data['model_names'] = ['t_fc_op_gpu']
    return ml_data

  def extract(self, param, device, do_data_transform=False, lat_values=None):
    if device == Device.CPU:
      return self._extract_cpu(param, lat_values)
    elif device == Device.GPU:
      return self._extract_gpu(param, lat_values)
    else:
      raise ValueError('Unsupported device: ' + device)
      return None

''' Deconv2D '''
class MLDeconv2dDataBaseUtils():
  @staticmethod
  def GetCpuDirectKeyParamInternal(ih, iw, ic, oc, kh, kw ,sh, sw):
    oh = ConvPoolCalcUtils.CalcInSize(ih, kh, sh)
    ow = ConvPoolCalcUtils.CalcInSize(iw, kw, sw)

    input_shape = [1, ih, iw, ic]
    filter_shape = [oc, ic, kh, kw]
    output_shape = [1, oh, ow, oc]
    # FLOPs per output channel.
    flops = CalcDeconvFlops(input_shape, filter_shape, output_shape)
    flops_per_oc = flops / oc
    # Output channels per item.
    oc_per_item = GetDeconv2dCpuDirectOcPerItem(kh, sh)

    # Thread pool.
    num_threads = GetGlobalThreadCount()
    threadpool = ThreadPool(num_threads)
    
    threadpool_info = threadpool.compute_2d(0, 1, 1, 0, oc, oc_per_item)
    tile_size = threadpool_info['tile_size']
    max_thread_tiles = threadpool_info['max_tile_in_threads'] * tile_size
    if num_threads == 1:
      oc_per_item = 1
    # FLOPs per tile.
    flops_per_tile = tile_size * oc_per_item * flops_per_oc

    key_param = {'flops': flops,
                 'tile_size': tile_size,
                 'max_thread_tiles': max_thread_tiles,
                 'flops_per_tile': flops_per_tile}

    return key_param

  @staticmethod
  def GetCpuDirectKeyParam(param):
    return MLDeconv2dDataBaseUtils.GetCpuDirectKeyParamInternal(
        param['input_shape'][1], param['input_shape'][2], param['input_shape'][3],
        param['filter_shape'][0], param['filter_shape'][2], param['filter_shape'][3],
        param['strides'][0], param['strides'][1])

  @staticmethod
  def GetGpuDirectKeyParamInternalWarpMode(ih, iw, ic, oh, ow, oc, ks, s):
    # Block size.
    kChannelBlockSize = Deconv2dOpenCLUtils.GetChannelBlockSize()
    owb = Deconv2dOpenCLUtils.CalcWidthBlocks(ow, s)

    # Workgroup size.
    icb = RoundUpDiv(ic, kChannelBlockSize)
    ocb = RoundUpDiv(oc, kChannelBlockSize)
    gws = [ocb, owb, oh]
    lws = Deconv2dOpenCLUtils.DefaultLocalWS(gws, GetMobileSocName())

    # Number of warps.
    n_warp = CalcWarpNumber(gws, lws, GetMobileSocName())

    # Other features.

    key_param = {'n_warp': n_warp,
                 'ih': ih, 'iw': iw, 'ic': ic, 'icb': icb,
                 'oh': oh, 'ow': ow, 'owb': owb, 'oc': oc, 'ocb': ocb}

    return key_param

  @staticmethod
  def GetGpuDirectKeyParamInternal(ih, iw, ic, oh, ow, oc, k, s):
    return MLDeconv2dDataBaseUtils.GetGpuDirectKeyParamInternalWarpMode(
        ih, iw, ic, oh, ow, oc, k, s)

  @staticmethod
  def GetGpuDirectKeyParam(param):
    return MLDeconv2dDataBaseUtils.GetGpuDirectKeyParamInternal(
        param['input_shape'][1], param['input_shape'][2], param['input_shape'][3],
        param['output_shape'][1], param['output_shape'][2], param['output_shape'][3],
        param['filter_shape'][2], param['strides'][0])

class Deconv2dDataUtils(OpDataUtils):
  def extract(self, param, device, do_data_transform=False, lat_values=None):
    return OpDataUtils.extract(self, param, device, do_data_transform, lat_values)

class MLDeconv2dDataUtils(Deconv2dDataUtils):
  def __init__(self):
    Deconv2dDataUtils.__init__(self)
    dataset_info = MLDeconv2dDatasetInfo()
    self._model_names = dataset_info.model_names()
    self._col_names = dataset_info.col_names()

  def collect_granularity(self):
    return 'fine'

  def _extract_cpu_direct(self, param, lat_values):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost_comp = lat_values[0]
    else:
      cost_comp = 0

    key_param = MLDeconv2dDataBaseUtils.GetCpuDirectKeyParam(param)

    # Time of computing a block data.
    t_comp = cost_comp / (key_param['max_thread_tiles'] * key_param['flops_per_tile'])

    ml_data = {}
    ml_data['model_names'] = ['t_deconv2d_cpu_direct']
    ml_data['train_vars'] = [[param['nn_name'], cost_comp, t_comp]]
    ml_data['feature_vars'] = [[param['input_shape'][1],
                                param['input_shape'][2],
                                param['input_shape'][3],
                                param['filter_shape'][0],
                                param['filter_shape'][2],
                                param['filter_shape'][3],
                                param['strides'][0],
                                param['strides'][1],
                                key_param['max_thread_tiles'],
                                key_param['flops_per_tile']]]

    return ml_data

  def _extract_gpu_direct(self, param, lat_values):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost_warp = lat_values[0]
    else:
      cost_warp = 0

    key_param = MLDeconv2dDataBaseUtils.GetGpuDirectKeyParam(param)

    # Time of computing a warp.
    t_warp = cost_warp / key_param['n_warp']
    t_warp_icb = t_warp / key_param['icb']

    ml_data = {}
    ml_data['model_names'] = ['t_deconv2d_gpu_direct']
    ml_data['train_vars'] = [[param['nn_name'], cost_warp, t_warp, t_warp_icb]]
    ml_data['feature_vars'] = [[key_param['ih'], key_param['iw'],
                                key_param['ic'], key_param['icb'],
                                key_param['oh'], key_param['ow'], key_param['owb'], 
                                key_param['oc'], key_param['ocb'],
                                param['filter_shape'][2], param['strides'][0],
                                key_param['n_warp']]]

    return ml_data

  def extract(self, param, device, do_data_transform=False, lat_values=None):
    if device == Device.CPU:
      if do_data_transform:
        return None
      return self._extract_cpu_direct(param, lat_values)
    elif device == Device.GPU:
      return self._extract_gpu_direct(param, lat_values)
    else:
      raise ValueError('Unsupported device: ' + device)
      return None

''' MatMul '''
class MLMatMulDataBaseUtils():
  @staticmethod
  def GetGpuDirectKeyParamInternalWarpMode(batch, m, k, n):
    # Block size.
    kMBlockSize = MatMulOpenCLUtils.GetMBlockSize()
    kKBlockSize = MatMulOpenCLUtils.GetKBlockSize()
    kNBlockSize = MatMulOpenCLUtils.GetNBlockSize()

    # Workgroup size.
    mb = RoundUpDiv(m, kMBlockSize)
    kb = RoundUpDiv(k, kKBlockSize)
    nb = RoundUpDiv(n, kNBlockSize)
    gws = [nb, mb * batch]
    lws = MatMulOpenCLUtils.DefaultLocalWS(GetMobileSocName())

    # Number of warps.
    n_warp = CalcWarpNumber(gws, lws, GetMobileSocName())

    # Other features.
    key_param = {'n_warp': n_warp, 'batch': batch,
                 'm': m, 'k': k, 'n': n, 'mb': mb, 'kb': kb, 'nb': nb}

    return key_param

  @staticmethod
  def GetGpuDirectKeyParamInternal(batch, m, k, n):
    return MLMatMulDataBaseUtils.GetGpuDirectKeyParamInternalWarpMode(
        batch, m, k, n)

  @staticmethod
  def GetGpuDirectKeyParam(param):
    batch = MatMulParamUtils.get_batch_size(param)
    m = MatMulParamUtils.get_m_size(param)
    k = MatMulParamUtils.get_k_size(param)
    n = MatMulParamUtils.get_n_size(param)
    return MLMatMulDataBaseUtils.GetGpuDirectKeyParamInternal(
        batch, m, k, n)

class MatMulDataUtils(OpDataUtils):
  def extract(self, param, device, do_data_transform=False, lat_values=None):
    return OpDataUtils.extract(self, param, device, do_data_transform, lat_values)

class MLMatMulDataUtils(MatMulDataUtils):
  def __init__(self):
    MatMulDataUtils.__init__(self)
    dataset_info = MLMatMulDatasetInfo()
    self._model_names = dataset_info.model_names()
    self._col_names = dataset_info.col_names()

  def collect_granularity(self):
    return 'fine'

  def _extract_cpu_gemm(self, param, lat_values):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 3)

    m = MatMulParamUtils.get_m_size(param)
    k = MatMulParamUtils.get_k_size(param)
    n = MatMulParamUtils.get_n_size(param)

    key_param = MLConv2dDataBaseUtils.GetCpuGemmKeyParamInternal(m, k, n)
    key_param['nn_name'] = param['nn_name']

    ml_data = MLConv2dDataUtils.extract_cpu_gemm(key_param, lat_values)
    ml_data['model_names'] = ['t_matmul_cpu_gemm']

    return ml_data

  def _extract_gpu_direct(self, param, lat_values):
    if lat_values is not None:
      self._check_label_len(len(lat_values), 1)
      cost_warp = lat_values[0]
    else:
      cost_warp = 0

    key_param = MLMatMulDataBaseUtils.GetGpuDirectKeyParam(param)

    # Time of computing a warp.
    t_warp = cost_warp / key_param['n_warp']
    # Time of computing a block in warp.
    t_warp_blk = t_warp / key_param['kb']

    ml_data = {}
    ml_data['model_names'] = ['t_matmul_gpu_direct']
    ml_data['train_vars'] = [[param['nn_name'], cost_warp, t_warp, t_warp_blk]]
    ml_data['feature_vars'] = [[key_param['batch'],
                                key_param['m'], key_param['k'], key_param['n'],
                                key_param['mb'], key_param['kb'], key_param['nb'],
                                key_param['n_warp']]]

    return ml_data

  def extract(self, param, device, do_data_transform=False, lat_values=None):
    if device == Device.CPU:
      if do_data_transform:
        return None
      return self._extract_cpu_gemm(param, lat_values)
    elif device == Device.GPU:
      return self._extract_gpu_direct(param, lat_values)
    else:
      raise ValueError('Unsupported device: ' + device)
      return None
