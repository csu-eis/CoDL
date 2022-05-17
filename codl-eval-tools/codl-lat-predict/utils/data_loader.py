
import numpy as np
import pandas as pd
from utils.log_utils import *
from utils.math_utils import RoundUpDiv

kEnableDebug = False

class DataLoader(object):
  def __init__(self, filename):
    self._filename = filename
    self._df = pd.read_csv(filename)
    self._train_df = None
    self._test_df = None
    self._name = None
    self._filter_list = []
    if kEnableDebug:
      print(self._df)

  def _add_extra_cols(self):
    return None

  def get_name(self):
    return self._name

  def have_col_name(self, name):
    return name in self._df.columns

  def get_data_count(self, set_name=None):
    if set_name is None:
      return len(self._df)
    if set_name == 'train':
      return len(self._train_df)
    elif set_name == 'test':
      return len(self._test_df)

  def split_data(self, frac=0.7, random_state=66):
    if frac < 1.0:
      self._train_df = self._df.sample(frac=frac, random_state=random_state, axis=0)
      self._test_df = self._df[~self._df.index.isin(self._train_df.index)]
    else:
      self._train_df = self._df
      self._test_df = self._df
  
  def get_features(self, feat_names=None, set_name=None):
    if feat_names is None:
      feat_names = self._all_feat_names
    LogUtil.print_log('Features: {}'.format(feat_names), LogLevel.DEBUG)
    if set_name is None:
      return self._df[feat_names].to_numpy()
    if set_name == 'train':
      return self._train_df[feat_names].to_numpy()
    elif set_name == 'test':
      return self._test_df[feat_names].to_numpy()
    
  def get_target(self, target_name=None, set_name=None):
    if target_name is None:
      target_name = self._target_name
    LogUtil.print_log('Target: {}'.format(target_name),  LogLevel.DEBUG)
    if set_name is None:
      return self._df[target_name].to_numpy()
    if set_name == 'train':
      return self._train_df[target_name].to_numpy()
    elif set_name == 'test':
      return self._test_df[target_name].to_numpy()
    
  def get_all(self):
    name_list = []
    for name in self._all_feat_names:
      name_list.append(name)
    #name_list.append(self._target_name)
    return self._df[name_list].to_numpy()

  def add_col_by_add(self, target_name, feat_names):
    self._df[target_name] = self._df[feat_names[0]]
    for i in range(1, len(feat_names)):
      self._df[target_name] = self._df[target_name] + self._df[feat_names[i]]
    #print(self._df)

  def add_col_by_mul(self, target_name, feat_names):
    self._df[target_name] = self._df[feat_names[0]]
    for i in range(1, len(feat_names)):
      self._df[target_name] = self._df[target_name] * self._df[feat_names[i]]
    #print(self._df)

  def add_col_by_roundupdiv(self, target_name, feat_names):
    self._df[target_name] = RoundUpDiv(self._df[feat_names[0]], self._df[feat_names[1]])
    #print(self._df)

  def reset_filters(self):
    self._df = pd.read_csv(self._filename)
    self._filter_list = []
    self._add_extra_cols()
    
  def add_filter(self, feat_name, condition, value):
    if condition == '==':
      self._df = self._df[self._df[feat_name] == value]
    elif condition == '!=':
      self._df = self._df[self._df[feat_name] != value]
    elif condition == '<':
      self._df = self._df[self._df[feat_name] < value]
    elif condition == '<=':
      self._df = self._df[self._df[feat_name] <= value]
    elif condition == '>':
      self._df = self._df[self._df[feat_name] > value]
    elif condition == '>=':
      self._df = self._df[self._df[feat_name] >= value]
    else:
      raise ValueError('Unsupported condition: {}'.format(condition))
    
    info_dict = {}
    info_dict[feat_name] = '{} {}'.format(condition, value)
    self._filter_list.append(info_dict)

  def add_filters(self, filters):
    for f in filters:
      self.add_filter(f[0], f[1], f[2])

  def get_filter_list(self):
    return self._filter_list

  def debug_print(self):
    print('name {}, filter {}'.format(self._name, self._filter_list))

  def print_content(self):
    print(self._df)

class DataSharingDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'DATA-SHARING'
    self._all_feat_names = [['FLOPS_IN'], ['FLOPS_OUT']]
    self._target_name = ['T_DS_IN', 'T_DS_OUT']
    self._add_extra_cols()

  def _add_extra_cols(self):
    DataLoader._add_extra_cols(self)
    self.add_col_by_mul('FLOPS_IN', ['IH', 'IW', 'IC'])
    self.add_col_by_mul('FLOPS_OUT', ['OH', 'OW', 'OC'])
    #self.add_col_by_add('T_DS_IN', ['T_DT_IN', 'T_MAP_IN'])
    #self.add_col_by_add('T_DS_OUT', ['T_DT_OUT', 'T_MAP_OUT'])

class CpuDirectDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-DIRECT'
    self._all_feat_names = ['IH', 'IW', 'IC', 'OC', 'KH', 'SH']
    self._target_name = 'T_FLOPS'
    
class CpuDirectPcaDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-DIRECT-PCA'
    self._all_feat_names = ['F0', 'F1', 'F2']
    self._target_name = 'T_FLOPS'
    
class CpuGemmPackRbDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-GEMM-PACK-RB'
    self._all_feat_names = ['K', 'N', 'KB', 'NB']
    self._target_name = 'T_PACK_RB'

class CpuGemvDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-GEMV'
    self._all_feat_names = ['M', 'K', 'MB', 'KB']
    self._target_name = 'T_COMP'

class CpuGemmDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-GEMM'
    self._all_feat_names = ['M', 'K', 'N', 'MB', 'KB', 'NB']
    self._target_name = 'T_COMP'

class CpuGemmTotalDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-GEMM-TOTAL'
    self._all_feat_names = ['M', 'K', 'N', 'MB', 'KB', 'NB']
    self._target_name = 'ALL'
    self._add_extra_cols()

  def _add_extra_cols(self):
    DataLoader._add_extra_cols(self)
    self.add_col_by_roundupdiv('MAX_KB', ('K', 'KB'))
    self.add_col_by_mul('PACK_RB_BLKS', ('MAX_KB', 'MAX_NB_THR'))
    self.add_col_by_mul('COMP_BLKS', ('MAX_MB_THR', 'MAX_KB', 'MAX_NB_THR'))
    self.add_col_by_add('T_ALL', ['T_PACK_RB', 'T_COMP'])
    self.add_col_by_add('ALL', ['PACK_RB', 'COMP'])

class CpuGemmPcaDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-GEMM-PCA'
    self._all_feat_names = ['F0', 'F1', 'F2']
    self._target_name = 'T_COMP'
    
class CpuWinogradTrInDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-WINOGRAD-TR-IN'
    self._all_feat_names = ['PIH', 'PIW', 'OT']
    self._target_name = 'T_TR_IN'

class CpuWinogradTrOutDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-WINOGRAD-TR-OUT'
    self._all_feat_names = ['POH', 'POW', 'OC', 'OT']
    self._target_name = 'T_TR_OUT'

class CpuWinogradGemmDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-WINOGRAD-GEMM'
    self._all_feat_names = ['M', 'K', 'N', 'OT']
    self._target_name = 'T_COMP'

class CpuWinogradTotalDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CPU-WINOGRAD-TOTAL'
    self._all_feat_names = ['M', 'K', 'N', 'PIH', 'PIW', 'POH', 'POW', 'OT']
    self._target_name = 'ALL'
    self._add_extra_cols()

  def _add_extra_cols(self):
    DataLoader._add_extra_cols(self)
    self.add_col_by_roundupdiv('MAX_KB', ('K', 'KB'))
    self.add_col_by_mul('TR_IN_BLKS', ('MAX_IC_THR', 'TOTAL_OT_COUNT'))
    self.add_col_by_mul('TR_OUT_BLKS', ('MAX_OC_THR', 'TOTAL_OT_COUNT'))
    self.add_col_by_mul('COMP_BLKS', ('MAX_MB_THR', 'MAX_KB', 'MAX_NB_THR'))
    self.add_col_by_add('T_ALL', ['T_TR_IN', 'T_TR_OUT', 'T_COMP'])
    self.add_col_by_add('ALL', ['TR_IN', 'TR_OUT', 'COMP'])

class Conv2dGpuDirectDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CONV2D-GPU-DIRECT'
    self._all_feat_names = ['OH', 'OWB', 'ICB', 'OCB', 'KS', 'S']
    self._target_name = 'T_WARP_ICB'
  
  '''
  def get_target(self):
    icb = self.get_features(['ICB'])
    icb = np.reshape(icb, (np.shape(icb)[0]))
    target = DataLoader.get_target(self)
    print('icb shape {}'.format(np.shape(icb)))
    print('target shape {}'.format(np.shape(target)))
    raise ValueError()
    return target / icb
  '''

class Conv2dGpuDirectPcaDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CONV2D-GPU-DIRECT-PCA'
    self._all_feat_names = ['F0', 'F1', 'F2']
    self._target_name = 'T_WARP'
    
class Conv2dDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'CONV2D'
    self._all_feat_names = ['IH', 'IW', 'IC', 'OC', 'KH', 'SH', 'FLOPS', 'PARAMS']
    self._target_name = 'COST'
    self._add_extra_cols()

  def _add_extra_cols(self):
    DataLoader._add_extra_cols(self)
    #self._add_neurosurgeon_features()

  def _add_neurosurgeon_features(self):
    self.add_col_by_mul('F0', ('IH', 'IW', 'IC'))
    self._df['F1'] = (self._df['KH'] / self._df['SH']) * \
        (self._df['KH'] / self._df['SH']) * self._df['OC']

class PoolingDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'POOLING'
    self._all_feat_names = ['IH', 'IW', 'C', 'KH', 'SH', 'PARAMS']
    self._target_name = 'COST'
    self._add_extra_cols()

  def _add_extra_cols(self):
    DataLoader._add_extra_cols(self)
    if self.have_col_name('IH'):
      self.add_col_by_mul('PARAMS', ('IH', 'IW', 'IC'))

class FullyConnectedGpuDirectDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'FC-GPU-DIRECT'
    self._all_feat_names = ['IH', 'IW', 'ICB', 'OCB']
    self._target_name = 'T_WARP_BLK'

class FullyConnectedDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'FC'
    self._all_feat_names = ['IH', 'IW', 'IC', 'OC', 'KH', 'FLOPS', 'PARAMS']
    self._target_name = 'COST'

class Deconv2dCpuDirectDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'DECONV2D-CPU-DIRECT'
    self._all_feat_names = ['IH', 'IW', 'IC', 'OC']
    self._target_name = 'T_COMP'

class Deconv2dGpuDirectDataLoader(DataLoader):
  def __init__(self, filename):
    DataLoader.__init__(self, filename)
    self._name = 'DECONV2D-GPU-DIRECT'
    self._all_feat_names = ['IH', 'IW', 'ICB', 'OWB', 'OCB']
    self._target_name = 'T_WARP_ICB'

def test():
  dl = CpuDirectDataLoader('data/atomic_latency_cpu_direct_sdm855.csv')
  dl.add_filter('KH', 3)
  dl.add_filter('SH', 1)
  print(dl.get_features())
  print(dl.get_target())

if __name__ == '__main__':
  test()
