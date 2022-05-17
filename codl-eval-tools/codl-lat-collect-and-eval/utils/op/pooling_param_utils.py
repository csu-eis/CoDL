
from utils.op.pooling_common_calc import *
from utils.op.conv_pool_param_utils import *

class PoolingParamUtils(ConvPoolParamUtils):
  def example(self):
    ConvPoolParamUtils.example(self)
    param = {}
    param['input_shape'] = [1, 224, 224, 64]
    param['filter_shape'] = [64, 64, 3, 3]
    param['strides'] = [2, 2]
    param['padding_type'] = 1
    param['paddings'] = [0, 0]
    param['dilations'] = [1, 1]
    param['pooling_type'] = 2
    return param

  def extract_from_row(self, row):
    ConvPoolParamUtils.extract_from_row(self, row)
    param = {}
    param['input_shape'] = [1, int(row['IH']), int(row['IW']), int(row['IC'])]
    param['filter_shape'] = [int(row['IC']), int(row['IC']),
                             int(row['KH']), int(row['KW'])]
    param['strides'] = [int(row['SH']), int(row['SW'])]
    param['padding_type'] = int(row['PAT'])
    param['paddings'] = [int(row['PH']), int(row['PW'])]
    param['dilations'] = [int(row['DH']), int(row['DW'])]
    param['pooling_type'] = int(row['POT'])
    return param

  def _is_param_equal(self, a, b):
    ConvPoolParamUtils._is_param_equal(self, a, b)
    def is_list_equal(a, b, iname, ilen):
      if len(a[iname]) != len(b[iname]):
        return False
      for i in range(ilen):
        if a[iname][i] != b[iname][i]:
          return False
      return True

    iname_list = ['input_shape', 'filter_shape', 'strides',
                  'paddings', 'dilations']
    ilen_list = [4, 4, 2, 2, 2]
    for iname, ilen in zip(iname_list, ilen_list):
      if not is_list_equal(a, b, iname, ilen):
        return False

    if a['padding_type'] != b['padding_type']:
      return False

    if a['pooling_type'] != b['pooling_type']:
      return False

    return True

  def calc_flops(self, param):
    ConvPoolParamUtils.calc_flops(self, param)
    return 0
