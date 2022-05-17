
from utils.op.conv2d_common_calc import *
from utils.op.conv_pool_param_utils import *

class Conv2dParamUtils(ConvPoolParamUtils):
  def example(self):
    ConvPoolParamUtils.example(self)
    conv2d_param = {}
    conv2d_param['input_shape'] = [1, 13, 13, 1024]
    conv2d_param['filter_shape'] = [1024, 1024, 3, 3]
    conv2d_param['strides'] = [1, 1]
    conv2d_param['padding_type'] = 1
    conv2d_param['paddings'] = [0, 0]
    conv2d_param['dilations'] = [1, 1]
    return conv2d_param

  def extract_from_row(self, row):
    ConvPoolParamUtils.extract_from_row(self, row)
    conv2d_param = {}
    conv2d_param['input_shape'] = [1, int(row['IH']),
                                   int(row['IW']), int(row['IC'])]
    conv2d_param['filter_shape'] = [int(row['OC']), int(row['IC']),
                                    int(row['KH']), int(row['KW'])]
    conv2d_param['strides'] = [int(row['SH']), int(row['SW'])]
    conv2d_param['padding_type'] = int(row['PT'])
    conv2d_param['paddings'] = [int(row['PH']), int(row['PW'])]
    conv2d_param['dilations'] = [int(row['DH']), int(row['DW'])]
    return conv2d_param

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

    return True

  def calc_flops(self, param):
    ConvPoolParamUtils.calc_flops(self, param)
    return CalcConvFlops(param['filter_shape'], param['output_shape'])
