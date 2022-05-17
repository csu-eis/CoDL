
from utils.common.math import *
from utils.common.basic_type import Device
from utils.op.op_common_calc import *
from utils.op.op_param_utils import *
from utils.op.conv_pool_common_calc import *
from utils.op.deconv2d_common_calc import *

class Deconv2dParamUtils(OpParamUtils):
  def extract_from_row(self, row):
    param = {}
    param['input_shape'] = [1, int(row['IH']), int(row['IW']), int(row['IC'])]
    param['filter_shape'] = [int(row['OC']), int(row['IC']),
                             int(row['KH']), int(row['KW'])]
    param['strides'] = [int(row['SH']), int(row['SW'])]
    param['padding_type'] = int(row['PT'])
    param['paddings'] = [int(row['PH']), int(row['PW'])]
    return param

  def _is_param_equal(self, a, b):
    def is_list_equal(a, b, iname, ilen):
      if len(a[iname]) != len(b[iname]):
        return False
      for i in range(ilen):
        if a[iname][i] != b[iname][i]:
          return False
      return True

    iname_list = ['input_shape', 'filter_shape', 'strides', 'paddings',]
    ilen_list = [4, 4, 2, 2]
    for iname, ilen in zip(iname_list, ilen_list):
      if not is_list_equal(a, b, iname, ilen):
        return False

    if a['padding_type'] != b['padding_type']:
      return False

    return True

  def calc_output_shape(self, param):
    in_h = param['input_shape'][1]
    in_w = param['input_shape'][2]
    out_ch = param['filter_shape'][0]
    k_h = param['filter_shape'][2]
    k_w = param['filter_shape'][3]
    s_h = param['strides'][0]
    s_w = param['strides'][1]
    padding_type = PaddingType(param['padding_type'])

    if padding_type == PaddingType.SAME:
      out_h = in_h * s_h
      out_w = in_w * s_w
    elif padding_type == PaddingType.VALID:
      out_h = ConvPoolCalcUtils.CalcInSize(in_h, k_h, s_h)
      out_w = ConvPoolCalcUtils.CalcInSize(out_w, k_w, s_w)
    else:
      raise ValueError('Unsupported padding type: ' + str(padding_type))

    param['output_shape'] = [1, out_h, out_w, out_ch]
    return param

  def calc_partition_shape(self, param, pdim, pratio, device):
    out_param = param.copy()
    out_ch = param['filter_shape'][0]
    out_shape = param['output_shape']
    if pdim == PartitionDimension.OUT_CHANNEL:
      if device == Device.GPU:
        out_pch = RoundUpMul(out_shape[3], pratio, 1)
      else:
        out_pch = out_shape[3] - RoundUpMul(out_shape[3], pratio, 1)
      out_pch = max(out_pch, 1)

      out_param['filter_shape'] = out_param['filter_shape'].copy()
      out_param['output_shape'] = out_param['output_shape'].copy()
      out_param['filter_shape'][0] = out_pch
      out_param['output_shape'][3] = out_pch
    else:
      raise ValueError('Unsupported partition dimension: ' + str(pdim))

    return out_param

  def calc_flops(self, param):
    return CalcDeconvFlops(param['input_shape'], param['filter_shape'], param['output_shape'])
