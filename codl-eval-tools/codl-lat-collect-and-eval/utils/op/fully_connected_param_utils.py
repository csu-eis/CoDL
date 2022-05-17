
from utils.common.math import *
from utils.common.basic_type import Device
from utils.op.op_common_calc import *
from utils.op.op_param_utils import *
from utils.op.fully_connected_common_calc import *

class FullyConnectedParamUtils(OpParamUtils):
  def extract_from_row(self, row):
    param = {}
    param['input_shape'] = [1, int(row['IH']),
                            int(row['IW']), int(row['IC'])]
    param['weight_shape'] = [int(row['OC']), int(row['IC']),
                             int(row['KH']), int(row['KW'])]
    return param

  def _is_param_equal(self, a, b):
    def is_list_equal(a, b, iname, ilen):
      if len(a[iname]) != len(b[iname]):
        return False
      for i in range(ilen):
        if a[iname][i] != b[iname][i]:
          return False
      return True

    iname_list = ['input_shape', 'weight_shape']
    ilen_list = [4, 4]
    for iname, ilen in zip(iname_list, ilen_list):
      if not is_list_equal(a, b, iname, ilen):
        return False

    return True

  def calc_output_shape(self, param):
    param['output_shape'] = [1, 1, 1, param['weight_shape'][0]]
    return param

  def calc_partition_shape(self, param, pdim, pratio, device):
    out_param = param.copy()
    out_shape = param['output_shape']
    if pdim == PartitionDimension.OUT_CHANNEL:
      if device == Device.GPU:
        out_pch = RoundUpMul(out_shape[3], pratio, 1)
      else:
        out_pch = out_shape[3] - RoundUpMul(out_shape[3], pratio, 1)
      out_pch = max(out_pch, 1)

      out_param['weight_shape'] = out_param['weight_shape'].copy()
      out_param['output_shape'] = out_param['output_shape'].copy()
      out_param['weight_shape'][0] = out_pch
      out_param['output_shape'][3] = out_pch
    else:
      raise ValueError('Unsupported partition dimension: ' + str(pdim))

    return out_param

  def calc_flops(self, param):
    return 0
