
from utils.op.conv_pool_common_calc import *
from utils.op.op_param_utils import *

class ConvPoolParamUtils(OpParamUtils):
  def calc_output_shape(self, param):
    OpParamUtils.calc_output_shape(self, param)
    param['output_shape'], padded_in_shape = ConvPoolCalcUtils.CalcOutputShape(
        param['input_shape'],
        param['filter_shape'],
        param['strides'],
        param['padding_type'],
        param['paddings'])
    param['input_shape'] = padded_in_shape
    return param

  def calc_partition_shape(self, param, pdim, pratio, device):
    OpParamUtils.calc_partition_shape(self, param, pdim, pratio, device)
    # Calculate partitioned shape.
    in_shape, k_shape, out_shape = ConvPoolCalcUtils.CalcPartitionShape(
        param['input_shape'],
        param['filter_shape'],
        param['output_shape'],
        param['strides'], pdim, pratio, device)
    # Copy a paramter to avoid the modification of original shape.
    out_param = param.copy()
    out_param['input_shape'] = in_shape
    out_param['filter_shape'] = k_shape
    out_param['output_shape'] = out_shape
    return out_param
