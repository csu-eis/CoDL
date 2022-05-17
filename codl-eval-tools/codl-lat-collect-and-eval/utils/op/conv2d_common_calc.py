
import copy
from enum import Enum
from utils.common.math import *
from utils.op.op_common_calc import *
from utils.op.conv_pool_common_calc import *

class Conv2dCpuImplType(Enum):
  DIRECT        = 0
  GEMM          = 1
  WINOGRAD      = 2
  WINOGRAD_GEMM = 3

def CalcConvFlops(kernel_shape, out_shape):
  return (out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]) \
    * (2 * kernel_shape[2] * kernel_shape[3] * kernel_shape[1] + 1)

def CalcConvParams(kernel_shape):
  return (kernel_shape[2] * kernel_shape[3] * kernel_shape[1] + 1) * kernel_shape[0]

def GetConv2dCpuImplType(filter_shape, strides, is_wino_gemm):
  if filter_shape[2] == 1 and filter_shape[3] == 1 and \
     strides[0] == 1 and strides[1] == 1:
    return Conv2dCpuImplType.GEMM
  elif (filter_shape[2] == 3 and filter_shape[3] == 3) and \
       (filter_shape[0] >= 8 and filter_shape[1] >= 8) and \
       strides[0] == 1 and strides[1] == 1:
    return Conv2dCpuImplType.WINOGRAD_GEMM \
        if is_wino_gemm else Conv2dCpuImplType.WINOGRAD
    return 
  else:
    return Conv2dCpuImplType.DIRECT

def GetConv2dGpuImplType():
  return Conv2dCpuImplType.DIRECT

def GetConv2dCpuDirectPaddingBlockSize(kernel_size, stride):
  if kernel_size == 3 and stride == 1:
    return 2, 4
  else:
    return 1, 4

def GetConv2dCpuDirectOcPerItem(kernel_size):
  if kernel_size == 3:
    return 2
  elif kernel_size == 5:
    return 4
  elif kernel_size == 7:
    return 4
  elif kernel_size == 9:
    return 4
  else:
    raise ValueError('Unsupported kernel size {}'.format(kernel_size))
    return 0

def CalcConv2dWinogradOutTileSize(in_shape, out_shape, padding_type_idx):
  in_height, in_width = in_shape[1], in_shape[2]
  padding_type = PaddingType(padding_type_idx)
  if padding_type == PaddingType.SAME:
    in_height, in_width = out_shape[1], out_shape[2]

  out_tile_size = 2
  if in_height > 16 and in_width > 16:
    out_tile_size = 6
  return out_tile_size

def CalcConv2dWinogradShape(in_shape, out_shape, out_tile_size):
  pad_oh = RoundUp(out_shape[1], out_tile_size)
  pad_ow = RoundUp(out_shape[2], out_tile_size)
  pad_out_shape = [out_shape[0], pad_oh, pad_ow, out_shape[3]]

  pad_ih = ConvPoolCalcUtils.CalcInSize(pad_oh, 3, 1)
  pad_iw = ConvPoolCalcUtils.CalcInSize(pad_ow, 3, 1)
  pad_in_shape = [in_shape[0], pad_ih, pad_iw, in_shape[3]]

  return pad_in_shape, pad_out_shape
