
from enum import Enum
from utils.common.basic_type import Device
from utils.common.math import RoundUp, RoundUpMul
from utils.op.op_common_calc import PartitionDimension

class PaddingType(Enum):
  VALID = 0
  SAME  = 1
  FULL  = 2

class PoolingType(Enum):
  NONE = 0
  AVG  = 1
  MAX  = 2

class ConvPoolCalcUtils(object):
  @staticmethod
  def CalcInSize(out_size, k_size, stride):
    return (out_size - 1) * stride + k_size

  @staticmethod
  def CalcOutSize(in_size, k_size, stride, pad_blk_size=1):
    out_size = int((in_size - k_size) / stride) + 1
    if pad_blk_size > 1:
      #if out_size % pad_blk_size > 0:
      #  out_size = out_size + (pad_blk_size - out_size % pad_blk_size)
      out_size = RoundUp(out_size, pad_blk_size)
      in_size = ConvPoolCalcUtils.CalcInSize(out_size, k_size, stride)
    return in_size, out_size

  @staticmethod
  def CalcOutputShape(in_shape, k_shape, strides, padding_type_idx, paddings):
    in_height = in_shape[1]
    in_width = in_shape[2]
    k_size = k_shape[2]
    stride = strides[0]
    if paddings is None:
      paddings = [0 ,0]
    padding_type = PaddingType(padding_type_idx)
    if padding_type == PaddingType.SAME:
      out_height = in_height
      out_width = in_width
      in_height = ConvPoolCalcUtils.CalcInSize(out_height, k_size, stride)
      in_width = ConvPoolCalcUtils.CalcInSize(out_width, k_size, stride)
      paddings[0] = in_height - out_height
      paddings[1] = in_width - out_width
      #print('paddings {}'.format(paddings))
      paddings = [0 ,0]
    elif padding_type == PaddingType.VALID:
      _, out_height = ConvPoolCalcUtils.CalcOutSize(in_height + paddings[0], k_size, stride)
      _, out_width = ConvPoolCalcUtils.CalcOutSize(in_width + paddings[1], k_size, stride)
    else:
      raise ValueError('Unsupported padding type ' + padding_type_idx)

    out_shape = [in_shape[0], out_height, out_width, k_shape[0]]
    padded_in_shape = [in_shape[0], in_height + paddings[0],
                       in_width + paddings[1], in_shape[3]]
    #print('padded_in_shape {}'.format(padded_in_shape))
    return out_shape, padded_in_shape

  @staticmethod
  def CalcPartitionShape(in_shape, k_shape, out_shape, strides, pd, pr, dev):
    if pd == PartitionDimension.HEIGHT:
      if dev == Device.GPU:
        out_ph = RoundUpMul(out_shape[1], pr, 1)
      else:
        out_ph = out_shape[1] - RoundUpMul(out_shape[1], pr, 1)
      out_ph = max(out_ph, 1)
      #print('out_ph {}'.format(out_ph))
      in_ph = ConvPoolCalcUtils.CalcInSize(out_ph, k_shape[2], strides[0])
      new_in_shape = [in_shape[0], in_ph, in_shape[2], in_shape[3]]
      new_k_shape = [k_shape[0], k_shape[1], k_shape[2], k_shape[3]]
      new_out_shape = [out_shape[0], out_ph, out_shape[2], out_shape[3]]
    elif pd == PartitionDimension.WIDTH:
      if dev == Device.GPU:
        out_pw = RoundUpMul(out_shape[2], pr, 1)
      else:
        out_pw = out_shape[2] - RoundUpMul(out_shape[2], pr, 1)
      out_pw = max(out_pw, 1)
      #print('out_pw {}'.format(out_pw))
      in_pw = ConvPoolCalcUtils.CalcInSize(out_pw, k_shape[2], strides[0])
      new_in_shape = [in_shape[0], in_shape[1], in_pw, in_shape[3]]
      new_k_shape = [k_shape[0], k_shape[1], k_shape[2], k_shape[3]]
      new_out_shape = [out_shape[0], out_shape[1], out_pw, out_shape[3]]
    elif pd == PartitionDimension.IN_CHANNEL:
      if dev == Device.GPU:
        in_pch = RoundUpMul(in_shape[3], pr, 1)
      else:
        in_pch = in_shape[3] - RoundUpMul(in_shape[3], pr, 1)
      in_pch = max(in_pch, 1)
      new_in_shape = [in_shape[0], in_shape[1], in_shape[2], in_pch]
      new_k_shape = [k_shape[0], in_pch, k_shape[2], k_shape[3]]
      new_out_shape = [out_shape[0], out_shape[1], out_shape[2], out_shape[3]]
    elif pd == PartitionDimension.OUT_CHANNEL:
      if dev == Device.GPU:
        out_pch = RoundUpMul(out_shape[3], pr, 1)
      else:
        out_pch = out_shape[3] - RoundUpMul(out_shape[3], pr, 1)
      out_pch = max(out_pch, 1)
      new_in_shape = [in_shape[0], in_shape[1], in_shape[2], in_shape[3]]
      new_k_shape = [out_pch, k_shape[1], k_shape[2], k_shape[3]]
      new_out_shape = [out_shape[0], out_shape[1], out_shape[2], out_pch]
    else:
      print('Unsupported partition dimension {}'.format(pd))

    return new_in_shape, new_k_shape, new_out_shape
