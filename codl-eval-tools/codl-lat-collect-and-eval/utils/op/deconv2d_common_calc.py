
from enum import Enum

class Deconv2dCpuImplType(Enum):
  DIRECT = 1

def GetDeconv2dCpuImplType():
  return Deconv2dCpuImplType.DIRECT

def CalcDeconvFlops(input_shape, filter_shape, output_shape):
  return (input_shape[0] * input_shape[1] * input_shape[2]) \
      * (2 * filter_shape[1] * filter_shape[2] * filter_shape[3]) * filter_shape[0] \
      + (output_shape[1] * output_shape[2] * output_shape[3])

def GetDeconv2dCpuDirectOcPerItem(ks, s):
  if ks == 2 or ks == 3 or ks == 4:
    if s == 1:
      return 2
    elif s == 2:
      return 1
    else:
      raise ValueError('Unsupport kernel size {} and stride {}'.format(ks, s))
      return 0
  else:
    return 1
