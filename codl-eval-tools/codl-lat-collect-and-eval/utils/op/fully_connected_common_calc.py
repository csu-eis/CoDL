
from enum import Enum

class FullyConnectedCpuImplType(Enum):
  GEMV = 1

def GetFullyConnectedCpuImplType():
  return FullyConnectedCpuImplType.GEMV

def CalcFullyConnectedFlops(kernel_shape, out_shape):
  return (out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]) \
    * (2 * kernel_shape[2] * kernel_shape[3] * kernel_shape[1] + 1)

def CalcFullyConnectedParams(kernel_shape):
  return (kernel_shape[2] * kernel_shape[3] * kernel_shape[1] + 1) * kernel_shape[0]
