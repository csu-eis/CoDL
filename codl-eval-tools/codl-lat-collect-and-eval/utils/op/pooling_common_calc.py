
from enum import Enum
from utils.op.conv_pool_common_calc import PoolingType

class PoolingCpuImplType(Enum):
  DIRECT_AVG = 0
  DIRECT_MAX = 1

def GetPoolingCpuImplType(pooling_type):
  if PoolingType(pooling_type) == PoolingType.AVG:
    return PoolingCpuImplType.DIRECT_AVG
  elif PoolingType(pooling_type) == PoolingType.MAX:
    return PoolingCpuImplType.DIRECT_MAX
  else:
    raise ValueError('Unknown pooling type: ' + pooling_type)
    return None
