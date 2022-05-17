
from enum import Enum

class PartitionDimension(Enum):
  NONE        = 0
  HEIGHT      = 1
  WIDTH       = 2
  IN_CHANNEL  = 3
  OUT_CHANNEL = 4
  M_OR_BATCH  = 4

def PartitionDimensionValueToText(value):
  if value == 0:
    return 'NONE'
  elif value == 1:
    return 'H'
  elif value == 2:
    return 'W'
  elif value == 3:
    return 'IC'
  elif value == 4:
    return 'OC'
  else:
    raise ValueError('Unsupported partition dimension value ' + str(value))
