
from enum import Enum

class OpType(Enum):
  Conv2D = 0,
  Pooling = 1,
  FullyConnected = 2,
  Deconv2D = 3,
  MatMul = 4

def StringToOpType(op_type):
  if op_type == 'Conv2D': return OpType.Conv2D
  elif op_type == 'Pooling': return OpType.Pooling
  elif op_type == 'FullyConnected': return OpType.FullyConnected
  elif op_type == 'Deconv2D': return OpType.Deconv2D
  elif op_type == 'MatMul': return OpType.MatMul
  else:
    raise ValueError('Unsupported op type: ' + op_type)
    return None
