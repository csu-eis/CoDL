
from enum import Enum

class OpType(Enum):
  Conv2D = 0
  Pooling = 1
  FullyConnected = 2
  MatMul = 3

class ComputeEngine(Enum):
  CPU = 0
  GPU = 1

def StringToOpType(op_type_str):
  if op_type_str == 'Conv2D':
    return OpType.Conv2D
  elif op_type_str == 'Pooling':
    return OpType.Pooling
  elif op_type_str == 'FullyConnected':
    return OpType.FullyConnected
  elif op_type_str == 'MatMul':
    return OpType.MatMul
  else:
    raise ValueError('Unsupported OP type: ' + op_type_str)
    return None

def StringToComputeEngine(ce_str):
  if ce_str == 'CPU':
    return ComputeEngine.CPU
  elif ce_str == 'GPU':
    return ComputeEngine.GPU
  else:
    raise ValueError('Unsupported compute engine: ' + ce_str)
    return None
