
from enum import Enum

class MatMulCpuImplType(Enum):
  GEMM = 0

def GetMatMulCpuImplType():
  return MatMulCpuImplType.GEMM
