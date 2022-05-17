
from utils.op.op_gpu_calc import *

class MatMulOpenCLUtils(object):
  @staticmethod
  def GetMBlockSize():
    return 4

  @staticmethod
  def GetKBlockSize():
    return 4

  @staticmethod
  def GetNBlockSize():
    return 4
  
  @staticmethod
  def DefaultLocalWS(soc_name):
    kwg_size = GetMaxWorkgroupSize(soc_name)

    lws = [kwg_size // 64, 64]
      
    return lws
