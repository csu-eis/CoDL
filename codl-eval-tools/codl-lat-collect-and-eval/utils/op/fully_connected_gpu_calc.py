
from utils.op.op_gpu_calc import *

class FullyConnectedOpenCLUtils(object):
  @staticmethod
  def GetChannelBlockSize():
    return 4
  
  @staticmethod
  def DefaultLocalWS(out_blocks, soc_name):
    wave_size = GetKernelWaveSize(soc_name)
    kwg_size = GetMaxWorkgroupSize(soc_name)

    if soc_name.startswith('Snapdragon'):
      gws = [4, wave_size // 4, out_blocks]
      inter_local_blks = kwg_size / (gws[0] * gws[1])
      lws = [gws[0], gws[1], inter_local_blks]
    else:
      gws = [4, 8, out_blocks]
      inter_local_blks = kwg_size / (gws[0] * gws[1])
      lws = [gws[0], gws[1], inter_local_blks]
      
    return gws, lws
