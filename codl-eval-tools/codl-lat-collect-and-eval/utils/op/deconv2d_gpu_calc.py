
from utils.op.op_gpu_calc import *

class Deconv2dOpenCLUtils(object):
  @staticmethod
  def GetChannelBlockSize():
    return 4

  @staticmethod
  def CalcWidthBlocks(width, stride_w):
    width_tile = 5
    n_strides = (width + stride_w - 1) // stride_w
    width_blocks = ((n_strides + width_tile - 1) // width_tile) * stride_w
    return width_blocks

  @staticmethod
  def DefaultLocalWS(gws, soc_name):
    return Default3DLocalWS(gws, soc_name)
