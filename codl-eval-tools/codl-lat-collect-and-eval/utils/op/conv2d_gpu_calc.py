
import sys
from utils.common.log import *
from utils.common.math import *
from utils.op.op_gpu_calc import *
from utils.op.conv2d_common_calc import *

kDebug = False

class Conv2dOpenCLUtils(object):
  @staticmethod
  def GetChannelBlockSize():
    return 4

  @staticmethod
  def GetWidthBlockSize(kernel_size):
    if kernel_size == 1:
      return 4
    elif kernel_size == 3:
      return 5
    else:
      return 4

  @staticmethod
  def DefaultK1x1LocalWS(gws, owb_size, soc_name):
    lws = [0, 0, 0]
    kwg_size = GetMaxWorkgroupSize(soc_name)
    if kwg_size != 0:
      cache_size = GetGlobalMemCacheSize(soc_name)
      kernel_cache_size = (owb_size + 4 + owb_size) * 4 * 4
      compute_units = max(GetComputeUnits(soc_name) // 2, 1)
      base = max(cache_size // kBaseGPUMemCacheSize, 1)

      if kDebug:
        format_text = 'cache_size {}, compute_units {}, base {}, kernel_cache_size {}, kwg_size {}'
        print(format_text.format(
            cache_size, compute_units, base, kernel_cache_size, kwg_size))

      lws[1] = min(gws[1], kwg_size)
      
      lws_limit = 128
      if lws[1] >= base:
        lws[0] = min(gws[0], base)
      elif (1 < lws[1] and lws[1] < base) and gws[0] >= lws_limit:
        lws[0] = min(gws[0], base)
      else:
        lws[0] = gws[0] // 8
        if lws[0] < base:
          lws[0] = max(gws[0] // 4, base)
      lws[0] = min(lws[0], kwg_size // lws[1])
      
      lws_size = lws[0] * lws[1]
      lws[2] = min((cache_size // kernel_cache_size // lws_size // compute_units) * 8, gws[2])
      if lws[2] == 0:
        lws[2] = min(gws[2], base)
      lws[2] = max(min(lws[2], kwg_size // lws_size), 1)
    else:
      lws = [1, 1, 1]

    CheckLWS(lws)

    return lws

  @staticmethod
  def DefaultK3x3LocalWS(gws, owb_size, soc_name):
    lws = [0, 0, 0]
    kwg_size = GetMaxWorkgroupSize(soc_name)
    if kwg_size != 0:
      cache_size = GetGlobalMemCacheSize(soc_name)
      kernel_cache_size = (owb_size + 4 + owb_size) * 4 * 4
      compute_units = max(GetComputeUnits(soc_name) // 2, 1)
      base = max(min(cache_size // kBaseGPUMemCacheSize, 4), 1)

      lws[1] = min(gws[1], kwg_size)
      
      lws[0] = min(min(gws[0], base), kwg_size // lws[1])
      lws_size = lws[0] * lws[1]

      num_kernels = cache_size // kernel_cache_size
      num_lws = num_kernels // lws_size
      num_lws_per_compute_unit = num_lws // compute_units
      if kDebug:
        print('cache_size {}, kernel_cache_size {}, lws_size {}, compute_units {}'.format(
            cache_size, kernel_cache_size, lws_size, compute_units))
        print('num_kernels {}, num_lws {}, num_lws_per_cu {}'.format(
            num_kernels, num_lws, num_lws_per_compute_unit))
      
      lws_limit = RoundUp(num_lws_per_compute_unit, base)

      if kDebug:
        print('lws_limit {}, gws[2] {}, base {}, kwg_size {}, lws_size {}'.format(
            lws_limit, gws[2], base, kwg_size, lws_size))
      
      lws[2] = min(lws_limit, gws[2])
      if lws[2] == 0:
        lws[2] = min(gws[2], base)
      lws[2] = max(min(lws[2], kwg_size // lws_size), 1)
    else:
      lws = [1, 1, 1]

    CheckLWS(lws)

    return lws

  @staticmethod
  def DefaultGeneralLocalWS(gws, owb_size, ks, soc_name):
    lws = [0, 0, 0]
    kwg_size = GetMaxWorkgroupSize(soc_name)
    if kwg_size != 0:
      cache_size = GetGlobalMemCacheSize(soc_name)
      kernel_cache_size = (owb_size + 4 + owb_size) * 4 * 4
      compute_units = GetComputeUnits(soc_name)
      base = max(cache_size // kBaseGPUMemCacheSize, 1)

      if kDebug:
        print('cache_size {}, compute_units {}, base {}, kernel_cache_size {}, kwg_size {}'.format(
            cache_size, compute_units, base, kernel_cache_size, kwg_size))

      lws_limit = 20

      lws[1] = min(gws[1], kwg_size)
      
      lws[0] = gws[0] // 4
      if lws[0] == 0:
        lws[0] = gws[0]
      lws[0] = min(lws[0], kwg_size // lws[1])
      
      lws_size = lws[0] * lws[1]
      lws[2] = min((cache_size // kernel_cache_size // ks // lws_size // compute_units) * 8, gws[2])
      if lws[2] == 0:
        if gws[2] < lws_limit:
          lws[2] = gws[2]
        else:
          lws[2] = base
      lws[2] = max(min(lws[2], kwg_size // lws_size), 1)
    else:
      lws = [1, 1, 1]

    CheckLWS(lws)

    return lws

  @staticmethod
  def DefaultLocalWS(gws, owb_size, ks, soc_name):
    if ks == 1:
      return Conv2dOpenCLUtils.DefaultK1x1LocalWS(gws, owb_size, soc_name)
    elif ks == 3:
      return Conv2dOpenCLUtils.DefaultK3x3LocalWS(gws, owb_size, soc_name)
    else:
      return Conv2dOpenCLUtils.DefaultGeneralLocalWS(gws, owb_size, ks, soc_name)

  @staticmethod
  def CalcGlobalWS(oh, ow, oc, ks):
    # Block size.
    kChannelBlockSize = Conv2dOpenCLUtils.GetChannelBlockSize()
    kWidthBlockSize = Conv2dOpenCLUtils.GetWidthBlockSize(ks)

    # Workgroup size.
    ocb = RoundUpDiv(oc, kChannelBlockSize)
    owb = RoundUpDiv(ow, kWidthBlockSize)
    gws = [ocb, owb, oh]

    return gws
