
from utils.common.log import *
from utils.common.math import *
from utils.soc.soc import *
from utils.soc.soc_registry import GetRegistedSocByName

kDebug = False

#kShaderCores = 3
kBaseGPUMemCacheSize = 16384

def GetMaxWorkgroupSize(soc_name):
  #soc = Soc.Create(soc_name)
  soc = GetRegistedSocByName(soc_name)
  return soc.max_work_group_size()

def GetGlobalMemCacheSize(soc_name):
  #soc = Soc.Create(soc_name)
  soc = GetRegistedSocByName(soc_name)
  return soc.global_mem_cache_size()

def GetComputeUnits(soc_name):
  #soc = Soc.Create(soc_name)
  soc = GetRegistedSocByName(soc_name)
  return soc.compute_units()

def GetKernelWaveSize(soc_name):
  #soc = Soc.Create(soc_name)
  soc = GetRegistedSocByName(soc_name)
  return soc.kernel_wave_size()

def CalcWorkgroupLength(ws):
  num = 1
  for i in ws:
    num = num * i
  return num

def CalcWorkgroupNumber(gws, lws):
  num = 1
  for n, m in zip(gws, lws):
    num = num * RoundUpDiv(n, m)
  return num

def CalcWarpNumber(gws, lws, soc_name):
  kComputeUnits = GetComputeUnits(soc_name)
  kWarpSize = GetKernelWaveSize(soc_name)

  len_lws = CalcWorkgroupLength(lws)
  num_wg = RoundUpDiv(CalcWorkgroupLength(gws), len_lws)
  num_wg_per_unit = RoundUpDiv(num_wg, kComputeUnits)
  num_warps_per_wg = RoundUpDiv(len_lws, kWarpSize)
  num_warps_per_unit = num_warps_per_wg * num_wg_per_unit
  num_warps = num_warps_per_unit

  if kDebug:
    format_text = 'gws {}, lws {}, len_lws {}, num_wg {}'.format(
        gws, lws, len_lws, num_wg)
    TerminalLogger.log(LogTag.INFO, format_text)
    format_text = 'num_wg_per_unit {}, num_warps_per_wg {}, num_warps_per_unit {}'.format(
        num_wg_per_unit, num_warps_per_wg, num_warps_per_unit)
    TerminalLogger.log(LogTag.INFO, format_text)
    TerminalUtils.pause()

  if not num_warps > 0:
    raise ValueError('Illegal wrap number: {}'.format(num_warps))

  return num_warps

def CheckLWS(ws):
  if ws[0] == 0 or ws[1] == 0 or ws[2] == 0:
    raise ValueError('Illegal workgroup size: {}'.format(ws))

def Default3DLocalWS(gws, soc_name):
  lws = [0, 0, 0]
  kwg_size = GetMaxWorkgroupSize(soc_name)
  if kwg_size != 0:
    cache_size = GetGlobalMemCacheSize(soc_name)
    base = max(cache_size // kBaseGPUMemCacheSize, 1)
    lws[1] = min(gws[1], kwg_size)
    lws[2] = min(min(gws[2], base), kwg_size // lws[1])
    lws_size = lws[1] * lws[2]
    lws[0] = max(min(base, kwg_size // lws_size), 1)
  else:
    lws = [1, 1, 1]
  
  return lws
