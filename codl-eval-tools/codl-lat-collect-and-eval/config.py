
from utils.common.log import *
from core.train_base import *

#kDefaultTargetSoc = TargetSoc.SDM855
kDefaultMobileSocName = 'sdm855'
kDefaultThreadCount = 4
kDefaultGpuMemoryType = 'image'
kDefaultTrainRatio = 0.7
kDefaultSampleType = SampleType.RANDOM
kDefaultTrainTag = 'all'
kDefaultTestTag = 'all'
kDefaultDatasetPath = 'lat_datasets'
kDefaultLRModelPath = 'models/lr_dir'
kDefaultRFModelPath = 'models/rf_dir'
kDefaultDateTag = None
kDefaultRFModelDepth = 20
kDefaultSeed = 66

kShowDefaultSetting = False

if kShowDefaultSetting:
  TerminalLogger.log(LogTag.INFO, 'Default sample ratio for training: {}'.format(kDefaultTrainRatio))
  TerminalLogger.log(LogTag.INFO, 'Default dataset path: {}'.format(kDefaultDatasetPath))
  TerminalLogger.log(LogTag.INFO, 'Default LR model path: {}'.format(kDefaultLRModelPath))
  TerminalLogger.log(LogTag.INFO, 'Default RF model path: {}'.format(kDefaultRFModelPath))
  #TerminalLogger.log(LogTag.INFO, 'Default dataset date info: {}'.format(kDefaultDateTag))
  TerminalLogger.log(LogTag.INFO, 'Default seed: {}'.format(kDefaultSeed))

'''
def SetGlobalTargetSoc(soc_name):
  global kDefaultTargetSoc
  if soc_name == 'sdm855':
    kDefaultTargetSoc = TargetSoc.SDM855
  elif soc_name == 'sdm865':
    kDefaultTargetSoc = TargetSoc.SDM865
  elif soc_name == 'kirin960':
    kDefaultTargetSoc = TargetSoc.KIRIN960
  else:
    raise ValueError('Unsupported soc name: ' + soc_name)

  TerminalLogger.log(LogTag.INFO,
      'Target soc: {}'.format(kDefaultTargetSoc.name))

def GetGlobalTargetSoc():
  global kDefaultTargetSoc
  return kDefaultTargetSoc
'''

def SetGlobalMobileSocName(soc_name):
  global kDefaultMobileSocName
  kDefaultMobileSocName = soc_name
  TerminalLogger.log(LogTag.INFO, 'Target mobile SoC: {}'.format(soc_name))

def GetGlobalMobileSocName():
  global kDefaultMobileSocName
  return kDefaultMobileSocName

def SetGlobalThreadCount(thread_count):
  global kDefaultThreadCount
  kDefaultThreadCount = thread_count
  TerminalLogger.log(LogTag.INFO,
      'Default thread count: {}'.format(kDefaultThreadCount))

def GetGlobalThreadCount():
  global kDefaultThreadCount
  return kDefaultThreadCount

def SetGlobalGpuMemoryType(mem_type):
  global kDefaultGpuMemoryType
  kDefaultGpuMemoryType = mem_type
  TerminalLogger.log(LogTag.INFO,
      'Default GPU memory type: {}'.format(kDefaultGpuMemoryType))

def GetGlobalGpuMemoryType():
  global kDefaultGpuMemoryType
  return kDefaultGpuMemoryType
