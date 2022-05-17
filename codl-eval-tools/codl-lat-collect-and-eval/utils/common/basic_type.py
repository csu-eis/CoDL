
from enum import Enum

class MemoryObject(Enum):
  CPU_BUFFER = 0
  GPU_BUFFER = 1
  GPU_IMAGE = 2

class Device(Enum):
  CPU = 0
  GPU = 1
  CPU_GPU = 2

def StringToMemoryObject(obj):
  if obj == 'cpu_buffer':
    return MemoryObject.CPU_BUFFER
  elif obj == 'gpu_buffer':
    return MemoryObject.GPU_BUFFER
  elif obj == 'gpu_image':
    return MemoryObject.GPU_IMAGE
  else:
    raise ValueError('Unsupported memory object: ' + obj)
    return None

def StringToDevice(device):
  if device == 'CPU': return Device.CPU
  elif device == 'GPU': return Device.GPU
  elif device == 'CPU+GPU' or device == 'CPU_GPU': return Device.CPU_GPU
  else:
    raise ValueError('Unsupported device: ' + device)
    return None
