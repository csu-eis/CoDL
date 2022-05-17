
from enum import Enum

class DLModel(Enum):
  YOLO_V2 = 0,
  VGG16 = 1,
  POSENET = 2,
  FAST_STYLE_TRANSFER = 3,
  RETINAFACE = 4

def StringToDLModel(text):
  if text == 'yolo_v2': return DLModel.YOLO_V2
  elif text == 'vgg16': return DLModel.VGG16
  elif text == 'posenet': return DLModel.POSENET
  elif text == 'fast_style_transfer': return DLModel.FAST_STYLE_TRANSFER
  elif text == 'retinaface': return DLModel.RETINAFACE
  else: raise ValueError('Unsupported DL model string: ' + text)

def ModelToModelName(model):
  if model == DLModel.YOLO_V2: return 'yolo_v2'
  elif model == DLModel.VGG16: return 'vgg16'
  elif model == DLModel.POSENET: return 'posenet'
  elif model == DLModel.FAST_STYLE_TRANSFER: return 'fast_style_transfer'
  elif model == DLModel.RETINAFACE: return 'retinaface'
  else: raise ValueError('Unsupported DL model')
