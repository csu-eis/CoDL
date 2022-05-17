
from utils.common.log import *

kVerbose = False

class StringUtils:
  @staticmethod
  def str_to_float_list(text):
    text = text[1:len(text)-2]
    texts = text.split(',')
    val_list = []
    for text in texts:
      try:
        val = float(text)
      except ValueError:
        val = 0
        if kVerbose:
          TerminalLogger.log(LogTag.WARNING,
              'Fail to convert text {} to float'.format(text))
      finally:
        if str(val) == 'nan':
          val = 0
        #TerminalLogger.log(LogTag.INFO, str(val))
        val_list.append(val)
    return val_list

  @staticmethod
  def str_to_list(text, delimiter, dtype):
    if text.startswith('['):
      text = text[1:]
    if text.endswith(']'):
      text = text[:-1]
    out_list = []
    items = text.split(delimiter)
    for it in items:
      if dtype == 'int': v = int(it)
      elif dtype == 'float': v = float(it)
      else: raise ValueError('Unsupported data type: ' + dtype)
      out_list.append(v)
    return out_list
