
import os

from utils.common.log import *
from utils.soc.soc import Soc

global_soc_dict = {}

def RegistrySoc():
  global global_soc_dict
  cur_dir = os.path.join('utils', 'soc')
  for f in os.listdir(cur_dir):
    name, ext = os.path.splitext(f)
    if ext == '.yaml':
      TerminalLogger.log(LogTag.INFO, 'Load {}'.format(f))
      file_path = os.path.join(cur_dir, f)
      soc = Soc(name)
      soc.load_yaml(file_path)
      global_soc_dict[name] = soc

def GetRegistedSocByName(name):
  global global_soc_dict
  return global_soc_dict[name]
