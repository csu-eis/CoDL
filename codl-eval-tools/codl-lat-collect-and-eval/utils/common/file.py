
import os
from utils.common.log import *

class FileUtils(object):
  @staticmethod
  def check_exist(path):
    if not os.path.exists(path):
      raise ValueError('File {} does not exist'.format(path))

  @staticmethod
  def create_dir(dirpath):
    if not os.path.exists(dirpath):
      TerminalLogger.log(LogTag.INFO, 'Create directory ' + dirpath)
      os.makedirs(dirpath)

  @staticmethod
  def count_line(filename):
    count = 0
    for count, _ in enumerate(open(filename, 'r')):
      pass
    count = count + 1
    return count

  @staticmethod
  def write_text(filepath, text):
    overwrite = True
    ret = True
    if os.path.exists(filepath):
      info = input('File {} exists, overwrite it? (y/N) '.format(filepath))
      if info != 'y' and info != 'Y':
        overwrite = False
        ret = False
    
    if overwrite:
      with open(filepath, 'w') as f:
        f.write(text)
    
    return ret
