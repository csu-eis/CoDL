
import time
from enum import Enum

class LogTag(Enum):
  NONE = 0
  DEBUG = 1
  VERBOSE = 2
  INFO = 3
  WARNING = 4
  ERROR = 5

kMinLogLevel = LogTag.VERBOSE

class FileLogger(object):
  def __init__(self, tag=None):
    self._filename = 'log/log_'
    if tag is not None:
      self._filename = self._filename + tag
    self._filename = self._filename + '_' + FileLogger.current_date_text() + '.txt'
    print('HINT: Create log file name ' + self._filename)

  @staticmethod
  def current_date_text():
    return time.strftime('%Y%m%d%H%M%S', time.localtime())

  def write(self, text):
    with open(self._filename, 'a') as f:
      f.write(text)

class TerminalLogger(object):
  @staticmethod
  def log(tag=LogTag.INFO, msg=None, end='\n'):
    if tag.value >= kMinLogLevel.value:
      print('[{:s}] {:s}'.format(tag.name, msg), end=end)

class TerminalUtils(object):
  def pause(hint_text=None, time_sec=-1):
    if time_sec > 0:
      text = '{:s} ({:.3f} sec)'.format(hint_text, time_sec)
      TerminalLogger.log(LogTag.INFO, text)
      time.sleep(time_sec)
    else:
      if hint_text is not None:
        TerminalLogger.log(LogTag.INFO, hint_text)
      if True:
        input('')

  def count_down(time_sec=0):
    if time_sec > 0:
      count_down = time_sec
      for i in range(time_sec):
        TerminalLogger.log(LogTag.INFO, '%d' % count_down, '\r')
        count_down = count_down - 1
        time.sleep(1)

class PyCheck(object):
  @staticmethod
  def check_not_none(var, msg):
    if var is None:
      TerminalLogger.log(LogTag.ERROR, msg)
