
from enum import Enum

class LogLevel(Enum):
  DEBUG = 0
  VERBOSE = 1
  INFO = 2
  WARNING = 3
  ERROR = 4
  FATAL = 5

GLOBAL_LOG_LEVEL=LogLevel.INFO.value

class LogUtil(object):
  @staticmethod
  def print_log(msg, log_level=LogLevel.VERBOSE):
    if log_level.value >= GLOBAL_LOG_LEVEL:
      print('[{:s}] {:s}'.format(log_level.name, msg))
