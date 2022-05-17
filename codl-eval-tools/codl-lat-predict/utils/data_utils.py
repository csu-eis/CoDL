
import numpy as np
from utils.log_utils import *

def scale_data(x):
  x = x.astype(float)
  
  means = np.mean(x, axis=0)
  stds = np.std(x, axis=0)
  mean_size = np.size(means)

  if mean_size == 1:
    if stds == 0:
      stds = 1
  else:
    for i in range(np.size(stds)):
      if stds[i] == 0:
        stds[i] = 1
  LogUtil.print_log('means_shape {}, stds_shape {}'.format(
      np.shape(means), np.shape(stds)), LogLevel.DEBUG)
  LogUtil.print_log('means {}, stds {}'.format(
      means, stds), LogLevel.DEBUG)
  if mean_size == 1:
    for i in range(np.shape(x)[0]):
      x[i] = (x[i] - means) / stds
  else:
    for i in range(np.shape(x)[0]):
      x[i,:] = (x[i,:] - means) / stds
  return x, means, stds

def recover_data(x, means, stds):
  mean_size = np.size(means)
  if mean_size == 1:
    x = x * stds + means
  else:
    for i in range(np.shape(x)[0]):
      x[i,:] = x[i,:] * stds + means
  return x
