
import numpy as np

def RMSE(y_test, y_true):
  return np.sqrt(np.mean((y_test - y_true) ** 2))

def R2(y_test, y_true):
  return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()

def R22(y_test, y_true):
  y_mean = np.array(y_true)
  y_mean[:] = y_mean.mean()
  #print('y_test, {}, y_true {}, y_mean {}'.format(
  #    np.shape(y_test), np.shape(y_true), np.shape(y_mean)))
  return 1 - RMSE(y_test, y_true) / RMSE(y_mean, y_true)

def PMACC(y_test, y_true, k):
  if np.size(y_test) == 0 or np.size(y_true) == 0:
    return -1

  diff_perct = abs(y_test - y_true) * 100 / y_true
  diff_perct = np.array(diff_perct)
  #print(diff_perct)
  diff_true = diff_perct[np.where(diff_perct <= k)]
  return len(diff_true) / len(diff_perct)

def PM10ACC(y_test, y_true):
  return PMACC(y_test, y_true, 10)

def IsPMACC(test, true, k):
  diff_perct = abs(test - true) * 100 / true
  return diff_perct <= k

def PrintTestTrue(x, y_test, y_true):
  for test, true in zip(y_test, y_true):
    print('{}'.format((x, test, true, IsPMACC(test, true, 10))))
  print('')
