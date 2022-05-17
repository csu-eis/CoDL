
import argparse
import numpy as np
import matplotlib.pyplot as plt
from base.metric import *
from utils.data_loader import *
from utils.log_utils import *

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, required=True, help='Data file')
  parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name')
  args = parser.parse_args()
  return args

class MedianPredictModel(object):
  def __init__(self, median):
    self._median = median

  def predict(self, x):
    num_values = np.shape(np.array(x))[0]
    y_pred = []
    for i in range(num_values):
      y_pred.append(self._median)
    return np.array(y_pred)

def median_fit(y, show_fig=False):
  #print('Y shape: {}'.format(np.shape(y)))
  num_values = np.shape(y)[0]
  if num_values == 0:
    return 0

  LogUtil.print_log('===== MEDIAN FIT INFO =====') 

  median = np.median(y)
  #print('Median: {}'.format(med))

  clf = MedianPredictModel(median)
  y_pred = clf.predict(y)

  pm10acc = PM10ACC(y, y_pred)
  pm20acc = PMACC(y, y_pred, 20)
  LogUtil.print_log('Value# {:d}, pm10acc {:.2f}, pm20acc {:.2f}'.format(
      num_values, pm10acc * 100, pm20acc * 100))

  if show_fig:
    # Sorted by y.
    y, y_pred = (list(t) for t in zip(*sorted(zip(y, y_pred))))
    y = np.array(y)
    y_pred = np.array(y_pred)

    x_array = np.linspace(1, np.size(y), np.size(y))
    plt.scatter(x_array, y)
    plt.scatter(x_array, y_pred)
    plt.legend(['gt', 'median'], loc='upper left')
    plt.show()

  return pm10acc, clf

def median_test(clf, y):
  num_values = np.shape(y)[0]
  if num_values == 0:
    return 0
  
  LogUtil.print_log('===== MEDIAN TEST INFO =====')
  y_pred = clf.predict(y)
  #PrintTestTrue(y_pred, y)
  pm10acc = PM10ACC(y, y_pred)
  pm20acc = PMACC(y, y_pred, 20)
  LogUtil.print_log('Value# {:d}, pm10acc {:.2f}, pm20acc {:.2f}'.format(
      num_values, pm10acc * 100, pm20acc * 100))

  return pm10acc

def med_fit_test(data_file, dataset_name):
  if dataset_name == 'CPU-DIRECT':
    dl = CpuDirectDataLoader(data_file)
    #dl.add_filter('KH', 9)
    #dl.add_filter('SH', 1)
    y = dl.get_target()
    
    dl.debug_print()
    median_fit(y)
  elif dataset_name == 'CPU-GEMM':
    dl = CpuGemmDataLoader(data_file)
    y = dl.get_target()
    
    dl.debug_print()
    median_fit(y)
  elif dataset_name == 'GPU-DIRECT':
    dl = Conv2dGpuDirectDataLoader(data_file)
    #dl.add_filter('KS', 3)
    #dl.add_filter('S', 1)
    y = dl.get_target()
    
    dl.debug_print()
    median_fit(y)
  else:
    raise ErrorValue('Unsupported dataset name: ' + dataset_name)

if __name__ == '__main__':
  args = parse_args()
  data = args.data
  dataset = args.dataset

  med_fit_test(data, dataset)
