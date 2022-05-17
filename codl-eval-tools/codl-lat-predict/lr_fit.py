
import os
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn import linear_model
from base.metric import *
from utils.data_loader import *
from utils.data_utils import *

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, required=False, default=None,
                      help='Data file')
  parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
  args = parser.parse_args()
  return args

def pause():
  input('Press ENTER to continue')

def sklearn_lr_fit(x,
                   y,
                   x_names=None,
                   y_name=None,
                   x_scale=False,
                   y_scale=False,
                   show_coef=False,
                   show_fig=False):
  if np.size(y) == 0:
    return 0

  if x_scale:
    x = scale(x)
  
  degree = [1]
  y_test = []
  y_test = np.array(y_test)

  LogUtil.print_log('x_shape {}, y_shape {}'.format(
      np.shape(x), np.shape(y)), LogLevel.DEBUG)

  if show_fig:
    x_array = np.linspace(1, np.size(y), np.size(y))
    plt.scatter(x_array, np.sort(y))

  for d in degree:
    # pipeline_conf = [('poly', PolynomialFeatures(degree=d)),
    #                  ('linear', LinearRegression(fit_intercept=True))]
    pipeline_conf = [('linear', LinearRegression(fit_intercept=True))]
    clf = Pipeline(pipeline_conf)
    clf.fit(x, y)
    y_test = clf.predict(x)
    #print(y_test)

    pm10acc = PM10ACC(y_test, y) * 100
    pm20acc = PMACC(y_test, y, 20) * 100

    #PrintTestTrue(x, y_test, y)
 
    LogUtil.print_log(('Results: degree %d, rmse %.2f, R2 %.2f, R22 %.2f, clf.score %.2f, pm10acc %.2f, pm20acc %.2f' %
        (d,
         RMSE(y_test, y),
         R2(y_test, y),
         R22(y_test, y),
         clf.score(x, y),
         pm10acc,
         pm20acc)))
    
    if show_coef:
      coefs = clf.named_steps['linear'].coef_
      inter = clf.named_steps['linear'].intercept_
      print('coefs {}, inter {}'.format(coefs, inter))

    # Sorted by y.
    _, y_test = (list(t) for t in zip(*sorted(zip(y, y_test))))
    y_test = np.array(y_test)
    
    if show_fig:
      #print('x {}, y_test {}'.format(np.shape(x_array), np.shape(y_test)))
      #plt.plot(x, y_test, linewidth=2)
      plt.scatter(x_array, y_test)

  if show_fig:
    #plt.grid()
    titles = []
    titles.append('gt')
    for d in degree:
      titles.append('lr (d=%d)' % d)
    plt.legend(titles, loc='upper left')
    plt.show()

  '''
  if show_fig:
    plt.clf()
    for i in range(np.shape(x)[1]):
      plt.scatter(x[:, i], y)
      if x_names is not None:
        plt.xlabel(x_names[i])
      if y_name is not None:
        plt.ylabel(y_name)
      plt.show()
  '''

  return pm10acc, clf, None, None

def grad_desc_lr_fit(x,
                     y,
                     x_names=None,
                     y_name=None,
                     x_scale=False,
                     y_scale=False,
                     show_coef=False,
                     show_fig=False):
  # TODO: LR training based on gradient descent.
  LogUtil.print_log(f"x: {x.shape}, y: {y.shape}", LogLevel.DEBUG)
  #print(x)

  if x_scale:
    #x = scale(x)
    x, means, stds = scale_data(x)
  else:
    means = None
    stds = None

  if y_scale:
    y, _, _ = scale_data(y)

  if show_fig:
    plt.scatter(x, y)

  from model.lr_model import GradDescentLRModel
  learning_rate = 1e-1
  num_epochs = 10000
  start_time = time.time()
  clf = GradDescentLRModel(learning_rate, num_epochs).fit(x, y)
  LogUtil.print_log('Train learning_rate {}, num_epochs {}'.format(
      learning_rate, num_epochs))
  LogUtil.print_log('Train time {:.2f}'.format(
      time.time() - start_time))

  coefs = clf.coefs()
  inter = clf.inter()
  if show_coef:
    print('coefs {}, inter {}'.format(coefs, inter))

  y_test = clf.predict(x)
  if y_scale:
    y_test = recover_data(y_test, means, stds)

  #PrintTestTrue(x, y_test, y)
  
  pm10acc = PM10ACC(y_test, y) * 100
  pm20acc = PMACC(y_test, y, 20) * 100

  #print('x {}, y {}'.format(np.shape(x), np.shape(y)))

  LogUtil.print_log('Results: rmse %.2f, R2 %.2f, R22 %.2f, clf.score %.2f, pm10acc %.2f, pm20acc %.2f' %
        (RMSE(y_test, y),
         R2(y_test, y),
         R22(y_test, y),
         clf.score(x, y),
         pm10acc,
         pm20acc))

  if show_fig:
    plt.scatter(x, y_test)
    titles = []
    titles.append('gt')
    titles.append('lr')
    plt.legend(titles, loc='upper left')
    plt.show()

  return pm10acc, clf, means, stds

def lr_fit(x,
           y,
           x_names=None,
           y_name=None,
           lr_type='sklearn',
           x_scale=False,
           y_scale=False,
           show_coef=False,
           show_fig=False):
  LogUtil.print_log('===== LR FIT INFO =====')
  if lr_type == 'sklearn':
    return sklearn_lr_fit(x, y, x_names, y_name, x_scale, y_scale, show_coef, show_fig)
  elif lr_type == 'grad_desc':
    return grad_desc_lr_fit(x, y, x_names, y_name, x_scale, y_scale, show_coef, show_fig)
  else:
    raise ValueError('Unsupported LR type ' + lr_type)

def lr_test(clf, x, y, x_scale=False, y_scale=False, means=None, stds=None):
  LogUtil.print_log('===== LR TEST INFO =====')
  LogUtil.print_log('y_shape {}'.format(np.shape(y)), LogLevel.DEBUG)
  if x_scale:
    x = scale(x)
  
  y_test = clf.predict(x)
  if y_scale:
    y_test = recover_data(y_test, means, stds)
  
  #PrintTestTrue(x, y_test, y)
  
  pm10acc = PM10ACC(y_test, y) * 100
  pm20acc = PMACC(y_test, y, 20) * 100
  LogUtil.print_log('Results: rmse %.2f, R2 %.2f, R22 %.2f, clf.score %.2f, pm10acc %.2f, pm20acc %.2f' %
      (RMSE(y_test, y),
       R2(y_test, y),
       R22(y_test, y),
       clf.score(x, y),
       pm10acc,
       pm20acc))
  return pm10acc

def lr_model_to_json(clf, lr_type, means=None, stds=None):
  if lr_type == 'sklearn':
    coefs = clf.named_steps['linear'].coef_
    inter = clf.named_steps['linear'].intercept_
  elif lr_type == 'grad_desc':
    coefs = clf.coefs()
    inter = clf.inter()
  LogUtil.print_log('coefs {}, inter {}'.format(coefs, inter), LogLevel.DEBUG)
  out_dict = {}
  out_dict['means'] = []
  out_dict['stds'] = []
  out_dict['coefs'] = []
  out_dict['inter'] = inter
  
  if means is not None:
    for v in means:
      out_dict['means'].append(v)
  else:
    for i in range(len(coefs) + 1):
      out_dict['means'].append(0)
  
  if stds is not None:
    for v in stds:
      out_dict['stds'].append(v)
  else:
    for i in range(len(coefs) + 1):
      out_dict['stds'].append(1)
  
  for v in coefs:
    out_dict['coefs'].append(v)
  
  return json.dumps(out_dict)

def save_model(dir, filename, text):
  if not os.path.exists(dir):
    os.makedirs(dir)
  filepath = os.path.join(dir, filename)
  with open(filepath, 'w') as f:
    f.write(text);
  #LogUtil.print_log('===== SAVE MODEL INFO ====')
  LogUtil.print_log('Saved model: ' + filepath)

def fit_test(data_file, dataset_name):
  if dataset_name == 'SIMPLE-TEST':
    x = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    y = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    lr_fit(np.array(x), np.array(y), lr_type="grad_desc")
    lr_fit(np.array(x), np.array(y), lr_type="sklearn")
  elif dataset_name == 'CPU-DIRECT':
    dl = CpuDirectDataLoader(data_file)
    #dl.add_filter('KH', 3)
    #dl.add_filter('SH', 1)
    x = dl.get_features(['IH', 'IW', 'IC', 'OC', 'KH', 'SH'])
    y = dl.get_target()

    lr_fit(x, y)
  elif dataset_name == 'CPU-DIRECT-PCA':
    dl = CpuDirectPcaDataLoader(data_file)
    x = dl.get_features()
    y = dl.get_target()

    lr_fit(x, y)
  elif dataset_name == 'CPU-GEMM':
    dl = CpuGemmDataLoader(data_file)
    x = dl.get_features(['M', 'K', 'N'])
    y = dl.get_target()
    
    lr_fit(x, y)
  elif dataset_name == 'CPU-GEMM-PCA':
    dl = CpuGemmPcaDataLoader(data_file)
    x = dl.get_features()
    y = dl.get_target()
    
    lr_fit(x, y)
  elif dataset_name == 'CONV2D-GPU-DIRECT':
    dl = Conv2dGpuDirectDataLoader(data_file)
    #dl.add_filter('KS', 3)
    #dl.add_filter('S', 1)
    x = dl.get_features(['OH', 'OWB', 'ICB', 'OCB', 'KS', 'S'])
    y = dl.get_target()
    
    lr_fit(x, y)
  elif dataset_name == 'CONV2D-GPU-DIRECT-PCA':
    dl = Conv2dGpuDirectPcaDataLoader(data_file)
    x = dl.get_features()
    y = dl.get_target()
    
    lr_fit(x, y)
  else:
    raise ValueError('Unsupported dataset name: ' + dataset_name)

if __name__ == '__main__':
  args = parse_args()
  data = args.data
  dataset = args.dataset
  
  fit_test(data, dataset)
