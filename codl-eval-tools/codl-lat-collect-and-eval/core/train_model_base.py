
import os
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from rich.console import Console
from rich.table import Column, Table
from utils.common.log import *
from utils.common.plot import *
from utils.dataset.dataset_utils import *
from core.train_base import *
from config import *

kLocalConfig = {'is_model_trainer_debug': False,
                'is_op_trainer_debug': True,
                'do_op_trainer_draw': False}

class EvalMetricUtils(object):
  @staticmethod
  def MeanSquaredError(ytrue, ypred):
    return np.mean(np.square(ytrue - ypred))

  @staticmethod
  def RootMeanSquaredError(ytrue, ypred):
    return np.sqrt(EvalMetricUtils.MeanSquaredError(ytrue, ypred))

  @staticmethod
  def RemoveZeroElement(ytrue, ypred):
    nz_indices = np.where(ytrue > 0)
    ytrue = ytrue[nz_indices]
    ypred = ypred[nz_indices]
    return ytrue, ypred

  @staticmethod
  def RootMeanSquarePercentageError(ytrue, ypred):
    ytrue, ypred = EvalMetricUtils.RemoveZeroElement(ytrue, ypred)
    return np.sqrt(np.mean(np.square((ytrue - ypred) / ytrue))) * 100

  @staticmethod
  def PlusMinusAccuracy(ytrue, ypred, err_bar):
    ytrue, ypred = EvalMetricUtils.RemoveZeroElement(ytrue, ypred)
    yerr = np.abs(ytrue - ypred) / ytrue * 100
    #print('yerr {}'.format(yerr))
    #print('sum (e <= eb) {} sum (t) {}'.format(np.sum(yerr <= err_bar), np.size(ytrue)))
    return np.sum(yerr <= err_bar) / np.size(ytrue) * 100

def test_eval_metrics():
  ytrue = np.array([10, 20, 30])
  ypred = np.array([11, 22, 33])
  rmspe = EvalMetricUtils.RootMeanSquarePercentageError(ytrue, ypred)
  pm5acc = EvalMetricUtils.PlusMinusAccuracy(ytrue, ypred, 5)
  pm10acc = EvalMetricUtils.PlusMinusAccuracy(ytrue, ypred, 10)
  print('rmspe {:.3f} pm5acc {:.3f} pm10acc {:.3f}'.format(rmspe, pm5acc, pm10acc))

def calc_train_test_indices(sample_type, train_df, test_df, ratio):
  if sample_type == SampleType.DEFAULT:
    train_indices = range(int(len(train_df) * ratio))
    test_indices = range(len(train_df) - train_indices)
  elif sample_type == SampleType.RANDOM:
    train_indices, test_indices = \
        DatasetUtils.RandomSplitIndex(len(train_df), ratio, kDefaultSeed)
  elif sample_type == SampleType.ALL:
    train_indices = range(len(train_df))
    test_indices = range(len(test_df))
  else:
    raise ValueError('Unsupported sample type ' + self._sample_type)

  return train_indices, test_indices

def print_eval_result(model_name, eval_result):
  '''
  TerminalLogger.log(LogTag.INFO, 'Eval result: mse {:.3f} rmse {:.3f} ' \
      'rmspe {:.3f} pm5acc {:.3f} pm10acc {:.3f} ' \
      'pm20acc {:.3f} pm30acc {:.3f} pm40acc {:.3f}'.format(
          eval_result['mse'], eval_result['rmse'], eval_result['rmspe'],
          eval_result['pm5acc'], eval_result['pm10acc'], eval_result['pm20acc'],
          eval_result['pm30acc'], eval_result['pm40acc']))
  '''
  table = Table(show_header=True, header_style='bold magenta')
  table.add_column('MODEL')
  table.add_column('MSE')
  table.add_column('RMSE')
  table.add_column('RMSPE')
  table.add_column('PM10ACC')
  table.add_row(model_name,
                '%.3f' % eval_result['mse'],
                '%.3f' % eval_result['rmse'],
                '%.3f' % eval_result['rmspe'],
                '%.3f' % eval_result['pm10acc'])
  console = Console()
  console.print(table)

def get_global_dataset_path():
  return kDefaultDatasetPath

def get_global_rf_model_path():
  return kDefaultRFModelPath

def get_global_rf_model_depth():
  return kDefaultRFModelDepth

def set_global_dataset_path(dataset_path):
  global kDefaultDatasetPath
  kDefaultDatasetPath = dataset_path
  TerminalLogger.log(LogTag.INFO, 'Set data path: %s' % kDefaultDatasetPath)

def set_global_rf_model_path(model_path):
  global kDefaultRFModelPath
  kDefaultRFModelPath = model_path
  TerminalLogger.log(LogTag.INFO, 'Set RF model path: %s' % kDefaultRFModelPath)

def set_global_rf_model_depth(depth):
  global kDefaultRFModelDepth
  kDefaultRFModelDepth = depth
  TerminalLogger.log(LogTag.INFO, 'Set RF model depth: %d' % kDefaultRFModelDepth)

def build_data_filepath(path, model_name, split_tag=None, date_tag=None):
  if path[-1] == '/':
    path = path[:-2]
  filename = '{}/{}'.format(path, model_name)
  if split_tag is not None:
    filename = filename + '_' + split_tag
  if date_tag is not None:
    filename = filename + '_' + date_tag
  filename = filename + '.csv'
  return filename

def find_data_filepath(path, model_name, split_tag=None, date_tag=None):
  target_filepath = build_data_filepath(path, model_name, split_tag, date_tag)
  TerminalLogger.log(LogTag.DEBUG, 'Try to find data file: {}'.format(target_filepath))
  for filename in os.listdir(path):
    if filename == target_filepath.split('/')[-1]:
      TerminalLogger.log(LogTag.INFO, 'Find data: {}'.format(target_filepath))
      return target_filepath
  
  target_filepath = build_data_filepath(path, model_name, split_tag)
  TerminalLogger.log(LogTag.DEBUG, 'Try to find data file: {}'.format(target_filepath))
  for filename in os.listdir(path):
    if filename == target_filepath.split('/')[-1]:
      TerminalLogger.log(LogTag.INFO, 'Find data: {}'.format(target_filepath))
      return target_filepath
  
  target_filepath = build_data_filepath(path, model_name)
  TerminalLogger.log(LogTag.DEBUG, 'Try to find data file: {}'.format(target_filepath))
  for filename in os.listdir(path):
    if filename == target_filepath.split('/')[-1]:
      TerminalLogger.log(LogTag.INFO, 'Find data: {}'.format(target_filepath))
      return target_filepath
  
  TerminalLogger.log(LogTag.DEBUG, 'Try to find data file: {}'.format(model_name))
  for filename in os.listdir(path):
    TerminalLogger.log(LogTag.DEBUG, 'Try a file: {}'.format(filename))
    if filename.startswith(model_name):
      target_filepath = os.path.join(path, filename)
      TerminalLogger.log(LogTag.INFO, 'Find data: {}'.format(target_filepath))
      return target_filepath

  raise ValueError('Can not find a data file for model ' + model_name)
  return None

class ModelTrainer(object):
  def __init__(self,
               model_name,
               train_data_filename,
               test_data_filename,
               label_name,
               feature_list,
               model_config=None,
               save_model=True,
               load_model=True):
    self._model_name = model_name
    self._train_df = pd.read_csv(train_data_filename).fillna(-1)
    self._test_df = pd.read_csv(test_data_filename).fillna(-1)
    self._label_name = label_name
    self._feature_list = feature_list
    self._train_ratio = kDefaultTrainRatio
    self._sample_type = kDefaultSampleType
    self._train_indices, self._test_indices = \
        calc_train_test_indices(self._sample_type,
                                self._train_df,
                                self._test_df,
                                self._train_ratio)

    self.update_model_config(model_config)

    self._total_start_time = time.time()
    self._is_debug = kLocalConfig['is_model_trainer_debug']

    #if self._is_debug:
    #  TerminalLogger.log(LogTag.INFO, 'sample type {} ratio {}'.format(
    #      self._sample_type, self._train_ratio))

  def _build_ret_dict(self, prof_data, pred_data):
    if prof_data is None or pred_data is None:
      mse = 0
      rmse = 0
      rmspe = 0
      pm5acc = 0
      pm10acc = 0
    else:
      mse = metrics.mean_squared_error(prof_data, pred_data)
      rmse = metrics.mean_squared_error(prof_data, pred_data, squared=False)
      rmspe = EvalMetricUtils.RootMeanSquarePercentageError(prof_data.values, pred_data)
      pm5acc = EvalMetricUtils.PlusMinusAccuracy(prof_data.values, pred_data, 5)
      pm10acc = EvalMetricUtils.PlusMinusAccuracy(prof_data.values, pred_data, 10)
      pm20acc = EvalMetricUtils.PlusMinusAccuracy(prof_data.values, pred_data, 20)
      pm30acc = EvalMetricUtils.PlusMinusAccuracy(prof_data.values, pred_data, 30)
      pm40acc = EvalMetricUtils.PlusMinusAccuracy(prof_data.values, pred_data, 40)
    return {'mse': mse, 'rmse': rmse, 'rmspe': rmspe,
            'pm5acc': pm5acc, 'pm10acc': pm10acc,
            'pm20acc': pm20acc, 'pm30acc': pm30acc,
            'pm40acc': pm40acc}

  def _ret_dict_to_string(self, ret_dict):
    text = ''
    if ret_dict is None:
      return text
    for key in ret_dict.keys():
      if key == 'pred_data':
        continue
      text = '{:s} {:s} {:.3f}'.format(text, key, ret_dict[key])
    return text

  def _print_ret_dict(self, ret_dict):
    table = Table(show_header=True, header_style='bold magenta')
    table.add_column('MODEL')
    table.add_column('MSE')
    table.add_column('RMSE')
    table.add_column('RMSPE')
    table.add_column('PM10ACC')
    table.add_column('TIME')
    table.add_row(self._model_name,
                  '%.3f' % ret_dict['mse'],
                  '%.3f' % ret_dict['rmse'],
                  '%.3f' % ret_dict['rmspe'],
                  '%.3f' % ret_dict['pm10acc'],
                  '%.3f' % ret_dict['time_cost'])
    console = Console()
    console.print(table)

  def _fit(self, feat_data, prof_data, save_model=False):
    return None

  def _print_cost(self, prof_cost, pred_cost):
    for v0, v1 in zip(prof_cost, pred_cost):
      TerminalLogger.log(LogTag.INFO, '{},{}'.format(v0, v1))

  def update_model_config(self, model_config):
    self._is_train = False

  def get_run_time(self):
    return time.time() - self._total_start_time

  def get_run_time_text(self):
    return TimeUtils.sec_to_format_text(round(time.time() - self._total_start_time))

  def is_trained_feature(self, feat_data):
    if not self._is_train:
      raise ValueError('Model has not been trained')
    # A wrong way.
    '''
    for index, col in feat_data.iteritems():
      train_cols = self._train_feat_data.loc[:, index]

      #if col[0] not in train_cols:
      #  return False

      is_equal = False
      for i, v in train_cols.items():
        if col[0] == v:
          is_equal = True
          break
      
      if not is_equal:
        #print(str(col))
        #print(str(train_cols))
        #input('')
        return False
    '''
    
    for index, row in self._train_feat_data.iterrows():
      is_equal = True
      for index, col in feat_data.iteritems():
        if col[0] != row[index]:
          is_equal = False
          break
      if is_equal:
        return True

    #print(str(feat_data))
    #print(str(self._train_feat_data))
    #input('')

    return False

  def predict(self, feat_data):
    TerminalLogger.log(LogTag.DEBUG, 'ModelTrainer predict')
    if not self._is_train:
      raise ValueError('Model has not been trained')
    return None

  def train(self, train_indices=None):
    self._start_time = time.time()
    df = self._train_df
    if self._sample_type == SampleType.DEFAULT:
      train_count = int(self._train_ratio * len(df))
      feat_data = df.loc[:train_count, self._feature_list]
      prof_data = df.loc[:train_count, self._label_name]
    elif self._sample_type == SampleType.RANDOM:
      train_indices = self._train_indices if train_indices is None else train_indices
      #print(train_indices)
      #print(self._feature_list)
      #print(self._label_name)
      feat_data = df.loc[train_indices, self._feature_list]
      prof_data = df.loc[train_indices, self._label_name]
    elif self._sample_type == SampleType.ALL:
      train_count = len(df)
      feat_data = df.loc[:train_count, self._feature_list]
      prof_data = df.loc[:train_count, self._label_name]
    else:
      raise ValueError('Unsupported sample type ' + self._sample_type)

    self._is_train = True
    self._feat_data = feat_data
    self._prof_data = prof_data

    self._train_feat_data = feat_data

  def test(self, test_indices=None):
    if not self._is_train:
      raise ValueError('Model has not been trained')
    df = self._test_df

    self._start_time = time.time()

    if self._sample_type == SampleType.DEFAULT:
      train_count = int(self._train_ratio * len(df))
      feat_data = df.loc[train_count:, self._feature_list]
      prof_data = df.loc[train_count:, self._label_name]
    elif self._sample_type == SampleType.RANDOM:
      if test_indices is None:
        test_indices = self._test_indices
      feat_data = df.loc[test_indices, self._feature_list]
      prof_data = df.loc[test_indices, self._label_name]
    elif self._sample_type == SampleType.ALL:
      train_count = len(df)
      feat_data = df.loc[:train_count, self._feature_list]
      prof_data = df.loc[:train_count, self._label_name]
    else:
      raise ValueError('Unsupported sample type ' + self._sample_type)

    self._feat_data = feat_data
    self._prof_data = prof_data

  def print_model_json(self):
    print('No model json')

class MultiModelTrainer(object):
  def __init__(self,
               model_name,
               sub_model_names,
               train_data_filename,
               test_data_filename,
               label_names,
               feature_multi_list,
               model_config=None,
               save_model=True,
               load_model=True):
    self._num_models = len(sub_model_names)
    self._train_df = pd.read_csv(train_data_filename).fillna(-1)
    self._test_df = pd.read_csv(test_data_filename).fillna(-1)
    self._train_ratio = kDefaultTrainRatio
    self._sample_type = kDefaultSampleType
    self._train_indices, self._test_indices = \
        calc_train_test_indices(self._sample_type,
                                self._train_df,
                                self._test_df,
                                self._train_ratio)
    self._trainers = []

  def get_train_dataframe(self):
    return self._train_df

  def get_test_dataframe(self):
    return self._test_df

  def get_test_indices(self):
    return self._test_indices

  def update_model_config(self, model_config):
    for trainer in self._trainers:
      trainer.update_model_config(model_config)

  def train(self):
    for trainer in self._trainers:
      trainer.train(self._train_indices)

  def test(self):
    pred_data_list = []
    ret_dict_list = []
    for trainer in self._trainers:
      ret_dict = trainer.test(self._test_indices)
      pred_data_list.append(ret_dict['pred_data'])
      ret_dict_list.append(ret_dict)
    return pred_data_list, ret_dict_list

  def is_trained_feature(self, feat_data_list):
    for trainer, feat_data in zip(self._trainers, feat_data_list):
      if not trainer.is_trained_feature(feat_data):
        return False
    return True

  def predict(self, feat_data_list):
    TerminalLogger.log(LogTag.DEBUG, 'MultiModelTrainer predict')
    pred_data_list = []
    for trainer, feat_data in zip(self._trainers, feat_data_list):
      pred_data_list.append(trainer.predict(feat_data))
    return pred_data_list

  def print_model_json(self):
    for trainer in self._trainers:
      trainer.print_model_json()

class OperatorTrainer(object):
  def __init__(self, model_config=None, save_model=True, load_model=True):
    self._multi_trainer = None
    self._is_debug = kLocalConfig['is_op_trainer_debug']

  def _read_prof_cost(self, data_filename, test_indices):
    df = pd.read_csv(data_filename).fillna(-1)
    return df.loc[test_indices, 'T_OP']

  def _build_null_eval_result(self):
    return {'mse': 0, 'rmse': 0, 'rmspe': 0,
            'pm5acc': 0, 'pm10acc': 0, 'pm20acc': 0,
            'pm30acc': 0, 'pm40acc': 0}

  def _print_cost(self, prof_cost, pred_cost):
    idx = 0
    for v0, v1 in zip(prof_cost, pred_cost):
      err = abs(v0 - v1) * 100.0 / v0
      TerminalLogger.log(LogTag.INFO,
          '{:d},{:.3f},{:.3f},{:.3f}%'.format(idx, v0, v1, err))
      idx = idx + 1

  def _calc_eval_metric(self, prof_cost, pred_cost):
    if len(prof_cost) != len(pred_cost):
      raise ValueError('Cost list length should be the same,' \
          ' of while prof is {} and pred is {}'.format(
              len(prof_cost), len(pred_cost)))

    mse = EvalMetricUtils.MeanSquaredError(prof_cost, pred_cost)
    rmse = EvalMetricUtils.RootMeanSquaredError(prof_cost, pred_cost)
    rmspe = EvalMetricUtils.RootMeanSquarePercentageError(prof_cost, pred_cost)
    pm5acc = EvalMetricUtils.PlusMinusAccuracy(prof_cost, pred_cost, 5)
    pm10acc = EvalMetricUtils.PlusMinusAccuracy(prof_cost, pred_cost, 10)
    pm20acc = EvalMetricUtils.PlusMinusAccuracy(prof_cost, pred_cost, 20)
    pm30acc = EvalMetricUtils.PlusMinusAccuracy(prof_cost, pred_cost, 30)
    pm40acc = EvalMetricUtils.PlusMinusAccuracy(prof_cost, pred_cost, 40)

    do_draw = kLocalConfig['do_op_trainer_draw']
    if do_draw:
      x = np.array(range(0, np.size(prof_cost)))
      PlotUtils.multi_draw_2d([x, x], [prof_cost, pred_cost],
          y_labels=['PROF', 'PRED'], xlabel_name='#', ylabel_name='Latency (ms)')

    return {'mse': mse, 'rmse': rmse, 'rmspe': rmspe,
            'pm5acc': pm5acc, 'pm10acc': pm10acc, 'pm20acc': pm20acc,
            'pm30acc': pm30acc, 'pm40acc': pm40acc}

  def _print_eval_result(self, eval_result):
    print_eval_result(self._model_name, eval_result)

  def update_model_config(self, model_config):
    if self._multi_trainer is not None:
      self._multi_trainer.update_model_config(model_config)

  def train(self):
    if self._multi_trainer is not None:
      self._multi_trainer.train()

  def test_internal(self):
    return

  def test(self):
    if self._multi_trainer is not None:
      return self._multi_trainer.test()

  def test_op(self):
    return

  def is_trained_feature(self, feat_data_list):
    if self._multi_trainer is not None:
      return self._multi_trainer.is_trained_feature(feat_data_list)

  def predict(self, feat_data_list):
    TerminalLogger.log(LogTag.DEBUG, 'OperatorTrainer predict')
    if self._multi_trainer is not None:
      return self._multi_trainer.predict(feat_data_list)

  def print_model_json(self):
    if self._multi_trainer is not None:
      self._multi_trainer.print_model_json()

class MultiOperatorTrainer(object):
  def __init__(self):
    self._multi_op_trainer = []
    self._is_debug = kLocalConfig['is_op_trainer_debug']

  def _avg_eval_results(self, eval_results):
    avg_eval_result = {
        'mse': 0, 'rmse': 0, 'rmspe': 0,
        'pm5acc': 0, 'pm10acc': 0, 'pm20acc': 0,
        'pm30acc': 0, 'pm40acc': 0}
    if len(eval_results) > 0:
      for r in eval_results:
        for key in r.keys():
          avg_eval_result[key] = avg_eval_result[key] + r[key]

      for key in avg_eval_result.keys():
        avg_eval_result[key] = avg_eval_result[key] / len(eval_results)

    return avg_eval_result

  def _print_eval_result(self, eval_result):
    print_eval_result(self._model_name, eval_result)

  def train(self):
    for trainer in self._multi_op_trainer:
      trainer.train()

  def test_op(self):
    eval_results = []
    for trainer in self._multi_op_trainer:
      eval_result = trainer.test_op()
      eval_results.append(eval_result)

    if self._is_debug:
      eval_results = self._avg_eval_results(eval_results)
      self._print_eval_result(eval_results)

  def print_model_json(self):
    for trainer in self._multi_op_trainer:
      trainer.print_model_json()
