
import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from rich.console import Console
from rich.table import Column, Table
from utils.common.log import *
from utils.common.plot import PlotUtils
from utils.dataset.dataset_utils import DatasetUtils
from core.train_model_base import *
from config import *

def train_linear_regression_model(model_name,
                                  data_filename,
                                  label_name,
                                  feature_list):
  # Check data file.
  if data_filename is None:
    raise ValueError('Data file is none')
  if not os.path.exists(data_filename):
    raise ValueError('Data file does not exist')

  df = pd.read_csv(data_filename).fillna(-1)
  if len(df) == 0:
    return
  lr_model = linear_model.LinearRegression()
  train_ratio = kDefaultTrainRatio
  sample_type = kDefaultSampleType
  TerminalLogger.log(LogTag.INFO, '')
  TerminalLogger.log(LogTag.INFO, 'Model name: {}'.format(model_name))
  TerminalLogger.log(LogTag.INFO, 'Data file: {}'.format(data_filename))
  TerminalLogger.log(LogTag.INFO, 'Train sample ratio: {}'.format(train_ratio))
  TerminalLogger.log(LogTag.INFO, 'Sample type: {}'.format(sample_type.name))

  # Draw data.
  do_draw_data = False
  if do_draw_data:
    df_feature = df.loc[:, feature_list]
    df_label = df.loc[:, label_name]
    corrcoef_list = []
    if df_feature.columns.size == 1:
      corrcoef_list.append(
          np.corrcoef(df_feature.values[:,0], df_label.values)[0,1])
      TerminalLogger.log(LogTag.INFO, 'Corrcoef: {}'.format(corrcoef_list))
      PlotUtils.draw_2d(
          df_feature.values, df_label.values, 'FLOPs', 'Latency (ms)')
    elif df_feature.columns.size == 2:
      #PlotUtils.draw_3d(
      #    df_feature.values[:,0], df_feature.values[:,1], df_label.values)
      print('shape {} {} {}'.format(
          np.shape(df_feature.values[:,0]),
          np.shape(df_feature.values[:,1]),
          np.shape(df_label.values[:,0])))
      corrcoef_list.append(
          np.corrcoef(np.array(df_feature.values[:,0]), df_label.values[:,0])[0,1])
      corrcoef_list.append(
          np.corrcoef(np.array(df_feature.values[:,1]), df_label.values[:,0])[0,1])
      TerminalLogger.log(LogTag.INFO, 'Corrcoef: {}'.format(corrcoef_list))

      #PlotUtils.draw_2d(
      #    df_feature.values[:,0], df_label.values, 'F1', 'Latency (ms)')
      #PlotUtils.draw_2d(
      #    df_feature.values[:,1], df_label.values, 'F2', 'Latency (ms)')
      
      def draw_one_dimension(df, target_col, target_col_idx, sort_col, sort_col_idx):
        # Sort.
        df = df.sort_values(by=sort_col)
        print(df)
        df_feature = df.loc[:, feature_list]
        df_label = df.loc[:, label_name]

        old_v = -1
        for v in df_feature.values[:,sort_col_idx]:
          if v != old_v:
            old_v = v
            target_indices = df_feature.values[:,sort_col_idx] == v
            print('v {} num {}'.format(v, np.sum(target_indices == True)))
            print('x {}'.format(df_feature.values[target_indices, target_col_idx]))
            print('y {}'.format(df_label.values[target_indices, 0]))
            PlotUtils.draw_2d(
                df_feature.values[target_indices, target_col_idx],
                df_label.values[target_indices, 0], target_col, 'Latency (ms)')
      
      #draw_one_dimension(df, 'F0', 0, 'F1', 1)
      #draw_one_dimension(df, 'F1', 1, 'F0', 0)

  do_show_eval_metrics = 'v2'

  if sample_type == SampleType.RANDOM:
    train_indices, test_indices = \
        DatasetUtils.RandomSplitIndex(len(df), train_ratio, kDefaultSeed)
    #TerminalLogger.log(LogTag.INFO, 'train features:')
    #TerminalLogger.log(df.loc[train_indices, feature_list])
    #TerminalLogger.log(LogTag.INFO, 'train label:')
    #TerminalLogger.log(df.loc[train_indices, label_name])
    x_train = df.loc[train_indices, feature_list]
    y_train = df.loc[train_indices, label_name]
    # Fit.
    lr_model.fit(x_train, y_train)
    # Coef and intercept.
    coef = lr_model.coef_
    intercept = lr_model.intercept_
    TerminalLogger.log(LogTag.INFO, 'Coef: {}'.format(coef))
    TerminalLogger.log(LogTag.INFO, 'Inter: {}'.format(intercept))
    if do_show_eval_metrics:
      # R2 score.
      r2 = lr_model.score(x_train, y_train)
      # Profiled and predicted data.
      ytrue_train = df.loc[train_indices, label_name]
      ypred_train = lr_model.predict(df.loc[train_indices, feature_list])
      ytrue_test = df.loc[test_indices, label_name]
      ypred_test = lr_model.predict(df.loc[test_indices, feature_list])
      # MSE metrics.
      mse_train = metrics.mean_squared_error(ytrue_train, ypred_train)
      mse_test = metrics.mean_squared_error(ytrue_test, ypred_test)
      # RMSE metrics.
      rmse_train = EvalMetricUtils.RootMeanSquaredError(ytrue_train.values, ypred_train)
      rmse_test = EvalMetricUtils.RootMeanSquaredError(ytrue_test.values, ypred_test)
      # RMSPE metrics.
      rmspe_train = EvalMetricUtils.RootMeanSquarePercentageError(ytrue_train.values, ypred_train)
      rmspe_test = EvalMetricUtils.RootMeanSquarePercentageError(ytrue_test.values, ypred_test)
      # PM 10-40% accurawcy metrics.
      pm10acc_train = EvalMetricUtils.PlusMinusAccuracy(ytrue_train.values, ypred_train, 10)
      pm10acc_test = EvalMetricUtils.PlusMinusAccuracy(ytrue_test.values, ypred_test, 10)
      pm20acc_train = EvalMetricUtils.PlusMinusAccuracy(ytrue_train.values, ypred_train, 20)
      pm20acc_test = EvalMetricUtils.PlusMinusAccuracy(ytrue_test.values, ypred_test, 20)
      pm30acc_train = EvalMetricUtils.PlusMinusAccuracy(ytrue_train.values, ypred_train, 30)
      pm30acc_test = EvalMetricUtils.PlusMinusAccuracy(ytrue_test.values, ypred_test, 30)
      pm40acc_train = EvalMetricUtils.PlusMinusAccuracy(ytrue_train.values, ypred_train, 40)
      pm40acc_test = EvalMetricUtils.PlusMinusAccuracy(ytrue_test.values, ypred_test, 40)

  if do_show_eval_metrics == 'v1':
    TerminalLogger.log(LogTag.INFO, 'R2 (train): {:.3f}'.format(r2))
    TerminalLogger.log(LogTag.INFO, 'MSE (train): {:.3f}'.format(mse_train))
    TerminalLogger.log(LogTag.INFO, 'MSE (test): {:.3f}'.format(mse_test))
    TerminalLogger.log(LogTag.INFO, 'RMSE (train): {:.3f}'.format(rmse_train))
    TerminalLogger.log(LogTag.INFO, 'RMSE (test): {:.3f}'.format(rmse_test))
    TerminalLogger.log(LogTag.INFO, 'RMSPE (train): {:.3f}'.format(rmspe_train))
    TerminalLogger.log(LogTag.INFO, 'RMSPE (test): {:.3f}'.format(rmspe_test))
    TerminalLogger.log(LogTag.INFO, 'PM10ACC (train): {:.3f}'.format(pm10acc_train))
    TerminalLogger.log(LogTag.INFO, 'PM10ACC (test): {:.3f}'.format(pm10acc_test))
    TerminalLogger.log(LogTag.INFO, 'PM20ACC (train): {:.3f}'.format(pm20acc_train))
    TerminalLogger.log(LogTag.INFO, 'PM20ACC (test): {:.3f}'.format(pm20acc_test))
    TerminalLogger.log(LogTag.INFO, 'PM30ACC (train): {:.3f}'.format(pm30acc_train))
    TerminalLogger.log(LogTag.INFO, 'PM30ACC (test): {:.3f}'.format(pm30acc_test))
    TerminalLogger.log(LogTag.INFO, 'PM40ACC (train): {:.3f}'.format(pm40acc_train))
    TerminalLogger.log(LogTag.INFO, 'PM40ACC (test): {:.3f}'.format(pm40acc_test))

  if do_show_eval_metrics == 'v2':
    table = Table(show_header=True, header_style='bold magenta')
    table.add_column('MODEL')
    table.add_column('SPLIT')
    table.add_column('MSE')
    table.add_column('RMSE')
    table.add_column('RMSPE')
    table.add_column('PM10ACC')
    table.add_row(model_name,
                  'train',
                  '%.3f' % mse_train,
                  '%.3f' % rmse_train,
                  '%.3f' % rmspe_train,
                  '%.3f' % pm10acc_train)
    table.add_row('',
                  'test',
                  '%.3f' % mse_test,
                  '%.3f' % rmse_test,
                  '%.3f' % rmspe_test,
                  '%.3f' % pm10acc_test)
    console = Console()
    console.print(table)

  return lr_model

def train_dataset_model(dataset_path, dataset_info):
  model_names = dataset_info.model_names()
  out_model_list = []
  for model_name in model_names:
    label_name = dataset_info.label_names()[model_name]
    feature_list = dataset_info.feature_names()[model_name]

    data_filename = find_data_filepath(dataset_path, model_name)
    model = train_linear_regression_model(model_name,
                                          data_filename,
                                          label_name,
                                          feature_list)
    out_model_list.append(model)
  return out_model_list
