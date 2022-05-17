
"""
Reference: https://github.com/zhaoxingfeng/RandomForest
"""

import os
import time
import math
import random
import joblib
import pickle
import numpy as np
from utils.common.log import *
from utils.common.file import *
from utils.model.model import *
from config import *

kDefaultTrainConfigFileName = 'models/rf_train_config.txt'

class Tree(object):
  def __init__(self):
    self.split_feature = None
    self.split_value = None
    self.leaf_value = None
    self.tree_left = None
    self.tree_right = None

  def calc_predict_value(self, dataset):
    if self.leaf_value is not None:
      return self.leaf_value
    elif dataset[self.split_feature] <= self.split_value:
      return self.tree_left.calc_predict_value(dataset)
    else:
      return self.tree_right.calc_predict_value(dataset)

  def describe_tree(self):
    if not self.tree_left and not self.tree_right:
      leaf_info = '{\"leaf_value\":' + str(self.leaf_value) + '}'
      return leaf_info
    left_info = self.tree_left.describe_tree()
    right_info = self.tree_right.describe_tree()
    tree_structure = '{\"split_feature\":\"' + str(self.split_feature) + '\"' \
                     ',\"split_value\":' + str(self.split_value) + \
                     ',\"left_tree\":' + left_info + \
                     ',\"right_tree\":' + right_info + '}'
    return tree_structure

class RandomForestRegressionConfig(object):
  def __init__(self,
               n_estimators=10,
               max_depth=-1,
               min_samples_split=2,
               min_samples_leaf=1,
               min_split_gain=0.0,
               colsample_bytree=None,
               subsample=0.8,
               random_state=None):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_samples_leaf = min_samples_leaf
    self.min_split_gain = min_split_gain
    self.colsample_bytree = colsample_bytree if colsample_bytree is not None else 'none'
    self.subsample = subsample
    self.random_state = random_state

  def to_string(self):
    return ('n_estimators {:d} max_depth {:d} min_samples_split {:d}'
        + ' min_samples_leaf {:d} min_split_gain {:.2f} colsample_bytree {:s}'
        + ' subsample {:.2f} random_state {:d}').format(
            self.n_estimators, self.max_depth, self.min_samples_split,
            self.min_samples_leaf, self.min_split_gain, self.colsample_bytree,
            self.subsample, self.random_state)

  @staticmethod
  def Default():
    return RandomForestRegressionConfig(n_estimators=5,
                                        max_depth=5,
                                        min_samples_split=50,
                                        min_samples_leaf=10,
                                        min_split_gain=0.0,
                                        colsample_bytree='sqrt',
                                        subsample=0.8,
                                        random_state=66)

class RandomForestRegression(Model):
  def __init__(self,
               n_estimators=10,
               max_depth=-1,
               min_samples_split=2,
               min_samples_leaf=1,
               min_split_gain=0.0,
               colsample_bytree=None,
               subsample=0.8,
               random_state=None,
               tag=None):
    self._n_estimators = n_estimators
    self._max_depth = max_depth if max_depth != -1 else float('inf')
    self._min_samples_split = min_samples_split
    self._min_samples_leaf = min_samples_leaf
    self._min_split_gain = min_split_gain
    self._colsample_bytree = colsample_bytree if colsample_bytree is not None else 'none'
    self._subsample = subsample
    self._random_state = random_state
    self._trees = None
    self._feature_importances = dict()
    self._tag = tag

  def __init__(self, config, tag=None):
    if config is None:
      config = RandomForestRegressionConfig.Default()
    self._n_estimators = config.n_estimators
    self._max_depth = config.max_depth if config.max_depth != -1 else float('inf')
    self._min_samples_split = config.min_samples_split
    self._min_samples_leaf = config.min_samples_leaf
    self._min_split_gain = config.min_split_gain
    self._colsample_bytree = config.colsample_bytree if config.colsample_bytree is not None else 'none'
    self._subsample = config.subsample
    self._random_state = config.random_state
    self._trees = None
    self._feature_importances = dict()
    self._tag = tag

  def _config_to_string(self):
    config_dict = {'n_estimators': self._n_estimators,
                   'max_depth': self._max_depth,
                   'min_samples_split': self._min_samples_split,
                   'min_samples_leaf': self._min_samples_leaf,
                   'min_split_gain': self._min_split_gain,
                   'colsample_bytree': self._colsample_bytree,
                   'subsample': self._subsample,
                   'random_state': self._random_state}
    return str(config_dict)
  
  def _calc_r2(self, left_targets, right_targets):
    r2 = 0
    for targets in [left_targets, right_targets]:
      mean = targets.mean()
      for dt in targets:
        r2 += (dt - mean) ** 2
    return r2

  def _choose_best_feature(self, dataset, targets):
    best_split_gain = float('inf')
    best_split_feature = None
    best_split_value = None

    for feature in dataset.columns:
      if dataset[feature].unique().__len__() <= 100:
        unique_values = sorted(dataset[feature].unique().tolist())
      else:
        unique_values = np.unique([np.percentile(dataset[feature], x)
                                   for x in np.linspace(0, 100, 100)])

      for split_value in unique_values:
        left_targets = targets[dataset[feature] <= split_value]
        right_targets = targets[dataset[feature] > split_value]
        split_gain = self._calc_r2(left_targets['label'], right_targets['label'])

        if split_gain < best_split_gain:
          best_split_feature = feature
          best_split_value = split_value
          best_split_gain = split_gain
    return best_split_feature, best_split_value, best_split_gain

  def _calc_leaf_value(self, targets):
    return targets.mean()

  def _calc_config_tag(self):
    def add(a, b):
      return a + b
    colsample_bytree_values_dict = {'none': 0, 'sqrt': 1, 'log2': 3}

    #TerminalLogger.log(LogTag.INFO, self._config_to_string())

    config_tag = 0
    config_tag = add(config_tag, self._n_estimators)
    if self._max_depth < float('inf'):
      config_tag = add(config_tag, self._max_depth)
    else:
      raise ValueError('Max depth should not be inf')
    config_tag = add(config_tag, self._min_samples_split)
    config_tag = add(config_tag, self._min_samples_leaf)
    config_tag = add(config_tag, int(self._min_split_gain * 100))
    config_tag = add(config_tag, colsample_bytree_values_dict[self._colsample_bytree])
    config_tag = add(config_tag, int(self._subsample * 100))
    config_tag = add(config_tag, self._random_state)

    return config_tag

  def _build_model_filename(self, path, file_type):
    config_tag = self._calc_config_tag()
    filename = 'tree'
    if self._tag is not None:
      filename = filename + '_' + self._tag
    filename = filename + '_' + str(config_tag) + '.' + file_type

    return os.path.join(path, filename)

  def _split_dataset(self, dataset, targets, split_feature, split_value):
    left_dataset = dataset[dataset[split_feature] <= split_value]
    left_targets = targets[dataset[split_feature] <= split_value]
    right_dataset = dataset[dataset[split_feature] > split_value]
    right_targets = targets[dataset[split_feature] > split_value]
    return left_dataset, right_dataset, left_targets, right_targets

  def _build_single_tree(self, dataset, targets, depth):
    if len(targets['label'].unique()) <= 1 or \
       dataset.__len__() <= self._min_samples_split:
      tree = Tree()
      tree.leaf_value = self._calc_leaf_value(targets['label'])
      return tree

    if depth < self._max_depth:
      best_split_feature, best_split_value, best_split_gain = \
          self._choose_best_feature(dataset, targets)
      left_dataset, right_dataset, left_targets, right_targets = \
          self._split_dataset(dataset, targets, best_split_feature, best_split_value)

      tree = Tree()
      if left_dataset.__len__() <= self._min_samples_leaf or \
         right_dataset.__len__() <= self._min_samples_leaf or \
         best_split_gain <= self._min_split_gain:
        tree.leaf_value = self._calc_leaf_value(targets['label'])
        return tree
      else:
        self._feature_importances[best_split_feature] = \
            self._feature_importances.get(best_split_feature, 0) + 1

        tree.split_feature = best_split_feature
        tree.split_value = best_split_value
        tree.tree_left = self._build_single_tree(left_dataset, left_targets, depth+1)
        tree.tree_right = self._build_single_tree(right_dataset, right_targets, depth+1)
        return tree
    else:
      tree = Tree()
      tree.leaf_value = self._calc_leaf_value(targets['label'])
      return tree

  def _parallel_build_trees(self, dataset, targets, random_state, debug=False):
    if random_state:
      random.seed(random_state)

    subcol_index = random.sample(dataset.columns.tolist(), self._num_colsample_bytree)
    dataset_stage = dataset.sample(n=int(self._subsample * len(dataset)),
                                   replace=True, 
                                   random_state=random_state).reset_index(drop=True)
    dataset_stage = dataset_stage.loc[:, subcol_index]
    targets_stage = targets.sample(n=int(self._subsample * len(dataset)),
                                   replace=True, 
                                   random_state=random_state).reset_index(drop=True)

    tree = self._build_single_tree(dataset_stage, targets_stage, depth=0)
    debug = False
    if debug:
      print(tree.describe_tree())
    return tree

  def _load_configure_from_file(self, filename):
    ret_dict = {}
    if os.path.exists(filename):
      for line in open(filename):
        data = line.split('=')
        key = data[0]
        value = data[1]
        if key == 'n_jobs':
          value = int(value)
        ret_dict[key] = value

      TerminalLogger.log(LogTag.DEBUG, 'Load config: {}'.format(ret_dict))
    else:
      ret_dict['n_jobs'] = -1
      ret_dict['backend'] = 'multiprocessing'

    return ret_dict

  def fit(self, dataset, targets):
    if self._trees is not None:
      #TerminalLogger.log(LogTag.INFO, 'Skip to fit the model')
      return

    targets = targets.to_frame(name='label')

    if self._random_state:
      #print('random state: {}'.format(self._random_state))
      random.seed(self._random_state)
    random_state_stages = random.sample(range(self._n_estimators),
                                        self._n_estimators)

    if self._colsample_bytree == 'sqrt':
      self._num_colsample_bytree = int(len(dataset.columns) ** 0.5)
    elif self._colsample_bytree == 'log2':
      self._num_colsample_bytree = int(math.log(len(dataset.columns)))
    else:
      self._num_colsample_bytree = len(dataset.columns)

    #print('colsample_bytree: {}'.format(self._colsample_bytree))
    #print('num_colsample_bytree: {}'.format(self._num_colsample_bytree))
    if self._num_colsample_bytree == 0:
      return

    config_dict = self._load_configure_from_file(kDefaultTrainConfigFileName)

    n_jobs = config_dict['n_jobs']  # [-1, 4, 8]
    backend = config_dict['backend']  # ['multiprocessing', 'threading']
    self._trees = joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        joblib.delayed(self._parallel_build_trees)(dataset, targets, random_state)
            for random_state in random_state_stages)

  def _parallel_predict(self, dataset, i):
    row = dataset.iloc[i]
    #TerminalLogger.log(LogTag.INFO, 'Row {}/{}'.format(i, len(dataset)))
    pred_list = []
    for tree in self._trees:
      pred_list.append(tree.calc_predict_value(row))
    return sum(pred_list) * 1.0 / len(pred_list)

  def parallel_predict(self, dataset):
    if self._trees is None:
      return None

    config_dict = self._load_configure_from_file(kDefaultTrainConfigFileName)
    n_jobs = config_dict['n_jobs']
    backend = 'threading'

    res = joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        joblib.delayed(self._parallel_predict)(dataset, i)
            for i in range(len(dataset)))

    return np.array(res)

  def predict(self, dataset):
    if self._trees is None:
      #raise ValueError('Tree is none')
      return None

    res = []
    for _, row in dataset.iterrows():
      pred_list = []
      for tree in self._trees:
        pred_list.append(tree.calc_predict_value(row))
      res.append(sum(pred_list) * 1.0 / len(pred_list))
    return np.array(res)

  def save(self, path):
    if self._trees is None:
      raise ValueError('Trees can not be none while saving')

    FileUtils.create_dir(path)
    filename = self._build_model_filename(path, 'pkl')
    if not os.path.exists(filename):
      with open(filename, 'wb') as f:
        tree_str = pickle.dumps(self._trees)
        #TerminalLogger.log(LogTag.INFO, tree_str)
        f.write(tree_str)
        TerminalLogger.log(LogTag.INFO, 'Save RF model file ' + filename)

  def load(self, path):
    filename = self._build_model_filename(path, 'pkl')
    #TerminalLogger.log(LogTag.INFO, 'Try to load model file ' + filename)
    if os.path.exists(filename):
      st = time.time()
      with open(filename, 'rb') as f:
        self._trees = pickle.loads(f.read())
      TerminalLogger.log(
          LogTag.INFO, 'Load RF model file {:s}, taking {:.3f} sec'.format(
              filename, time.time() - st))

  def get_json(self):
    if self._trees is None:
      print('Trees is none')
      return None

    forest_desc = '{'
    for i in range(len(self._trees)):
      tree = self._trees[i]
      tree_desc = tree.describe_tree()
      forest_desc = forest_desc + '\"tree{}\":{},'.format(i, tree_desc)
    forest_desc = forest_desc[:-1] if forest_desc[-1] == ',' else forest_desc
    forest_desc = forest_desc + '}'

    return forest_desc

  def save_json(self, path):
    if self._trees is None:
      raise ValueError('Trees can not be none while saving')

    FileUtils.create_dir(path)
    filename = self._build_model_filename(path, 'json')
    if not os.path.exists(filename):
      with open(filename, 'w') as f:
        f.write(self.get_json())
        TerminalLogger.log(LogTag.INFO, 'Save RF model file ' + filename)
