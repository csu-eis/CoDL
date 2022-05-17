
from utils.dataset.dataset_info import *
from utils.model.linear_regression_model import *

def train_flops_model(dataset_path):
  return train_dataset_model(dataset_path, FLOPsLRConv2dDatasetInfo())

def train_mulayer_model(dataset_path):
  return train_dataset_model(dataset_path, MuLayerLRConv2dDatasetInfo())

class FLOPsLRModelUtils(object):
  @staticmethod
  def train(dataset_path):
    return train_flops_model(dataset_path)

class MuLayerLRModelUtils(object):
  @staticmethod
  def train(dataset_path):
    return train_mulayer_model(dataset_path)
