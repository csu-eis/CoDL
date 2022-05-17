
from pandas import DataFrame
from utils.dataset.dataset_info import *

class ModelUtils(object):
  def __init__(self):
    self._is_debug = False
    self._dataset_info = None

  def train(self, dataset_path, model_path, model_names=None):
    return None

  def _to_dataframe(self, feat_names, feat_vals):
    data = {}
    for fname, fval in zip(feat_names, feat_vals):
      data[fname] = [fval]
    return DataFrame(data)

  def _do_model_predict(self, models, model_name, feat_vals):
    if model_name not in models.keys():
      return 0

    feat_names = MLConv2dDatasetInfo().feature_names()[model_name]
    feat_datafarme = self._to_dataframe(feat_names, feat_vals)
    return models[model_name].predict(feat_datafarme)[0]

  def _do_model_predict_v2(self, models, model_name, feat_vals_list, is_debug=False):
    if model_name not in models.keys():
      return 0
    feat_datafarme_list = []
    for i in range(len(feat_vals_list)):
      feat_names = self._dataset_info.feature_names()[model_name][i]
      feat_vals = feat_vals_list[i]
      feat_datafarme = self._to_dataframe(feat_names, feat_vals)
      feat_datafarme_list.append(feat_datafarme)
    
    if is_debug:
      print(feat_datafarme_list)

    model = models[model_name]
    pred_data_list = model.predict(feat_datafarme_list)
    #is_trained_feature = model.is_trained_feature(feat_datafarme_list)
    is_trained_feature = False

    return pred_data_list, is_trained_feature

  def predict(self, models, features, conv2d_param, dev_name, do_data_transform=False):
    return None
