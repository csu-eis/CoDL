
from utils.model.model_utils import *
from utils.dataset.dataset_info import *
from utils.dataset.ml_dataset_utils import *
from train_or_test_rf_model_v2 import *

class Conv2dRFOursModelUtils(ModelUtils):
  def __init__(self):
    ModelUtils.__init__(self)
    self._dataset_info = MLConv2dDatasetInfoV2()
    self._is_debug = False

  def train(self, dataset_path, model_path, model_names=None):
    ModelUtils.train(self, dataset_path, model_path, model_names)
    return train_or_test_conv2d_dataset_rf_ours(dataset_path, model_path, model_names)

  def _predict_data_transform(self, models, features, conv2d_param):
    key_param = MLConv2dDataBaseUtils.GetDataTransformKeyParam(conv2d_param)

    features_dt_in = [features[0][6]]
    features_map_in = [features[0][6]]
    features_map_out = [features[0][7]]
    features_sync = [features[0][6]]
    features_dt_out = [features[0][7]]
    features_list = [features_dt_in,
                     features_map_in,
                     features_map_out,
                     features_sync,
                     features_dt_out]
    pred_data_list, is_trained_feature = self._do_model_predict_v2(
        models, 't_data_transform', features_list)
    t_tr_in = pred_data_list[0]
    t_map_in = pred_data_list[1]
    t_map_out = pred_data_list[2]
    t_sync = pred_data_list[3]
    t_tr_out = pred_data_list[4]

    cost_data_transform = t_tr_in + t_map_in + t_map_out + t_sync + t_tr_out

    return cost_data_transform, is_trained_feature

  def _predict_cpu_direct(self, models, features, conv2d_param):
    key_param = MLConv2dDataBaseUtils.GetCpuDirectKeyParam(conv2d_param)
    flops = key_param['flops']
    tile_size = key_param['tile_size']
    max_thread_tiles = key_param['max_thread_tiles']
    flops_per_tile = key_param['flops_per_tile']

    pred_data_list, is_trained_feature = self._do_model_predict_v2(
        models, 't_cpu_direct', features, False)
    t_flops = pred_data_list[0]

    cost_comp = max_thread_tiles * flops_per_tile * t_flops

    if self._is_debug:
      debug_text = 'flops {} tile_size {} max_thread_tiles {} flops_per_tile {} t_flops {}'.format(
         flops, tile_size, max_thread_tiles, flops_per_tile, t_flops)
      TerminalLogger.log(LogTag.INFO, debug_text)

    return cost_comp, is_trained_feature

  def _predict_cpu_gemm(self, models, features, conv2d_param):
    key_param = MLConv2dDataBaseUtils.GetCpuGemmKeyParam(conv2d_param)

    m = features[0][0]
    k = features[0][1]
    n = features[0][2]

    max_thread_mb = key_param['max_thread_mb']
    max_thread_nb = key_param['max_thread_nb']
    kb = key_param['kb']
    nb = key_param['nb']

    features_pack_lhs = [m, k]
    features_pack_rhs = [k, n]
    features_comp = [m, k, n]
    features_list = [features_pack_lhs, features_pack_rhs, features_comp]
    pred_data_list, is_trained_feature = self._do_model_predict_v2(
        models, 't_cpu_gemm', features_list)
    t_pack_lb = pred_data_list[0]
    t_pack_rb = pred_data_list[1]
    t_comp = pred_data_list[2]

    cost_pack_lb = t_pack_lb * max_thread_mb * kb
    cost_pack_rb = t_pack_rb * kb * max_thread_nb
    cost_comp = t_comp * max_thread_mb * kb * nb

    cost_gemm = cost_pack_lb + cost_pack_rb + cost_comp

    if self._is_debug:
      debug_text = 'm {} k {} n {}'.format(m, k, n)
      TerminalLogger.log(LogTag.INFO, debug_text)
      debug_text = 't_pack_lb {} t_pack_rb {} t_comp {}'.format(t_pack_lb, t_pack_rb, t_comp)
      TerminalLogger.log(LogTag.INFO, debug_text)
      debug_text = 'max_thread_mb {} max_thread_nb {} kb {} nb {}'.format(max_thread_mb, max_thread_nb, kb, nb)
      TerminalLogger.log(LogTag.INFO, debug_text)
      debug_text = 'cost_pack_lb {} cost_pack_rb {} cost_comp {}'.format(cost_pack_lb, cost_pack_rb, cost_comp)
      TerminalLogger.log(LogTag.INFO, debug_text)

    return cost_gemm, is_trained_feature

  def _predict_cpu_winograd(self, models, features, conv2d_param):
    wino_key_param = MLConv2dDataBaseUtils.GetCpuWinogradKeyParam(conv2d_param)

    features_pad = [features[0][0], features[0][1], features[0][2]]
    features_tr_in = [features[0][6], features[0][7], features[0][10]]
    features_tr_out = [features[0][8], features[0][9], features[0][10]]
    features_unpad = [features[0][3], features[0][4], features[0][5]]
    features_list = [features_pad, features_tr_in, features_tr_out, features_unpad]
    pred_data_list, is_trained_feature = self._do_model_predict_v2(
        models, 't_cpu_winograd', features_list, False)
    t_pad = pred_data_list[0]
    t_tr_in = pred_data_list[1]
    t_tr_out = pred_data_list[2]
    t_unpad = pred_data_list[3]

    cost_pad = t_pad * (conv2d_param['input_shape'][1] * \
        conv2d_param['input_shape'][2] * conv2d_param['input_shape'][3])
    cost_tr_in = t_tr_in * (wino_key_param['max_thread_ic'] * \
        wino_key_param['total_out_tile_count'])
    cost_tr_out = t_tr_out * (wino_key_param['max_thread_oc'] * \
        wino_key_param['total_out_tile_count'])
    cost_unpad = t_unpad * (conv2d_param['output_shape'][1] * \
        conv2d_param['output_shape'][2] * conv2d_param['output_shape'][3])

    gemm_param = MLConv2dDataBaseUtils.BuildGemmParamFromWinogradParam(
        conv2d_param, wino_key_param)
    gemm_key_param = MLConv2dDataBaseUtils.GetCpuGemmKeyParam(gemm_param)
    gemm_features = [[gemm_key_param['m'], gemm_key_param['k'], gemm_key_param['n'],
                      gemm_key_param['m'], gemm_key_param['k'], gemm_key_param['n']]]
    pred_data_list, _ = self._predict_cpu_gemm(models, gemm_features, gemm_param)
    cost_gemm = pow((wino_key_param['out_tile_size'] + 2), 2) * pred_data_list[0]

    cost_winograd = cost_pad + cost_tr_in + cost_gemm + cost_tr_out + cost_unpad
    if self._is_debug:
      TerminalLogger.log(
          LogTag.INFO, 'pad {} tr_in {} gemm {} tr_out {} unpad {}'.format(
              cost_pad, cost_tr_in, cost_gemm, cost_tr_out, cost_unpad))

    return cost_winograd, is_trained_feature

  def _predict_gpu_direct_warp(self, models, features, conv2d_param):
    key_param = MLConv2dDataBaseUtils.GetGpuDirectKeyParam(conv2d_param)

    pred_data_list, is_trained_feature = self._do_model_predict_v2(
        models, 't_gpu_direct', features, False)
    t_warp = pred_data_list[0]

    n_warp = key_param['n_warp']
    cost_warp = t_warp * n_warp

    if self._is_debug:
      TerminalLogger.log(LogTag.INFO, 't_warp {} n_warp {}'.format(t_warp, n_warp))

    return cost_warp, is_trained_feature

  def _predict_gpu_direct(self, models, features, conv2d_param):
    return self._predict_gpu_direct_warp(models, features, conv2d_param)

  def predict(self, models, features, conv2d_param, dev_name, do_data_transform=False):
    ModelUtils.predict(self, models, features, conv2d_param, dev_name, do_data_transform)

    if do_data_transform:
      return self._predict_data_transform(models, features, conv2d_param)

    if dev_name == 'CPU':
      filter_shape = conv2d_param['filter_shape']
      strides = conv2d_param['strides']
      if GetConv2dCpuImplType(filter_shape, strides) == Conv2dCpuImplType.GEMM:
        return self._predict_cpu_gemm(models, features, conv2d_param)
      elif GetConv2dCpuImplType(filter_shape, strides) == Conv2dCpuImplType.WINOGRAD:
        return self._predict_cpu_winograd(models, features, conv2d_param)
      else:
        return self._predict_cpu_direct(models, features, conv2d_param)
    elif dev_name == 'GPU':
      return self._predict_gpu_direct(models, features, conv2d_param)
    else:
      raise ValueError('Unsupported device ' + dev_name)
      return None

class Conv2dRFBlackBoxModelUtils(ModelUtils):
  def __init__(self):
    ModelUtils.__init__(self)
    self._dataset_info = MLConv2dOpDatasetInfo()

  def train(self, dataset_path, model_path, model_names=None):
    ModelUtils.train(self, dataset_path, model_path, model_names)
    return train_or_test_conv2d_dataset_rf_bb(dataset_path, model_path, model_names)

  def _predict_cpu_or_gpu(self, models, features, conv2d_param, model_name):
    target_features = [[]]
    for feature in features[0]:
      target_features[0].append(feature)
    pred_data_list, is_trained_feature = self._do_model_predict_v2(
        models, model_name, target_features)
    pred_cost = pred_data_list[0]
    return pred_cost, is_trained_feature

  def _predict_cpu(self, models, features, conv2d_param):
    return self._predict_cpu_or_gpu(models, features, conv2d_param, 't_op_cpu')

  def _predict_gpu(self, models, features, conv2d_param):
    return self._predict_cpu_or_gpu(models, features, conv2d_param, 't_op_gpu')

  def predict(self, models, features, conv2d_param, dev_name, do_data_transform=False):
    if dev_name == 'CPU':
      if not do_data_transform:
        return self._predict_cpu(models, features, conv2d_param)
      else:
        return [0], False
    elif dev_name == 'GPU':
      return self._predict_gpu(models, features, conv2d_param)
    else:
      raise ValueError('Unsupported device ' + dev_name)
      return None
