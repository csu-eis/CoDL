
import os
import numpy as np
import argparse
from base.basic_type import *
from utils.data_loader import *
from lr_fit import *

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, required=True, help='Data file')
  parser.add_argument('--op_type', type=str, required=True, help='OP type')
  parser.add_argument('--device', type=str, required=True, help='Device')
  parser.add_argument('--show_coef', action='store_true', help='Show coefficient')
  parser.add_argument('--output_path', type=str, default=None, help='Output path')
  args = parser.parse_args()
  return args

def fit_test(data_file, op_type, device, show_coef, output_path):
  op_type = StringToOpType(op_type)
  if op_type == OpType.Conv2D:
    model_name = 'Conv2D-{:s}'.format(device)
    saved_model_name = 'mulayer_conv2d_{:s}'.format(device.lower())
    dl = Conv2dDataLoader(data_file)
    #dl.add_filter('SD_COST', '<', 5)
    #dl.add_filter('FLOPS', '<', 5 * 1e8)
    #dl.add_filter('T_OP', '>', 10)
    #x_names = ['FLOPS']
    x_names = ['F0', 'F1']
  elif op_type == OpType.FullyConnected:
    model_name = 'FullyConnected-{:s}'.format(device)
    saved_model_name = 'mulayer_fc_{:s}'.format(device.lower())
    dl = FullyConnectedDataLoader(data_file)
    #dl.add_filter('SD_COST', '<', 5)
    #x_names = ['FLOPS']
    x_names = ['F0', 'F1']
  elif op_type == OpType.Pooling:
    model_name = 'Pooling-{:s}'.format(device)
    saved_model_name = 'mulayer_pooling_{:s}'.format(device.lower())
    dl = PoolingDataLoader(data_file)
    #dl.add_filter('SD_COST', '<', 5)
    #x_names = ['PARAMS']
    x_names = ['F0', 'F1']
  else:
    raise ValueError('Unsupported OP type: ' + op_type)
    return None

  #dl.print_content()
  dl.split_data(frac=0.7, random_state=66)

  lr_type = 'grad_desc'  # ['sklearn', 'grad_desc']
  x_scale, y_scale = False, False
  use_logarithmic = False
  
  if lr_type == 'grad_desc':
    x_scale = True

  set_name = 'train'
  x = dl.get_features(x_names, set_name=set_name)
  if use_logarithmic:
    x = np.log(x)
  y = dl.get_target(set_name=set_name)

  acc, clf, means, stds = lr_fit(x, y, lr_type=lr_type,
                                 x_scale=x_scale, y_scale=y_scale,
                                 show_coef=show_coef, show_fig=False)

  set_name = 'test'
  y = dl.get_target(set_name=set_name)
  x = dl.get_features(x_names, set_name=set_name)

  acc = lr_test(clf, x, y,
                x_scale=x_scale, y_scale=y_scale, means=means, stds=stds)
  LogUtil.print_log('model {:s}, acc {:.2f}'.format(model_name, acc), LogLevel.INFO)

  if output_path is not None:
    json_text = lr_model_to_json(clf, lr_type, means, stds)
    LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
    save_model(output_path, saved_model_name + '.json', json_text)

if __name__ == '__main__':
  args = parse_args()

  data_file = args.data
  op_type = args.op_type
  device = args.device
  show_coef = args.show_coef
  output_path = args.output_path

  fit_test(data_file, op_type, device, show_coef, output_path)
