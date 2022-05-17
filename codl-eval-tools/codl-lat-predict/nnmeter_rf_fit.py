
import argparse
from basic_type import *
from data_loader import *
from nnmeter_rf_models import NnMeterModelHelper
from sklearn.preprocessing import scale
from metric import *

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, required=True, help='Data file')
  parser.add_argument('--op_type', type=str, required=True, help='OP type')
  parser.add_argument('--ce', type=str, required=True, help='Computing engine')
  args = parser.parse_args()
  return args

def nn_meter_rf_fit(x, y, op_type, compute_engine):
  print('==== nnMeter Fit Info =====')
  
  #x = scale(x)

  rf = NnMeterModelHelper.BuildModel(op_type, compute_engine)
  rf.fit(x, y)
  yt = rf.predict(x)

  pm10acc = PM10ACC(yt, y) * 100

  print('pm10acc {:.2f}'.format(pm10acc))

  return rf, pm10acc

def nn_meter_rf_test(rf, x, y):
  print('==== nnMeter Test Info =====')
  
  #x = scale(x)

  yt = rf.predict(x)

  pm10acc = PM10ACC(yt, y) * 100

  print('pm10acc {:.2f}'.format(pm10acc))

def fit_test(data_file,
             op_type,
             compute_engine):
  op_type = StringToOpType(op_type)
  compute_engine = StringToComputeEngine(compute_engine)
  if op_type == OpType.Conv2D:
    dl = Conv2dDataLoader(data_file)
    x_names = ['IH', 'IW', 'IC', 'OC', 'KH', 'SH', 'FLOPS', 'PARAMS']
  elif op_type == OpType.FullyConnected:
    dl = FullyConnectedDataLoader(data_file)
    x_names = ['IH', 'IW', 'IC', 'OC', 'KH', 'FLOPS', 'PARAMS']
  elif op_type == OpType.Pooling:
    dl = PoolingDataLoader(data_file)
    x_names = ['IH', 'IW', 'IC', 'KH', 'SH', 'PARAMS']
  else:
    raise ValueError('Unsupported OP type: ' + op_type)
    return None

  dl.split_data()

  set_name = 'train'
  x = dl.get_features(x_names, set_name=set_name)
  y = dl.get_target(set_name=set_name)

  rf, acc = nn_meter_rf_fit(x, y, op_type, compute_engine)

  set_name = 'test'
  y = dl.get_target(set_name=set_name)
  x = dl.get_features(x_names, set_name=set_name)

  acc = nn_meter_rf_test(rf, x, y)

if __name__ == '__main__':
  args = parse_args()

  data_file = args.data
  op_type = args.op_type
  compute_engine = args.ce

  fit_test(data_file, op_type, compute_engine)
