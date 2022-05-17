
from basic_type import *
from sklearn.ensemble import RandomForestRegressor

class NnMeterModelHelper(object):
  @staticmethod
  def BuildModel(op_type, compute_engine):
    model = None
    if op_type == OpType.Conv2D:
      if compute_engine == ComputeEngine.CPU:
        model = RandomForestRegressor(
            max_depth=70,
            n_estimators=320,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=6,
            oob_score=True,
            random_state=10)
      elif compute_engine == ComputeEngine.GPU:
        model = RandomForestRegressor(
            max_depth=80,
            n_estimators=550,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=5,
            oob_score=True,
            n_jobs=32,
            random_state=10)
    elif op_type == OpType.FullyConnected:
      if compute_engine == ComputeEngine.CPU:
        model = RandomForestRegressor(
            max_depth=50,
            n_estimators=370,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=2,
            oob_score=True,
            random_state=10)
      elif compute_engine == ComputeEngine.GPU:
        model = RandomForestRegressor(
            max_depth=70,
            n_estimators=330,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=4,
            oob_score=True,
            random_state=10)
    elif op_type == OpType.Pooling:
      if compute_engine == ComputeEngine.CPU:
        model = RandomForestRegressor(
            max_depth=50,
            n_estimators=210,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=5,
            oob_score=True,
            random_state=10)
      elif compute_engine == ComputeEngine.GPU:
        model = RandomForestRegressor(
            max_depth=50,
            n_estimators=370,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=5,
            oob_score=True,
            random_state=10)
    
    return model
