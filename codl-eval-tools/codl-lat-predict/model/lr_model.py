
import numpy as np
import abc
from typing import Union
from utils.log_utils import *

class Model(object):
  @abc.abstractmethod
  def fit(self, x: Union[np.ndarray], y: Union[np.ndarray]):
    pass

  @abc.abstractmethod
  def predict(self, x: Union[np.ndarray]):
    pass

  @abc.abstractmethod
  def score(self, x: Union[np.ndarray], y: Union[np.ndarray]):
    pass

class GradDescentLRModel(Model):
  def score(self, x: Union[np.ndarray], y: Union[np.ndarray]):
    y_true = y
    y_pred = self.predict(x)

    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (
        (y_true - np.average(y_true, axis=0)) ** 2
    ).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    #print('nonzero_denominator {}, nonzero_numerator {}, valid_score {}'.format(
    #    nonzero_denominator, nonzero_numerator, valid_score
    #))
    if not valid_score:
      return 0
    
    output_scores = np.ones([y_true.shape[0]])
    output_scores[valid_score] = \
        1 - (numerator[valid_score] / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    return np.average(output_scores)

  def __init__(self, learning_rate=1e-2, num_epochs=1500):
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs
    self.theta = None

  def predict(self, x: Union[np.ndarray]):
    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    return np.dot(x, self.theta)

  def coefs(self):
    return self.theta[:-1]

  def inter(self):
    return self.theta[-1]

  def fit(self, x: Union[np.ndarray], y: Union[np.ndarray]):
    # add bias for x and theta
    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    self.theta = np.ones(x.shape[1], dtype=float)
    m, n = x.shape

    for i in range(self.num_epochs):
      y_pred = x.dot(self.theta)
      error = np.dot(x.T, (y_pred - y))
      self.theta -= self.learning_rate * 1 / m * error
      if i % 1000 == 0:
        LogUtil.print_log('epoch {} error {}'.format(i, error), LogLevel.DEBUG)

    return self
