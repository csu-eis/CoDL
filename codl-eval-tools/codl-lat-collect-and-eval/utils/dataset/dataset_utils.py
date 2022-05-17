
import random

class DatasetUtils(object):
  @staticmethod
  def SimpleSplitIndex(count, ratio):
    target_train_count = int(count * ratio)
    train_indices = range(0, target_train_count)
    test_indices = range(target_train_count, count - 1)
    return train_indices, test_indices

  @staticmethod
  def RandomSplitIndex(count, ratio, seed=None):
    train_indices = []
    test_indices = []
    target_train_count = int(count * ratio)
    random.seed(seed)
    while len(train_indices) < target_train_count:
      i = random.randint(0, count - 1)
      if i not in train_indices:
        train_indices.append(i)
    for i in range(count):
      if i not in train_indices:
        test_indices.append(i)
    #print(train_indices)
    #print(test_indices)
    return train_indices, test_indices

if __name__ == '__main__':
  count = 1000
  ratio = 0.7
  train_indices, test_indices = DatasetUtils.RandomSplitIndex(count, ratio)
  print('Train indices ({}):'.format(len(train_indices)))
  print(train_indices)
  print('Test indices ({}):'.format(len(test_indices)))
  print(test_indices)
