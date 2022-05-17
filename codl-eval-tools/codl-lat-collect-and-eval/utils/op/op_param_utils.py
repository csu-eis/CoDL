
class OpParamUtils(object):
  def example(self):
    return None

  def extract_from_row(self, row):
    return None

  def _is_param_equal(self, a, b):
    return False

  def is_in_list(self, param, param_list):
    result = False
    a = param
    for b in param_list:
      if self._is_param_equal(a, b):
        result = True
        break
    return result

  def calc_output_shape(self, param):
    return None

  def calc_partition_shape(self, param, pdim, pratio, dev_name):
    return None

  def calc_flops(self, param):
    return 0
