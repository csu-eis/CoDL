
from utils.common.math import *
from utils.common.basic_type import Device
from utils.op.op_common_calc import *
from utils.op.op_param_utils import *
from utils.op.matmul_common_calc import *

class MatMulParamUtils(OpParamUtils):
  def extract_from_row(self, row):
    param = {}
    batch = int(row['B'])
    M = int(row['M'])
    N = int(row['N'])
    K = int(row['K'])
    transpose_a = int(row['TA'])
    transpose_b = int(row['TB'])
    if batch == 1:
      param['input_shape'] = [M, K] if not transpose_a else [K, M]
      param['rhs_shape'] = [K, N] if not transpose_b else [N, K]
    elif batch > 1:
      param['input_shape'] = [1, batch, M, K] if not transpose_a else [1, batch, K, M]
      param['rhs_shape'] = [1, batch, K, N] if not transpose_b else [1, batch, N, K]

    param['transpose_a'] = True if transpose_a == 1 else False
    param['transpose_b'] = True if transpose_b == 1 else False
    return param

  def _is_param_equal(self, a, b):
    def is_list_equal(a, b, iname, ilen):
      if len(a[iname]) != len(b[iname]):
        return False
      for i in range(ilen):
        if a[iname][i] != b[iname][i]:
          return False
      return True

    rank = len(a['input_shape'])

    iname_list = ['input_shape', 'rhs_shape']
    ilen_list = [rank, rank]
    for iname, ilen in zip(iname_list, ilen_list):
      if not is_list_equal(a, b, iname, ilen):
        return False

    return True

  @staticmethod
  def get_batch_size(param):
    rank = len(param['input_shape'])
    return param['input_shape'][1] if rank == 4 else 1

  @staticmethod
  def get_m_size(param):
    m_idx = 0 if not param['transpose_a'] else 1
    return param['input_shape'][m_idx]

  @staticmethod
  def get_k_size(param):
    k_idx = 1 if not param['transpose_a'] else 0
    return param['input_shape'][k_idx]

  @staticmethod
  def get_n_size(param):
    n_idx = 1 if not param['transpose_b'] else 0
    return param['rhs_shape'][n_idx]

  def _set_m_size(self, param, m):
    param['input_shape'] = param['input_shape'].copy()
    param['output_shape'] = param['output_shape'].copy()
    m_idx = 0 if not param['transpose_a'] else 1
    param['input_shape'][m_idx] = m
    param['output_shape'][0] = m
    return param

  def _set_batch_size(self, param, batch):
    param['input_shape'] = param['input_shape'].copy()
    param['rhs_shape'] = param['rhs_shape'].copy()
    param['output_shape'] = param['output_shape'].copy()
    param['input_shape'][1] = batch
    param['rhs_shape'][1] = batch
    param['output_shape'][1] = batch
    return param

  def calc_output_shape(self, param):
    rank = len(param['input_shape'])
    M = param['input_shape'][-2] if not param['transpose_a'] else param['input_shape'][-1]
    N = param['rhs_shape'][-1] if not param['transpose_b'] else param['rhs_shape'][-2]
    if rank == 2:
      param['output_shape'] = [M, N]
    elif rank == 4:
      batch = param['input_shape'][1]
      param['output_shape'] = [1, batch, M, N]
    return param

  def calc_partition_shape(self, param, pdim, pratio, device):
    out_param = param.copy()
    out_shape = param['output_shape']
    rank = len(param['input_shape'])
    M = out_shape[-2]
    N = out_shape[-1]
    if rank == 2:
      if device == Device.GPU:
        out_m = RoundUpMul(M, pratio, 1)
      else:
        out_m = M - RoundUpMul(M, pratio, 1)
      out_m = max(out_m, 1)

      out_param = self._set_m_size(out_param, out_m)
    elif rank == 4:
      batch = out_shape[-3]
      if device == Device.GPU:
        out_batch = RoundUpMul(batch, pratio, 1)
      else:
        out_batch = batch - RoundUpMul(batch, pratio, 1)
      out_batch = max(out_batch, 1)

      out_param = self._set_batch_size(out_param, out_batch)
    else:
      #raise ValueError('Unsupported partition dimension: ' + str(pdim))
      return None

    return out_param

  def calc_flops(self, param):
    return 0
