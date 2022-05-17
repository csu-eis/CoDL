
from utils.common.math import RoundUpDiv

kMaxCostUsingSingleThread = 100
kTileCountPerThread = 2

class ThreadPool(object):
  def __init__(self, num_threads, is_debug=False):
    self._num_threads = num_threads
    self._default_tile_count = num_threads * kTileCountPerThread
    self._is_debug = is_debug

  def _run(self, iterations):
    thread_count = self._num_threads
    iters_per_thread = iterations // thread_count
    remainder = iterations % thread_count

    threads_range_len = []
    for i in range(thread_count):
      range_len = iters_per_thread + 1 if i < remainder else iters_per_thread
      threads_range_len.append(range_len)

    if self._is_debug:
      print('ThreadRun: threads_range_len {}'.format(threads_range_len))

    return threads_range_len

  def compute_1d(self, s, e, st, ts=0, cpi=-1):
    ret_info = {}

    items = 1 + (e - s - 1) // st
    if self._num_threads <= 1 or (cpi >= 0 and items * cpi < kMaxCostUsingSingleThread):
      ret_info['tile_size'] = 1
      ret_info['max_tile_in_threads'] = e - s
      return ret_info
    else:
      if ts == 0:
        ts = max(1, items // self._default_tile_count)

      sts = st * ts
      tc = RoundUpDiv(items, ts)

      if self._is_debug:
        print('compute_1d: items {} ts {} tc {}'.format(items, ts, tc))

      thread_ranges = self._run(tc)

      ret_info['tile_size'] = ts
      ret_info['max_tile_in_threads'] = max(thread_ranges)
      return ret_info

  def compute_2d(self, s0, e0, st0, s1, e1, st1, ts0=0, ts1=0, cpi=-1):
    ret_info = {}

    items0 = 1 + (e0 - s0 - 1) // st0
    items1 = 1 + (e1 - s1 - 1) // st1
    if self._num_threads <= 1 or (cpi >= 0 and items0 * items1 * cpi < kMaxCostUsingSingleThread):
      ret_info['tile_size'] = 1
      ret_info['max_tile_in_threads'] = e1 - s1
      return ret_info
    else:
      if ts0 == 0 or ts1 == 0:
        if items0 >= self._default_tile_count:
          ts0 = items0 // self._default_tile_count
          ts1 = items1
        else:
          ts0 = 1
          ts1 = max(1, items1 * items0 // self._default_tile_count)

      sts0 = st0 * ts0
      sts1 = st1 * ts1
      tc0 = RoundUpDiv(items0, ts0)
      tc1 = RoundUpDiv(items1, ts1)

      if self._is_debug:
        print('compute_2d: items {} ts {} tc {}'.format(items1, ts1, tc1))

      thread_ranges = self._run(tc0 * tc1)

      ret_info['tile_size'] = ts1
      ret_info['max_tile_in_threads'] = max(thread_ranges)
      return ret_info
