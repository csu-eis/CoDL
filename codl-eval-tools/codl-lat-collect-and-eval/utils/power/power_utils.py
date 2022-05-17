
import time
import threading
import numpy as np
from utils.common.adb import AdbUtils

kDebug = False

class BatteyAdbUtils(object):
  @staticmethod
  def read_capacity(adb_devname=None):
    capacity_path = '/sys/class/power_supply/battery/capacity'
    cap = int(AdbUtils.cat(capacity_path, adb_devname))
    return cap

class BatteryPowerAdbUtils(object):
  @staticmethod
  def read_battery_power(adb_devname=None):
    battery_path = '/sys/class/power_supply/battery'
    vol_path = battery_path + '/voltage_now'
    cur_path = battery_path + '/current_now'
    charge_cur_path = battery_path + '/constant_charge_current'
    vol_val = float(AdbUtils.cat(vol_path, adb_devname)) / 1000000.0
    cur_val = float(AdbUtils.cat(cur_path, adb_devname)) / 1000000.0
    charge_cur_val = float(AdbUtils.cat(charge_cur_path, adb_devname)) / 1000000.0
    pow_val = vol_val * cur_val
    if kDebug:
      print('vol %.2f V, cur %.2f A, ch_cur %.2f A, power %.2f W' % (
          vol_val, cur_val, charge_cur_val, pow_val))
    return pow_val, vol_val, cur_val

class BatteryPowerSampler(object):
  def __init__(self, sample_time=1.0, adb_devname=None, enable_debug=False):
    self._adb_devname = adb_devname
    self._delay = 2
    self._sample_time = sample_time
    self._reset()
    self._enable_debug = enable_debug

  def _reset(self):
    self._is_terminated = False
    self._power_list = []

  def _work_task(self):
    # Delay.
    time.sleep(self._delay)
    sample_count = 0
    while not self._is_terminated:
      start_time = time.time()
      pow_val, vol_val, cur_val = BatteryPowerAdbUtils.read_battery_power(self._adb_devname)
      bat_cap = BatteyAdbUtils.read_capacity(self._adb_devname)
      sample_count = sample_count + 1
      if self._enable_debug:
        print('BatteryPowerSampler: count {:d}, power {:.3f} W, vol {:.3f} V, cur {:.3f} A, bat {:d} %'.format(
            sample_count, pow_val, vol_val, cur_val, bat_cap))
      self._power_list.append(pow_val)
      sleep_time = self._sample_time - (time.time() - start_time)
      if sleep_time > 0:
        time.sleep(sleep_time)

  def start(self):
    # Reset.
    self._reset()
    # Create working thread.
    self._work_thread = threading.Thread(target=self._work_task)
    self._work_thread.setDaemon(True)
    self._work_thread.start()

  def stop(self):
    if self._work_thread is not None:
      self._is_terminated = True
      self._work_thread.join()

  def get_avg_power(self, min_limit=None):
    power_arr = np.array(self._power_list)
    if min_limit is not None:
      power_arr = power_arr[np.where(power_arr >= min_limit)]
    if np.size(power_arr) == 0:
      return 0
    return np.mean(power_arr)
