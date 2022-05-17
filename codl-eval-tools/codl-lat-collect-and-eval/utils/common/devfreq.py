
from utils.common.adb import AdbUtils

class DevfreqUtils(object):
  def __init__(self, adb_devname=None):
    self._cpu_base_path = '/sys/devices/system/cpu'
    self._cpu_core_freqs = []
    self._gpu_core_freqs = []
    self._adb_devname = adb_devname
    return

  def _get_cur_cpu_core_freq(self, core_idx):
    filename = '{:s}/cpu{:d}/cpufreq/scaling_cur_freq'.format(self._cpu_base_path, core_idx)
    return AdbUtils.cat(filename, self._adb_devname)

  def _get_cur_gpu_core_freq(self):
    return

  def _set_cpu_core_freq(self, core_idx, freq_val):
    filename = '{:s}/cpu{:d}/cpufreq/scaling_setspeed'.format(self._cpu_base_path, core_idx)
    return AdbUtils.echo(filename, freq_val, self._adb_devname)

  def _set_gpu_core_freq(self, freq_val):
    return

  def get_cpu_freq_count(self):
    return len(self._cpu_core_freqs)

  def get_gpu_freq_count(self):
    return len(self._gpu_core_freqs)

  def get_cur_cpu_freq(self):
    return

  def get_cur_gpu_freq(self):
    return self._get_cur_gpu_core_freq()

  def get_cur_cpu_freq_scale(self):
    return

  def get_cur_gpu_freq_scale(self):
    return

  def set_cpu_freq_by_idx(self, freq_idx):
    return

  def set_gpu_freq_by_idx(self, freq_idx):
    DevfreqUtils.set_gpu_freq_by_idx(self, freq_idx)
    freq_val = self._gpu_core_freqs[freq_idx]
    self._set_gpu_core_freq(freq_val)

class SdmDevfreqUtils(DevfreqUtils):
  def _get_cur_gpu_core_freq(self):
    DevfreqUtils._get_cur_gpu_core_freq(self)
    filename = '/sys/class/kgsl/kgsl-3d0/gpuclk'
    return AdbUtils.cat(filename, self._adb_devname)

  def _set_gpu_core_freq(self, freq_val):
    DevfreqUtils._set_gpu_core_freq(self, freq_val)
    filename = '/sys/class/kgsl/kgsl-3d0/devfreq/userspace/set_freq'
    return AdbUtils.echo(filename, freq_val, self._adb_devname)

  def get_cur_cpu_freq(self):
    DevfreqUtils.get_cur_cpu_freq(self)
    freq_list = []
    kLittleCoreRange = range(0, 4)
    kBigCoreRange = range(4, 8)
    core_range = kBigCoreRange
    for i in core_range:
      freq_list.append(self._get_cur_cpu_core_freq(i))
    return freq_list

  def get_cur_cpu_freq_scale(self):
    SdmDevfreqUtils.get_cur_cpu_freq_scale(self)
    DevfreqUtils.get_cur_cpu_freq_scale(self)
    cur_val = float(self.get_cur_cpu_freq()[0])
    max_val = self._cpu_gold_core_freqs[0]
    return cur_val * 1.0 / max_val

  def get_cur_gpu_freq_scale(self):
    SdmDevfreqUtils.get_cur_gpu_freq_scale(self)
    cur_val = float(self.get_cur_gpu_freq())
    max_val = self._gpu_core_freqs[0]
    return cur_val * 1.0 / max_val

  def set_cpu_freq_by_idx(self, freq_idx):
    DevfreqUtils.set_cpu_freq_by_idx(self, freq_idx)
    # Core 4-6.
    freq_val = self._cpu_gold_core_freqs[freq_idx]
    self._set_cpu_core_freq(4, freq_val)
    self._set_cpu_core_freq(5, freq_val)
    self._set_cpu_core_freq(6, freq_val)
    # Core 7.
    freq_val = self._cpu_prime_core_freqs[freq_idx]
    self._set_cpu_core_freq(7, freq_val)

class Sdm855DevfreqUtils(SdmDevfreqUtils):
  def __init__(self, adb_devname=None):
    SdmDevfreqUtils.__init__(self, adb_devname)
    self._cpu_gold_core_freqs = [2419200,2323200,2227200,2131200,2016000,
                                 1920000,1804800,1708800,1612800,1497600,
                                 1401600,1286400,1171200,1056000,940800,
                                 825600,710400]
    self._cpu_prime_core_freqs = [2419200,2323200,2227200,2131200,2016000,
                                  1920000,1804800,1708800,1612800,1497600,
                                  1401600,1286400,1171200,1056000,940800,
                                  825600,825600]
    self._cpu_core_freqs = self._cpu_gold_core_freqs
    self._gpu_core_freqs = [585000000,499200000,427000000,345000000,257000000]

  def set_cpu_freq_by_percent(self, percent):
    if percent == 1.0:
      self.set_cpu_freq_by_idx(0)
    elif percent == 0.5:
      self.set_cpu_freq_by_idx(11)

  def set_gpu_freq_by_percent(self, percent):
    if percent == 1.0:
      self.set_gpu_freq_by_idx(0)
    elif percent == 0.5:
      self.set_gpu_freq_by_idx(4)

class Sdm865DevfreqUtils(SdmDevfreqUtils):
  def __init__(self, adb_devname=None):
    SdmDevfreqUtils.__init__(self, adb_devname)
    self._cpu_gold_core_freqs = [2419200,2342400,2246400,2150400,2054400,1958400,
                                 1862400,1766400,1670400,1574400,1478400,1382400,
                                 1286400,1171200,1056000,940800,825600,710400]
    self._cpu_prime_core_freqs = [2457600,2361600,2265600,2169600,2073600,1977600,
                                  1862400,1747200,1632000,1632000,1516800,1401600,
                                  1305600,1190400,1075200,960000,844800,844800]
    self._cpu_core_freqs = self._cpu_gold_core_freqs
    self._gpu_core_freqs = [587000000,525000000,490000000,441600000,400000000,305000000]

  def set_cpu_freq_by_percent(self, percent):
    if percent == 1.0:
      self.set_cpu_freq_by_idx(0)
    elif percent == 0.5:
      self.set_cpu_freq_by_idx(13)

  def set_gpu_freq_by_percent(self, percent):
    if percent == 1.0:
      self.set_gpu_freq_by_idx(0)
    elif percent == 0.5:
      self.set_gpu_freq_by_idx(5)

class Sdm888DevfreqUtils(SdmDevfreqUtils):
  def __init__(self, adb_devname=None):
    SdmDevfreqUtils.__init__(self, adb_devname)
    self._cpu_gold_core_freqs = []
    self._cpu_prime_core_freqs = []
    self._cpu_core_freqs = self._cpu_gold_core_freqs
    self._gpu_core_freqs = []

  def set_cpu_freq_by_percent(self, percent):
    return

  def set_gpu_freq_by_percent(self, percent):
    return

class KirinDevfreqUtils(DevfreqUtils):
  def get_cur_cpu_freq(self):
    DevfreqUtils.get_cur_cpu_freq(self)
    freq_list = []
    kLittleCoreRange = range(0, 4)
    kBigCoreRange = range(4, 8)
    core_range = kBigCoreRange
    for i in core_range:
      freq_list.append(self._get_cur_cpu_core_freq(i))
    return freq_list

class Kirin960DevfreqUtils(KirinDevfreqUtils):
  def _get_cur_gpu_core_freq(self):
    DevfreqUtils._get_cur_gpu_core_freq(self)
    filename = '/sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali/cur_freq'
    return AdbUtils.cat(filename, self._adb_devname)

  def _set_gpu_core_freq(self, freq_val):
    DevfreqUtils._set_gpu_core_freq(self, freq_val)
    filename = '/sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali/userspace/set_freq'
    return AdbUtils.echo(filename, freq_val, self._adb_devname)
