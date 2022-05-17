
import os
import confuse
from enum import Enum

kDebug = False

class SocName(Enum):
  Unknown = 0
  Snapdragon855 = 1
  Snapdragon865 = 2
  Kirin960 = 3
  Kirin970 = 4
  Kirin980 = 5

class Soc(object):
  def __init__(self, name):
    self._name = name
    self._global_mem_cache_size = 0
    self._base_gpu_mem_cache_size = 16384
    self._compute_units = 0
    self._max_work_group_size = 0
    self._kernel_wave_size = 0
    self._config_path = os.path.join('utils', 'soc')

  def load_yaml(self, filepath):
    source = confuse.YamlSource(filepath)
    config = confuse.RootView([source])
    self._global_mem_cache_size = config['global_mem_cache_size'].get()
    self._compute_units = config['compute_units'].get()
    self._max_work_group_size = config['max_work_group_size'].get()
    self._kernel_wave_size = config['kernel_wave_size'].get()
    if kDebug:
      config_text = '{}'.format([self._global_mem_cache_size,
                                 self._compute_units,
                                 self._max_work_group_size,
                                 self._kernel_wave_size])
      text = 'name {}, config {}'.format(self._name, config_text)
      print(text)

  def global_mem_cache_size(self):
    return self._global_mem_cache_size

  def base_gpu_mem_cache_size(self):
    return self._base_gpu_mem_cache_size

  def compute_units(self):
    return self._compute_units

  def max_work_group_size(self):
    return self._max_work_group_size

  def kernel_wave_size(self):
    return self._kernel_wave_size

  @staticmethod
  def Create(soc_name):
    if soc_name == DeviceName.Snapdragon855.name:
      return Snapdragon855Device()
    elif soc_name == DeviceName.Snapdragon865.name:
      return Snapdragon865Device()
    elif soc_name == DeviceName.Kirin960.name:
      return Kirin960Device()
    else:
      raise ValueError('Unsupported device: ' + soc_name)
      return None

class Snapdragon855Soc(Soc):
  def __init__(self):
    Soc.__init__(self, 'sdm855')
    '''
    self._global_mem_cache_size = 131072
    self._compute_units = 2
    self._max_work_group_size = 1024
    self._kernel_wave_size = 64
    '''
    self.load_yaml(os.path.join(self._config_path, 'sdm855.yaml'))

class Snapdragon865Soc(Soc):
  def __init__(self):
    Soc.__init__(self, 'sdm865')
    '''
    self._global_mem_cache_size = 131072
    self._compute_units = 3
    self._max_work_group_size = 1024
    self._kernel_wave_size = 64
    '''
    self.load_yaml(os.path.join(self._config_path, 'sdm865.yaml'))

class Kirin960Soc(Soc):
  def __init__(self):
    Soc.__init__(self, 'kirin960')
    
    self._global_mem_cache_size = 524288
    self._compute_units = 8
    self._max_work_group_size = 384
    self._kernel_wave_size = 4
    
    #self.load_yaml(os.path.join(self._config_path, 'kirin960.yaml'))

class Kirin970Soc(Soc):
  def __init__(self):
    Soc.__init__(self, 'kirin970')
    self.load_yaml(os.path.join(self._config_path, 'kirin970.yaml'))

class Kirin980Soc(Soc):
  def __init__(self):
    Soc.__init__(self, 'kirin980')
    self.load_yaml(os.path.join(self._config_path, 'kirin980.yaml'))
