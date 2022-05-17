
import os
import tkinter
from enum import Enum

def play_sound():
  duration = 2 # Second
  freq = 400 # Hz
  times = 1
  for i in range(times):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

def show_messagebox(title, msg):
  tkinter.messagebox.showinfo(title, msg)

def calc_optimal_pratio(lat_cpu, lat_gpu):
  return lat_cpu / (lat_gpu + lat_cpu)

def calc_ideal_speedup(la, lb):
  return min(la, lb) * (la + lb) / (la * lb)

def calc_total_result(res_list):
  lat_cpu_sum = 0
  lat_gpu_sum = 0
  lat_cpu_gpu_sum = 0
  IDX_LAT_CPU = 3
  IDX_LAT_GPU = 4
  IDX_LAT_CPUGPU = 5
  IDX_PDIM = 6
  IDX_PRATIO = 7

  pdim_str = ''
  pratio_str = ''
  for i in range(len(res_list)):
    res_text = res_list[i]
    texts = res_text.split(',')
    lat_cpu_sum = lat_cpu_sum + float(texts[IDX_LAT_CPU])
    lat_gpu_sum = lat_gpu_sum + float(texts[IDX_LAT_GPU])
    lat_cpu_gpu_sum = lat_cpu_gpu_sum + float(texts[IDX_LAT_CPUGPU])
    
    if i < len(res_list) - 1:
      pdim_str = pdim_str + texts[IDX_PDIM] + ' '
      pratio_str = pratio_str + texts[IDX_PRATIO] + ' '
    else:
      pdim_str = pdim_str + texts[IDX_PDIM]
      pratio_str = pratio_str + texts[IDX_PRATIO]
  speedup_ideal = calc_ideal_speedup(lat_cpu_sum, lat_gpu_sum)
  speedup = min(lat_cpu_sum, lat_gpu_sum) / lat_cpu_gpu_sum
  ret = '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(
      lat_cpu_sum, lat_gpu_sum, lat_cpu_gpu_sum, speedup_ideal, speedup)
  return ret, pdim_str, pratio_str
