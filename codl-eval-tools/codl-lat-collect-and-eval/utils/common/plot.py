
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.common.log import *

kDefaultFontname = 'Liberation Sans'

class PlotUtils(object):
  @staticmethod
  def draw_2d(x, y, xlabel_name=None, ylabel_name=None):
    plt.scatter(x, y, s=16, c='k', zorder=3)
    fontname = kDefaultFontname
    label_font = {'family': fontname, 'weight': 'bold', 'size': 12}
    plt.xlabel(xlabel_name, label_font)
    plt.ylabel(ylabel_name, label_font)
    plt.xticks(fontname=fontname, weight='bold')
    plt.yticks(fontname=fontname, weight='bold')
    plt.tick_params(labelsize=12)
    plt.grid(color='lightgray')
    plt.show()

  @staticmethod
  def multi_draw_2d(multi_x, multi_y, y_labels=None, xlabel_name=None, ylabel_name=None):
    num_lines = len(multi_x)
    if num_lines == 0:
      return

    TerminalLogger.log(LogTag.INFO, 'Number of x: {}'.format(len(multi_x[0])))

    lns = []
    for i in range(num_lines):
      x = multi_x[i]
      y = multi_y[i]
      label = y_labels[i] if y_labels is not None else None
      ln = plt.plot(x, y, label=label, marker='o', markersize=4)
      lns.append(ln)
    label_font = {'family': kDefaultFontname, 'weight': 'bold', 'size': 12}
    plt.legend(prop=label_font)
    plt.xlabel(xlabel_name, label_font)
    plt.ylabel(ylabel_name, label_font)
    plt.xticks(fontname=kDefaultFontname, weight='bold')
    plt.yticks(fontname=kDefaultFontname, weight='bold')
    plt.tick_params(labelsize=12)
    plt.grid(color='lightgray')
    plt.show()

  @staticmethod
  def draw_3d(feat1, feat2, target):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.array(feat1)
    y = np.array(feat2)
    z = np.zeros((np.shape(x)[0], np.shape(y)[0]))
    for i in range(np.shape(x)[0]):
      #print('i {} xi {} yi {} zi {}'.format(i, x[i], y[i], target[i]))
      z[i, i] = target[i]
    #print('shape: x {} y {} z {}'.format(np.shape(x), np.shape(y), np.shape(z)))
    x, y = np.meshgrid(x, y)
    ax.scatter(x, y, z, c=None, marker='.', s=50, label='')
    plt.show()
