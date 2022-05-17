
import os
import argparse
import numpy as np
from sklearn.preprocessing import scale
from base.metric import *
from utils.log_utils import *
from utils.array_utils import *
from utils.data_loader import *
from lr_fit import lr_fit, lr_test, lr_model_to_json, save_model
from med_fit import median_fit, median_test

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, required=True, help='Data file')
  parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
  parser.add_argument('--frac', type=float, required=False, default=0.7, help='Dataset fraction')
  parser.add_argument('--unrolling_aware', action='store_true', help='Unrolling-aware')
  parser.add_argument('--show_coef', action='store_true', help='Show coefficient')
  parser.add_argument('--show_fig', action='store_true', help='Show figure')
  parser.add_argument('--output_path', type=str, required=False, default=None, help='Output path')
  args = parser.parse_args()
  return args

def print_acc_result(title,
                     med_true_count,
                     lr_true_count,
                     total_train_count,
                     total_test_count):
  med_acc = med_true_count * 100 / total_test_count
  lr_acc = lr_true_count / total_test_count
  print('\nTitle: {}'.format(title))
  print('Total train data count: {}'.format(total_train_count))
  print('Total test data count: {}'.format(total_test_count))
  print('Median acc: {:.2f} %'.format(med_acc))
  print('LR acc: {:.2f} %\n'.format(lr_acc))

def fit_test(data_file,
             dataset_name,
             frac=0.7,
             unrolling_aware=False,
             show_coef=False,
             show_fig=False,
             output_path=None):
  med_true_count = 0
  lr_true_count = 0
  total_train_data_count = 0
  total_data_count = 0
  lr_train_type = 'grad_desc'
  do_x_scale, do_y_scale = False, False
  if lr_train_type == 'grad_desc':
    do_x_scale = True
  
  if dataset_name == 'DATA-SHARING':
    dl = DataSharingDataLoader(data_file)
    #dl.add_filter('T_DT_IN', '>', 0.5)
    #dl.add_filter('T_MAP_IN', '>', 0.5)
    #dl.add_filter('T_MAP_OUT', '>', 0.5)
    #dl.add_filter('T_DT_OUT', '>', 0.5)
    dl.split_data(frac=frac)
    x_names_list = [['FLOPS_IN'], ['FLOPS_IN'], ['FLOPS_OUT'], ['FLOPS_OUT']]
    y_names = ['T_DT_IN', 'T_MAP_IN', 'T_MAP_OUT', 'T_DT_OUT']
    model_names = ['lr_t_dt_in', 'lr_t_map_in', 'lr_t_map_out', 'lr_t_dt_out']
    for i in range(len(y_names)):
      y_name = y_names[i]
      model_name = model_names[i]
      x_names = x_names_list[i]
      set_name = 'train'
      y = dl.get_target(y_name, set_name=set_name)
      x = dl.get_features(x_names, set_name=set_name)
      acc, clf, means, stds = lr_fit(x, y, x_names, 't',
                                     lr_type=lr_train_type,
                                     x_scale=do_x_scale, y_scale=do_y_scale,
                                     show_coef=show_coef, show_fig=show_fig)

      set_name = 'test'
      y = dl.get_target(y_name, set_name=set_name)
      data_count = dl.get_data_count(set_name=set_name)
      total_data_count = total_data_count + data_count

      x = dl.get_features(x_names, set_name=set_name)
      acc = lr_test(clf, x, y,
                    x_scale=do_x_scale, y_scale=do_y_scale,
                    means=means, stds=stds)
      lr_true_count = lr_true_count + int(data_count * acc / 100.0)

      if output_path is not None:
        json_text = lr_model_to_json(clf, lr_train_type, means, stds)
        LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
        save_model(output_path, model_name + '.json', json_text)

    lr_acc = lr_true_count / total_data_count * 100.0

    #LogUtil.print_log('==== TOTAL MODEL INFO (DATA SHARING) =====')
    #LogUtil.print_log('model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
    #    'DATA-SHARING', total_data_count, 0, lr_acc), LogLevel.INFO)
    LogUtil.print_log('model {:s}, acc {:.2f}'.format(
       'DATA-SHARING', lr_acc), LogLevel.INFO)
  elif dataset_name == 'CONV2D-DIRECT-CPU':
    med_true_count = lr_true_count = total_data_count = 0
    
    dl = CpuDirectDataLoader(data_file)

    filters_list = [[['KH', '==', 3], ['SH', '==', 1]],
                    [['KH', '==', 3], ['SH', '==', 2]],
                    [['KH', '==', 5], ['SH', '==', 1]],
                    [['KH', '==', 5], ['SH', '==', 2]],
                    [['KH', '==', 7], ['SH', '==', 1]],
                    [['KH', '==', 7], ['SH', '==', 2]],
                    [['KH', '==', 9], ['SH', '==', 1]],
                    [['KH', '==', 9], ['SH', '==', 2]]]

    model_names = ['lr_t_flops_cpu_direct_k3x3s1',
                   'lr_t_flops_cpu_direct_k3x3s2',
                   'lr_t_flops_cpu_direct_k5x5s1',
                   'lr_t_flops_cpu_direct_k5x5s2',
                   'lr_t_flops_cpu_direct_k7x7s1',
                   'lr_t_flops_cpu_direct_k7x7s2',
                   'lr_t_flops_cpu_direct_k9x9s1',
                   'lr_t_flops_cpu_direct_k9x9s2']

    if unrolling_aware:
      filters_list = [[['KH', '==', 3], ['SH', '==', 1], ['OC', '<', 2]],
                      [['KH', '==', 3], ['SH', '==', 1], ['OC', '>=', 2]],
                      [['KH', '==', 3], ['SH', '==', 2], ['OC', '<', 1]],
                      [['KH', '==', 3], ['SH', '==', 2], ['OC', '>', 1]],
                      [['KH', '==', 5], ['SH', '==', 1], ['OC', '<', 4]],
                      [['KH', '==', 5], ['SH', '==', 1], ['OC', '>=', 4]],
                      [['KH', '==', 7], ['SH', '==', 1], ['OC', '<', 4]],
                      [['KH', '==', 7], ['SH', '==', 1], ['OC', '>=', 4]],
                      [['KH', '==', 7], ['SH', '==', 2], ['OC', '<', 4]],
                      [['KH', '==', 7], ['SH', '==', 2], ['OC', '>=', 4]],
                      [['KH', '==', 9], ['SH', '==', 1], ['OC', '<', 4]],
                      [['KH', '==', 9], ['SH', '==', 1], ['OC', '>=', 4]]]
    
    for filters, model_name in zip(filters_list, model_names):
      dl.reset_filters()
      #dl.add_filter('COMP', '>', 1)
      dl.add_filter('SD_COMP', '<', 5)
      dl.add_filters(filters)
      dl.split_data(frac=frac)
      data_count = dl.get_data_count()
      if data_count == 0:
        continue
      #LogUtil.print_log('Total data count: {}'.format(data_count), LogLevel.INFO)
      #total_data_count = total_data_count + data_count
      set_name='train'
      y = dl.get_target(set_name=set_name)
      total_train_data_count = \
          total_train_data_count + dl.get_data_count(set_name=set_name)
      
      #dl.debug_print()

      #acc, med_clf = median_fit(y, show_fig=show_fig)
      #med_true_count = med_true_count + int(data_count * acc)

      x_names = ['IH', 'IW', 'IC', 'OC']
      x = dl.get_features(x_names, set_name=set_name)
      acc, clf, means, stds = lr_fit(x, y, x_names, 't',
                                     lr_type=lr_train_type,
                                     x_scale=do_x_scale, y_scale=do_y_scale,
                                     show_coef=show_coef, show_fig=show_fig)
      #lr_true_count = lr_true_count + int(data_count * acc)

      set_name = 'test'
      data_count = dl.get_data_count(set_name=set_name)
      total_data_count = total_data_count + data_count
      y = dl.get_target(set_name=set_name)
      
      #acc = median_test(med_clf, y)
      #med_true_count = med_true_count + int(data_count * acc)

      x = dl.get_features(x_names, set_name=set_name)
      acc = lr_test(clf, x, y,
                    x_scale=do_x_scale, y_scale=do_y_scale,
                    means=means, stds=stds)
      lr_true_count = lr_true_count + int(data_count * acc / 100.0)

      if output_path is not None:
        json_text = lr_model_to_json(clf, lr_train_type, means, stds)
        LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
        save_model(output_path, model_name + '.json', json_text)

    #print_acc_result('Conv2D CPU Direct',
    #                 med_true_count, lr_true_count,
    #                 total_train_data_count, total_data_count)

    lr_acc = lr_true_count / total_data_count * 100.0

    #LogUtil.print_log('==== TOTAL MODEL INFO (CPU DIRECT) =====')
    #LogUtil.print_log('model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
    #    'DIRECT-CPU', total_data_count, 0, lr_acc), LogLevel.INFO)
    LogUtil.print_log('model {:s}, acc {:.2f}'.format(
        'DIRECT-CPU', lr_acc), LogLevel.INFO)
  elif dataset_name == 'GEMV-CPU':
    dl = CpuGemvDataLoader(data_file)

    dl.reset_filters()
    dl.add_filter('SD_COMP', '<', 5)
    dl.add_filter('COMP', '>', 0.1)
    dl.split_data(frac=frac)
    #dl.debug_print()
    
    set_name = 'train'
    y = dl.get_target(set_name=set_name)
    
    #acc, med_clf = median_fit(y, show_fig=show_fig)
    
    x_names = ['M', 'K']
    x = dl.get_features(x_names, set_name=set_name)
    acc, lr_clf, means, stds = lr_fit(x, y,
                                      lr_type=lr_train_type,
                                      x_scale=do_x_scale, y_scale=do_y_scale,
                                      show_coef=show_coef, show_fig=show_fig)
    
    set_name = 'test'
    data_count = dl.get_data_count(set_name=set_name)
    total_data_count = total_data_count + data_count

    y = dl.get_target(set_name=set_name)
    
    #acc = median_test(med_clf, y)
    #med_true_count = med_true_count + int(data_count * acc)

    x = dl.get_features(x_names, set_name=set_name)
    acc = lr_test(lr_clf, x, y,
                  x_scale=do_x_scale, y_scale=do_y_scale,
                  means=means, stds=stds)
    lr_true_count = lr_true_count + int(data_count * acc / 100.0)

    if output_path is not None:
      model_name = 'lr_t_comp_gemv_cpu'
      json_text = lr_model_to_json(lr_clf, lr_train_type, means, stds)
      LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
      save_model(output_path, model_name + '.json', json_text)
    
    lr_acc = lr_true_count / total_data_count * 100.0

    #LogUtil.print_log('==== TOTAL MODEL INFO (CPU GEMV) =====')
    #LogUtil.print_log('model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
    #    'GEMV-CPU', total_data_count, 0, lr_acc), LogLevel.INFO)
    LogUtil.print_log('model {:s}, acc {:.2f}'.format(
        'GEMV-CPU', lr_acc), LogLevel.INFO)
  elif dataset_name.startswith('GEMM') and dataset_name.endswith('CPU'):
    if dataset_name == 'GEMM-PACK-RB-CPU' or dataset_name == 'GEMM-TOTAL-CPU':
      filters_list = [[['PACK_RB', '>', 1]]]

      dl = CpuGemmPackRbDataLoader(data_file)

      for filters in filters_list:
        dl.reset_filters()
        dl.add_filter('SD_COMP', '<', 5)
        dl.add_filters(filters)
        dl.split_data(frac=frac)
        
        set_name = 'train'
        y = dl.get_target(set_name=set_name)

        #dl.debug_print()
        
        #pm10acc, med_pack_clf = median_fit(y, show_fig=show_fig)

        x_names = ['K', 'N']
        x = dl.get_features(x_names, set_name=set_name)
        pm10acc, lr_pack_clf, means, stds = lr_fit(x, y, x_names, 't',
                                                   lr_type=lr_train_type,
                                                   x_scale=do_x_scale,
                                                   y_scale=do_y_scale,
                                                   show_coef=show_coef,
                                                   show_fig=show_fig)

        set_name = 'test'
        data_count = dl.get_data_count(set_name=set_name)
        y = dl.get_target(set_name=set_name)
        x = dl.get_features(x_names, set_name=set_name)
        acc = lr_test(lr_pack_clf, x, y,
                      x_scale=do_x_scale, y_scale=do_y_scale,
                      means=means, stds=stds)

        if output_path is not None:
          model_name = 'lr_t_pack_rb_gemm_cpu'
          json_text = lr_model_to_json(lr_pack_clf, lr_train_type, means, stds)
          LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
          save_model(output_path, model_name + '.json', json_text)
    if dataset_name == 'GEMM-CPU' or dataset_name == 'GEMM-TOTAL-CPU':
      dl = CpuGemmDataLoader(data_file)

      dl.reset_filters()
      dl.add_filter('SD_COMP', '<', 5)
      dl.split_data(frac=frac)
      data_count = dl.get_data_count()
      #LogUtil.print_log('Total data count: {}'.format(data_count), LogLevel.INFO)
      
      set_name='train'
      y = dl.get_target(set_name=set_name)

      total_train_data_count = \
          total_train_data_count + dl.get_data_count(set_name=set_name)
    
      #dl.debug_print()
      
      #acc, med_comp_clf = median_fit(y, show_fig=show_fig)
      #med_true_count = med_true_count + int(data_count * acc)

      x_names = ['M', 'K', 'N']
      x = dl.get_features(x_names, set_name=set_name)
      acc, lr_comp_clf, means, stds = lr_fit(x, y, x_names, 't',
                                             lr_type=lr_train_type,
                                             x_scale=do_x_scale,
                                             y_scale=do_y_scale,
                                             show_coef=show_coef,
                                             show_fig=show_fig)
      #lr_true_count = lr_true_count + int(data_count * acc)

      set_name = 'test'
      data_count = dl.get_data_count(set_name=set_name)
      total_data_count = total_data_count + data_count
      y = dl.get_target(set_name=set_name)
      x = dl.get_features(x_names, set_name=set_name)
      acc = lr_test(lr_comp_clf, x, y,
                    x_scale=do_x_scale, y_scale=do_y_scale,
                    means=means, stds=stds)

      if output_path is not None:
        model_name = 'lr_t_comp_gemm_cpu'
        json_text = lr_model_to_json(lr_comp_clf, lr_train_type, means, stds)
        LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
        save_model(output_path, model_name + '.json', json_text)
    
    if dataset_name == 'GEMM-TOTAL-CPU':
      filters_list = [[['PACK_RB', '>', 1]]]
      dl = CpuGemmTotalDataLoader(data_file)
      for filters in filters_list:
        dl.reset_filters()
        dl.add_filter('SD_COMP', '<', 5)
        dl.add_filters(filters)
        dl.split_data(frac=frac)

        set_name = 'test'
        y = dl.get_target(set_name=set_name)
        #print(np.shape(y))

        total_data_count = dl.get_data_count(set_name=set_name)

        pack_blks = ReshapeToOneDim(dl.get_features(['PACK_RB_BLKS'], set_name=set_name))
        comp_blks = ReshapeToOneDim(dl.get_features(['COMP_BLKS'], set_name=set_name))
        
        pack_x = dl.get_features(['K', 'N'], set_name=set_name)
        comp_x = dl.get_features(['M', 'K', 'N'], set_name=set_name)
        #print(np.shape(pack_x))
        #print(np.shape(comp_x))

        #dl.debug_print()

        #pack_y = med_pack_clf.predict(pack_x)
        #comp_y = med_comp_clf.predict(comp_x)
        #print(np.shape(pack_blks))
        #print(np.shape(pack_y))
        #print(np.shape(comp_blks))
        #print(np.shape(comp_y))
        #print(np.shape(y))
        #total_y = pack_y * pack_blks + comp_y * comp_blks
        #PrintTestTrue(total_y, y)
        #med_acc = PM10ACC(total_y, y) * 100
        med_acc = 0

        if do_x_scale:
          pack_x = scale(pack_x)
        pack_y = lr_pack_clf.predict(pack_x)
        #PrintTestTrue(pack_y, dl.get_features('T_PACK_RB', set_name=set_name))
        if do_x_scale:
          comp_x = scale(comp_x)
        comp_y = lr_comp_clf.predict(comp_x)
        #PrintTestTrue(comp_y, dl.get_features('T_COMP', set_name=set_name))
        total_y = pack_y * pack_blks + comp_y * comp_blks
        #PrintTestTrue(total_y, y)
        lr_acc = PM10ACC(total_y, y) * 100

        #LogUtil.print_log('==== TOTAL MODEL INFO (GEMM) =====')
        #LogUtil.print_log('model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
        #    'GEMM-CPU', np.size(y), med_acc, lr_acc), LogLevel.INFO)
        LogUtil.print_log('model {:s}, acc {:.2f}'.format(
            'GEMM-CPU', lr_acc), LogLevel.INFO)

    #print_acc_result('Conv2D CPU GEMM',
    #                 med_true_count, lr_true_count,
    #                 total_train_data_count, total_data_count)
  elif dataset_name.startswith('WINOGRAD') and dataset_name.endswith('CPU'):
    tile_aware = False
    med_true_count = lr_true_count = total_data_count = 0
    if dataset_name == 'WINOGRAD-TR-IN-CPU' or \
       dataset_name == 'WINOGRAD-TOTAL-CPU':
      dl = CpuWinogradTrInDataLoader(data_file)

      if tile_aware:
        filters_list = [[['OT', '==', 2]], [['OT', '==', 6]]]
      else:
        filters_list = [[]]

      for filters in filters_list:
        dl.reset_filters()
        dl.add_filter('SD_TR_IN', '<', 5)
        #dl.add_filter('TR_IN', '>', 1)
        dl.add_filters(filters)

        dl.split_data(frac=frac)

        data_count = dl.get_data_count()
        #total_data_count = total_data_count + data_count
        
        set_name='train'
        y = dl.get_target(set_name=set_name)
        
        #dl.debug_print()

        #acc, ti_med_clf = median_fit(y, show_fig=show_fig)
        #med_true_count = med_true_count + int(data_count * acc)

        x_names = ['PIH', 'PIW', 'OT']
        x = dl.get_features(x_names, set_name=set_name)
        acc, ti_lr_clf, means, stds = lr_fit(x, y,
                                             lr_type=lr_train_type,
                                             x_scale=do_x_scale,
                                             y_scale=do_y_scale,
                                             show_coef=show_coef,
                                             show_fig=show_fig)
        #lr_true_count = lr_true_count + int(data_count * acc)

        set_name = 'test'
        data_count = dl.get_data_count(set_name=set_name)
        total_data_count = total_data_count + data_count
        y = dl.get_target(set_name=set_name)
        x = dl.get_features(x_names, set_name=set_name)
        #acc = median_test(ti_med_clf, y)
        #med_true_count = med_true_count + int(data_count * acc)
        acc = lr_test(ti_lr_clf, x, y,
                      x_scale=do_x_scale, y_scale=do_y_scale,
                      means=means, stds=stds)
        lr_true_count = lr_true_count + int(data_count * acc)
        
        if output_path is not None:
          model_name = 'lr_t_tr_in_winograd_cpu'
          json_text = lr_model_to_json(ti_lr_clf, lr_train_type, means, stds)
          LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
          save_model(output_path, model_name + '.json', json_text)

      #print_acc_result('Conv2D CPU Winograd TI',
      #                 med_true_count, lr_true_count,
      #                 total_train_data_count, total_data_count)
    
    med_true_count = lr_true_count = total_data_count = 0
    if dataset_name == 'WINOGRAD-TR-OUT-CPU' or \
       dataset_name == 'WINOGRAD-TOTAL-CPU':
      dl = CpuWinogradTrOutDataLoader(data_file)

      tile_aware = False
      if tile_aware:
        tile_filters_list = [[['OT', '==', 2]], [['OT', '==', 6]]]
      else:
        tile_filters_list = [[['OT', '>', 0]]]
      filters_list = [[['OC', '!=', 16], ['OC', '!=', 32], ['OC', '!=', 64],
                       ['OC', '!=', 128], ['OC', '!=', 256], ['OC', '!=', 512]]]

      for filters in filters_list:
        for tile_filters in tile_filters_list:
          dl.reset_filters()
          dl.add_filter('SD_TR_OUT', '<', 5)
          #dl.add_filter('TR_OUT', '>', 1)
          dl.add_filters(filters)
          dl.add_filters(tile_filters)
          
          dl.split_data(frac=frac)

          data_count = dl.get_data_count()
          #total_data_count = total_data_count + data_count
          
          set_name='train'
          y = dl.get_target(set_name=set_name)
          if np.size(y) == 0:
            continue
          
          #dl.debug_print()

          #acc, to_med_clf = median_fit(y, show_fig=show_fig)
          #med_true_count = med_true_count + int(data_count * acc)

          x_names = ['POH', 'POW', 'OT']
          x = dl.get_features(x_names, set_name=set_name)
          acc, to_lr_clf, means, stds = lr_fit(x, y,
                                               lr_type=lr_train_type,
                                               x_scale=do_x_scale,
                                               y_scale=do_y_scale,
                                               show_coef=False,
                                               show_fig=show_fig)
          #lr_true_count = lr_true_count + int(data_count * acc)

          set_name = 'test'
          data_count = dl.get_data_count(set_name=set_name)
          total_data_count = total_data_count + data_count
          y = dl.get_target(set_name=set_name)
          x = dl.get_features(x_names, set_name=set_name)
          acc = lr_test(to_lr_clf, x, y,
                        x_scale=do_x_scale, y_scale=do_y_scale,
                        means=means, stds=stds)
          lr_true_count = lr_true_count + int(data_count * acc)

          if output_path is not None:
            model_name = 'lr_t_tr_out_winograd_cpu'
            json_text = lr_model_to_json(to_lr_clf, lr_train_type, means, stds)
            LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
            save_model(output_path, model_name + '.json', json_text)

      #print_acc_result('Conv2D CPU Winograd TO',
      #                 med_true_count, lr_true_count,
      #                 total_train_data_count, total_data_count)
    
    med_true_count = lr_true_count = total_data_count = 0
    if dataset_name == 'WINOGRAD-GEMM-CPU' or \
       dataset_name == 'WINOGRAD-TOTAL-CPU':
      dl = CpuWinogradGemmDataLoader(data_file)

      tile_aware = False
      if tile_aware:
        tile_filters_list = [[['OT', '==', 2]], [['OT', '==', 6]]]
      else:
        tile_filters_list = [[['OT', '>=', 0]]]
      filters_list = [[['N', '>', 16], ['M', '>', 64]]]

      for filters in filters_list:
        for tile_filters in tile_filters_list:
          dl.reset_filters()
          dl.add_filter('SD_COMP', '<', 5)
          dl.add_filters(filters)
          dl.add_filters(tile_filters)
          
          dl.split_data(frac=frac)

          data_count = dl.get_data_count()
          #total_data_count = total_data_count + data_count
          
          set_name='train'
          y = dl.get_target(set_name=set_name)

          total_train_data_count = \
              total_train_data_count + dl.get_data_count(set_name=set_name)
        
          #dl.debug_print()
          
          #acc, gemm_med_clf = median_fit(y, show_fig=show_fig)
          #med_true_count = med_true_count + int(data_count * acc)

          x_names = ['M', 'K', 'N']
          x = dl.get_features(x_names, set_name=set_name)
          acc, gemm_lr_clf, means, stds = lr_fit(x, y,
                                                 lr_type=lr_train_type,
                                                 x_scale=do_x_scale,
                                                 y_scale=do_y_scale,
                                                 show_coef=False,
                                                 show_fig=show_fig)
          #lr_true_count = lr_true_count + int(data_count * acc)

          set_name = 'test'
          data_count = dl.get_data_count(set_name=set_name)
          total_data_count = total_data_count + data_count
          y = dl.get_target(set_name=set_name)
          x = dl.get_features(x_names, set_name=set_name)
          acc = lr_test(gemm_lr_clf, x, y,
                        x_scale=do_x_scale, y_scale=do_y_scale,
                        means=means, stds=stds)
          lr_true_count = lr_true_count + int(data_count * acc)

          if output_path is not None:
            model_name = 'lr_t_gemm_winograd_cpu'
            json_text = lr_model_to_json(gemm_lr_clf, lr_train_type, means, stds)
            LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
            save_model(output_path, model_name + '.json', json_text)

      #print_acc_result('Conv2D CPU Winograd GEMM',
      #                 med_true_count, lr_true_count,
      #                 total_train_data_count, total_data_count)

    if dataset_name == 'WINOGRAD-TOTAL-CPU':
      dl = CpuWinogradTotalDataLoader(data_file)
      dl.add_filter('SD_TR_IN', '<', 5)
      dl.add_filter('SD_TR_OUT', '<', 5)
      dl.add_filter('SD_COMP', '<', 5)
      #dl.add_filter('M', '>', 64)
      #dl.add_filter('N', '>', 16)
      dl.add_filter('OC', '!=', 32)
      dl.add_filter('OC', '!=', 64)
      dl.add_filter('OC', '!=', 128)
      dl.add_filter('OC', '!=', 256)
      dl.add_filter('OC', '!=', 512)

      dl.split_data(frac=frac)

      #dl.debug_print()

      set_name='test'
      ti_x = dl.get_features(['PIH', 'PIW', 'OT'], set_name=set_name)
      to_x = dl.get_features(['POH', 'POW', 'OT'], set_name=set_name)
      gemm_x = dl.get_features(['M', 'K', 'N'], set_name=set_name)

      ti_y = dl.get_target('T_TR_IN', set_name=set_name)
      #med_acc = PM10ACC(ti_med_clf.predict(ti_x), ti_y) * 100
      #med_acc  = 0
      #lr_acc = PM10ACC(ti_lr_clf.predict(ti_x), ti_y) * 100
      #LogUtil.print_log('PM10ACC: model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
      #    'TR-IN', np.size(ti_y), med_acc, lr_acc))

      #to_y = dl.get_target('T_TR_OUT', set_name=set_name)
      #med_acc = PM10ACC(to_med_clf.predict(to_x), to_y) * 100
      #med_acc  = 0
      #lr_acc = PM10ACC(to_lr_clf.predict(to_x), to_y) * 100
      #LogUtil.print_log('PM10ACC: model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
      #    'TR-OUT', np.size(to_y), med_acc, lr_acc))

      #gemm_y = dl.get_target('T_COMP', set_name=set_name)
      #med_acc = PM10ACC(gemm_med_clf.predict(gemm_x), gemm_y) * 100
      #med_acc  = 0
      #lr_acc = PM10ACC(gemm_lr_clf.predict(gemm_x), gemm_y) * 100
      #LogUtil.print_log('PM10ACC: model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
      #    'GEMM', np.size(gemm_y), med_acc, lr_acc))

      ti_blks = ReshapeToOneDim(dl.get_target('TR_IN_BLKS', set_name=set_name))
      to_blks = ReshapeToOneDim(dl.get_target('TR_OUT_BLKS', set_name=set_name))
      comp_blks = ReshapeToOneDim(dl.get_target('COMP_BLKS', set_name=set_name))

      y = dl.get_target(set_name=set_name)
      
      #total_y = ti_med_clf.predict(ti_x) * ti_blks + \
      #          to_med_clf.predict(to_x) * to_blks + \
      #          gemm_med_clf.predict(gemm_x) * comp_blks
      #PrintTestTrue(total_y, y)
      #med_acc = PM10ACC(total_y, y) * 100
      med_acc = 0

      if do_x_scale:
        ti_x = scale(ti_x)
        to_x = scale(to_x)
        gemm_x = scale(gemm_x)

      total_y = ti_lr_clf.predict(ti_x) * ti_blks + \
                to_lr_clf.predict(to_x) * to_blks + \
                gemm_lr_clf.predict(gemm_x) * comp_blks
      #PrintTestTrue(total_y, y)
      lr_acc = PM10ACC(total_y, y) * 100

      #LogUtil.print_log('==== TOTAL MODEL INFO (WINOGRAD) =====')
      #LogUtil.print_log('model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
      #    'WINOGRAD-CPU', np.size(y), med_acc, lr_acc), LogLevel.INFO)
      LogUtil.print_log('model {:s}, acc {:.2f}'.format(
          'WINOGRAD-CPU', lr_acc), LogLevel.INFO)
  elif dataset_name == 'POOLING-CPU' or \
       dataset_name == 'POOLING-GPU':
    dl = PoolingDataLoader(data_file)
    dl.add_filter('SD_COST', '<', 5)
    dl.add_filter('COST', '>', 0.5)
    dl.split_data(frac=frac, random_state=61)
    #dl.debug_print()

    x_names = ['PARAMS']

    set_name = 'train'
    y = dl.get_target(set_name=set_name)
    x = dl.get_features(x_names, set_name=set_name)
    acc, clf, means, stds = lr_fit(x, y, x_names, 't',
                                   lr_type=lr_train_type,
                                   x_scale=do_x_scale,
                                   y_scale=do_y_scale,
                                   show_coef=show_coef, show_fig=show_fig)

    set_name = 'test'
    data_count = dl.get_data_count(set_name=set_name)
    total_data_count = total_data_count + data_count
    y = dl.get_target(set_name=set_name)
    
    x = dl.get_features(x_names, set_name=set_name)
    acc = lr_test(clf, x, y,
                  x_scale=do_x_scale, y_scale=do_y_scale,
                  means=means, stds=stds)
    lr_acc = acc

    if output_path is not None:
      if dataset_name == 'POOLING-CPU':
        model_name = 'lr_t_params_pooling_cpu'
      elif dataset_name == 'POOLING-GPU':
        model_name = 'lr_t_params_pooling_gpu'
      else:
        raise ValueError('Unsupported dataset name: ' + dataset_name)
      
      json_text = lr_model_to_json(clf, lr_train_type, means, stds)
      LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
      save_model(output_path, model_name + '.json', json_text)
    
    if dataset_name == 'POOLING-CPU':
      model_name = 'DIRECT-CPU'
    elif dataset_name == 'POOLING-GPU':
      model_name = 'DIRECT-GPU'
    #LogUtil.print_log('model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
    #    model_name, total_data_count, 0, lr_acc), LogLevel.INFO)
    LogUtil.print_log('model {:s}, acc {:.2f}'.format(
        model_name, lr_acc), LogLevel.INFO)
  elif dataset_name == 'CONV2D-DIRECT-GPU':
    dl = Conv2dGpuDirectDataLoader(data_file)
    filters_list = [[['KS', '==', 1], ['S', '==', 1], ['COMP', '>', 1]],
                    [['KS', '==', 1], ['S', '==', 2]],
                    [['KS', '==', 3], ['S', '==', 1], ['ICB', '>', 16], ['OCB', '>', 16]],
                    [['KS', '==', 3], ['S', '==', 2], ['OCB', '>', 2], ['COMP', '>', 1]],
                    [['KS', '==', 5], ['S', '==', 1]],
                    [['KS', '==', 5], ['S', '==', 2]],
                    [['KS', '==', 7], ['S', '==', 1], ['COMP', '>', 4]],
                    [['KS', '==', 7], ['S', '==', 2]],
                    [['KS', '==', 9], ['S', '==', 1]],
                    [['KS', '==', 9], ['S', '==', 2]]]

    model_names = ['lr_t_warp_gpu_direct_k1x1s1',
                   'lr_t_warp_gpu_direct_k1x1s2',
                   'lr_t_warp_gpu_direct_k3x3s1',
                   'lr_t_warp_gpu_direct_k3x3s2',
                   'lr_t_warp_gpu_direct_k5x5s1',
                   'lr_t_warp_gpu_direct_k5x5s2',
                   'lr_t_warp_gpu_direct_k7x7s1',
                   'lr_t_warp_gpu_direct_k7x7s2',
                   'lr_t_warp_gpu_direct_k9x9s1',
                   'lr_t_warp_gpu_direct_k9x9s2']
    
    for filters, model_name in zip(filters_list, model_names):
      dl.reset_filters()
      #dl.add_filter('CNN', '==', 'RETINAFACE')
      #dl.add_filter('PDIM', '==', 4)
      #dl.add_filter('OH', '==', 26)
      dl.add_filter('SD_COMP', '<', 5)
      dl.add_filter('COMP', '>', 1)
      dl.add_filters(filters)
      dl.split_data(frac=frac)
      data_count = dl.get_data_count()
      if data_count <= 2:
        continue
      #total_data_count = total_data_count + data_count
      set_name='train'
      y = dl.get_target(set_name=set_name)

      total_train_data_count = \
          total_train_data_count + dl.get_data_count(set_name=set_name)
    
      #dl.debug_print()
    
      #acc, med_clf = median_fit(y, show_fig=show_fig)
      #med_true_count = med_true_count + int(data_count * acc)

      x_names = ['OH', 'OWB', 'ICB', 'OCB']
      x = dl.get_features(x_names, set_name=set_name)
      acc, clf, means, stds = lr_fit(x, y, x_names, 't',
                                     lr_type=lr_train_type,
                                     x_scale=do_x_scale,
                                     y_scale=do_y_scale,
                                     show_coef=False, show_fig=show_fig)
      #lr_true_count = lr_true_count + int(data_count * acc)

      set_name = 'test'
      data_count = dl.get_data_count(set_name=set_name)
      total_data_count = total_data_count + data_count
      y = dl.get_target(set_name=set_name)
      
      #acc = median_test(med_clf, y)
      #med_true_count = med_true_count + int(data_count * acc)

      x = dl.get_features(x_names, set_name=set_name)
      acc = lr_test(clf, x, y,
                    x_scale=do_x_scale, y_scale=do_y_scale,
                    means=means, stds=stds)
      lr_true_count = lr_true_count + int(data_count * acc / 100.0)

      if output_path is not None:
        json_text = lr_model_to_json(clf, lr_train_type, means, stds)
        LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
        save_model(output_path, model_name + '.json', json_text)

    #print_acc_result('Covn2D GPU Direct',
    #                 med_true_count, lr_true_count,
    #                 total_train_data_count, total_data_count)

    lr_acc = lr_true_count / total_data_count * 100.0

    #LogUtil.print_log('==== TOTAL MODEL INFO (GPU DIRECT) =====')
    #LogUtil.print_log('model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
    #    'DIRECT-GPU', total_data_count, 0, lr_acc), LogLevel.INFO)
    LogUtil.print_log('model {:s}, acc {:.2f}'.format(
        'DIRECT-GPU', lr_acc), LogLevel.INFO)
  elif dataset_name == 'FC-DIRECT-GPU':
    dl = FullyConnectedGpuDirectDataLoader(data_file)
    dl.add_filter('SD_COMP', '<', 2)
    #dl.add_filter('COMP', '>', 1)
    #dl.add_filter('T_WARP_BLK', '<', 5e-06)
    dl.split_data(frac=frac)
    #dl.debug_print()
    
    #data_count = dl.get_data_count()
    #total_data_count = total_data_count + data_count
    
    set_name = 'train'
    y = dl.get_target(set_name=set_name)
    
    #print('y {}'.format(y))

    #acc, _ = median_fit(y, show_fig=show_fig)
    #med_true_count = med_true_count + int(data_count * acc)

    x_names = ['IH', 'IW', 'ICB', 'OCB']
    x = dl.get_features(x_names, set_name=set_name)
    acc, clf, means, stds = lr_fit(x, y,
                                   lr_type=lr_train_type,
                                   x_scale=do_x_scale,
                                   y_scale=do_y_scale,
                                   show_coef=show_coef, show_fig=show_fig)
    #lr_true_count = lr_true_count + int(data_count * acc)

    set_name = 'test'
    data_count = dl.get_data_count(set_name=set_name)
    total_data_count = total_data_count + data_count
    y = dl.get_target(set_name=set_name)

    x = dl.get_features(x_names, set_name=set_name)
    acc = lr_test(clf, x, y,
                  x_scale=do_x_scale, y_scale=do_y_scale,
                  means=means, stds=stds)
    lr_acc = acc

    if output_path is not None:
      model_name = 'lr_t_warp_fc_direct_gpu'
      json_text = lr_model_to_json(clf, lr_train_type, means, stds)
      LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
      save_model(output_path, model_name + '.json', json_text)

    #LogUtil.print_log('==== TOTAL MODEL INFO (GPU FC) =====')
    #LogUtil.print_log('model {:s}, count {:d}, median {:.2f}, lr {:.2f}'.format(
    #    'DIRECT-GPU', total_data_count, 0, lr_acc), LogLevel.INFO)
    LogUtil.print_log('model {:s}, acc {:.2f}'.format(
        'DIRECT-GPU', lr_acc), LogLevel.INFO)
  elif dataset_name == 'DECONV2D-DIRECT-CPU':
    dl = Deconv2dCpuDirectDataLoader(data_file)
    dl.add_filter('SD_COMP', '<', 5)
    data_count = dl.get_data_count()
    total_data_count = total_data_count + data_count
    y = dl.get_target()
  
    #dl.debug_print()
  
    show_fig = False

    #acc, _ = median_fit(y, show_fig=show_fig)
    #med_true_count = med_true_count + int(data_count * acc)

    x = dl.get_features()
    acc, _, means, stds = lr_fit(x, y, lr_type=lr_train_type,
                                 x_scale=do_x_scale, y_scale=do_y_scale,
                                 show_coef=False, show_fig=show_fig)
    lr_true_count = lr_true_count + int(data_count * acc)
  elif dataset_name == 'DECONV2D-GPU-DIRECT':
    dl = Deconv2dGpuDirectDataLoader(data_file)
    dl.add_filter('SD_COMP', '<', 5)
    data_count = dl.get_data_count()
    total_data_count = total_data_count + data_count
    y = dl.get_target()
  
    #dl.debug_print()
  
    show_fig = False

    #acc, _ = median_fit(y, show_fig=show_fig)
    #med_true_count = med_true_count + int(data_count * acc)

    x = dl.get_features()
    acc, _, means, stds = lr_fit(x, y, lr_type=lr_train_type,
                                 x_scale=do_x_scale, y_scale=do_y_scale,
                                 show_coef=False, show_fig=show_fig)
    lr_true_count = lr_true_count + int(data_count * acc)
  else:
    raise ValueError('Unsupported dataset name: ' + dataset_name)

if __name__ == '__main__':
  args = parse_args()
  data = args.data
  dataset = args.dataset
  frac = args.frac
  unrolling_aware = args.unrolling_aware
  show_coef = args.show_coef
  show_fig = args.show_fig
  output_path = args.output_path

  fit_test(data, dataset, frac, unrolling_aware, show_coef, show_fig, output_path)
