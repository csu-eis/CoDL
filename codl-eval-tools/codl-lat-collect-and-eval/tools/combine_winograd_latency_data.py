
import os
import argparse
import pandas as pd

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, required=True, help='Data dir')
  args = parser.parse_args()
  return args

def add_data_by_col(dst_df, dst_col_idx, src_df, col_name):
  if dst_col_idx == -1:
    dst_col_idx = dst_df.shape[1]
  dst_df.insert(dst_col_idx, col_name, src_df[col_name])

def replace_data_by_col(dst_df, src_df, col_name):
  dst_df[col_name] = src_df[col_name]

def save_to_csv(df, filepath):
  #print(df)
  df.to_csv(filepath, index=False)

def combine_winograd_latency_data(data_dir):
  wino_data_csv = os.path.join(data_dir, 't_conv2d_cpu_winograd.csv')
  wino_gemm_data_csv = os.path.join(data_dir, 't_conv2d_cpu_winograd_gemm.csv')

  wino_df = pd.read_csv(wino_data_csv)
  wino_gemm_df = pd.read_csv(wino_gemm_data_csv)

  add_data_by_col(wino_df, 4, wino_gemm_df, 'COMP')
  add_data_by_col(wino_df, 8, wino_gemm_df, 'T_COMP')
  replace_data_by_col(wino_df, wino_gemm_df, 'SD_COMP')
  add_data_by_col(wino_df, -1, wino_gemm_df, 'M')
  add_data_by_col(wino_df, -1, wino_gemm_df, 'K')
  add_data_by_col(wino_df, -1, wino_gemm_df, 'N')
  add_data_by_col(wino_df, -1, wino_gemm_df, 'MB')
  add_data_by_col(wino_df, -1, wino_gemm_df, 'KB')
  add_data_by_col(wino_df, -1, wino_gemm_df, 'NB')
  add_data_by_col(wino_df, -1, wino_gemm_df, 'MAX_MB_THR')
  add_data_by_col(wino_df, -1, wino_gemm_df, 'MAX_NB_THR')

  save_to_csv(wino_df, os.path.join(data_dir, 't_conv2d_cpu_winograd_combined.csv'))
  #print('Combine winograd data successfully.')

if __name__ == '__main__':
  args = parse_args()
  combine_winograd_latency_data(args.data_dir)
