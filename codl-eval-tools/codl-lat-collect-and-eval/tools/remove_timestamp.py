
import os
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, required=True, help='Data dir')
  args = parser.parse_args()
  return args

def split_dirpath_and_filename(filepath):
  items = filepath.split('/')
  out_dirpath = ''
  for i in range(len(items) - 1):
    out_dirpath = os.path.join(out_dirpath, items[i])
  filename = items[-1]
  return out_dirpath, filename

def split_filename_and_ext(filename):
  items = filename.split('.')
  return items[0], items[1]

def remove_timestamp_for_filename(filename):
  filename, ext = split_filename_and_ext(filename)
  #print('filename {}, ext {}'.format(filename, ext))
  names = filename.split('_')
  #print('names {}'.format(names))
  timestamp = names[-1]
  if not timestamp.isnumeric():
    # No timestamp, keep file name.
    return filename + '.' + ext
  out_filename = ''
  for i in range(len(names) - 1):
    out_filename = out_filename + names[i] + '_'
  #print('out_filename {}'.format(out_filename))
  return out_filename[:-1] + '.' + ext
  
def remove_timestamp_for_filepath(filepath):
  dirpath, filename = split_dirpath_and_filename(filepath)
  #print('dirpath {}, filename {}'.format(dirpath, filename))
  filename = remove_timestamp_for_filename(filename)
  return os.path.join(dirpath, filename)

def remove_timestamp_for_dir(data_dir):
  for filepath, dirnames, filenames in os.walk(data_dir):
    for filename in filenames:
      in_filepath = os.path.join(filepath, filename)
      #print(filepath)
      out_filepath = remove_timestamp_for_filepath(in_filepath)
      #print('src {}, dst {}'.format(in_filepath, out_filepath))
      os.rename(in_filepath, out_filepath)

if __name__ == '__main__':
  args = parse_args()
  remove_timestamp_for_dir(args.data_dir)
