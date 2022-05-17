"""

实验设计:
- 生成多种feature的数据文件
- 每个feature数量都传输一次并运行
- 统计时间

"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import subprocess
import re


def repeat(x, total_rows):
    if total_rows < x.shape[0]:
        return x.iloc[:total_rows, :]
    
    n = x.shape[0]
    num_segment = total_rows // n
    repeat_times = [num_segment + 1 if num_segment * n + i < total_rows else num_segment for i in range(n)]
    data = x.loc[x.index.repeat(repeat_times)]
    return data


def analysis(log_path, num_samples):
    with open(log_path, "r") as f:
        s = "".join(f.readlines())
    
    pattern = r"Num features: (\d+), Time cost: ([0-9.]*)"
    match = re.findall(pattern, s)
    time_cost = np.array([[int(i[0]), float(i[1])] for i in match])

    for i in range(time_cost[-1][0].astype(int).item()):
        data = time_cost[time_cost[:, 0] == i + 1]
        n = len(data)
        num_repeat = n // len(num_samples)
        print(num_repeat)
        for idx, sample in enumerate(num_samples):
            times = data[idx*num_repeat:(idx+1)*num_repeat, 1].mean()
            print(f"{i+1}-{sample}: {times*1000:.5f}ms")


def main():
    feature_names = ["OH", "OWB", "ICB", "OCB"]
    label_names = ["T_WARP_ICB"]

    dst_folder = "dataset"
    data_filepath = "dataset/gpu_direct.csv"
    data = pd.read_csv(data_filepath)

    num_samples = [100, 500, 1000, 5000, 10000]

    for i in range(len(feature_names)):
        features = feature_names[:i+1]
        # X_train, X_test, y_train, y_test = train_test_split(data.loc[:, features], data.loc[:, label_names], test_size=0.2)
        
        for num_sample in num_samples:
            print(features, num_sample)
            X_train, X_test, y_train, y_test = train_test_split(data.loc[:, features], data.loc[:, label_names], test_size=0.2)
            
            X_train = repeat(X_train, num_sample)
            X_test = repeat(X_test, num_sample)
            y_train = repeat(y_train, num_sample)
            y_test = repeat(y_test, num_sample)

            X_train.to_csv(f"{dst_folder}/train_x_{i}_{num_sample}.csv", index=False)
            X_test.to_csv(f"{dst_folder}/test_x_{i}_{num_sample}.csv", index=False)
            y_train.to_csv(f"{dst_folder}/train_y_{i}_{num_sample}.csv", index=False)
            y_test.to_csv(f"{dst_folder}/test_y_{i}_{num_sample}.csv", index=False)

            os.system(f"adb push {dst_folder}/train_x_{i}_{num_sample}.csv /data/local/tmp/bangwhe")
            os.system(f"adb push {dst_folder}/test_x_{i}_{num_sample}.csv /data/local/tmp/bangwhe")
            os.system(f"adb push {dst_folder}/train_y_{i}_{num_sample}.csv /data/local/tmp/bangwhe")
            os.system(f"adb push {dst_folder}/test_y_{i}_{num_sample}.csv /data/local/tmp/bangwhe")

    os.system("adb push android_run.sh /data/local/tmp/bangwhe")
    os.system("adb shell 'cd /data/local/tmp/bangwhe && sh android_run.sh'")
    os.system("adb pull /data/local/tmp/bangwhe/time.log .")


if __name__ == "__main__":
    # dst_folder = "dataset"
    # data_filepath = "dataset/gpu_direct.csv"
    # data = pd.read_csv(data_filepath)
    # print(data.shape)

    # data = repeat(data, 5000)
    # print(data.shape)
    # main()

    analysis("time.log", (100, 1000, 10000, 500, 5000))
    # analysis("third_party/tinyML/time.log", (100, 500, 1000, 5000, 10000))