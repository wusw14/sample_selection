import argparse
import os
import pandas as pd
import time
import pynvml
import numpy as np


def check_log(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    flag = False
    for line in lines:
        if "Precision" in line:
            flag = True
            break
    return flag


pynvml.nvmlInit()
gpuDeriveInfo = pynvml.nvmlSystemGetDriverVersion()

parser = argparse.ArgumentParser()
parser.add_argument("gpus", type=str)
parser.add_argument("--dataset", type=str, nargs="+", default=["AG", "AB"])
parser.add_argument("--dirty", action="store_true")
parser.add_argument("--version", type=str, default="1117")
args = parser.parse_args()
gpus = args.gpus
print(args)

dataset_list = args.dataset

handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpus.split(",")[0]))
memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(memoryInfo.used / memoryInfo.total)
cnt, iter_num = 0, 0
while True:
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpus.split(",")[0]))
    memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory = memoryInfo.used / memoryInfo.total
    if memory > 0.1:
        cnt = 0
        time.sleep(300)
        iter_num += 1
    else:
        cnt += 1
        print(f"cnt: {cnt}, memory: {memory}")
        time.sleep(np.random.randint(10, 20))
    if iter_num % 12 == 0:
        print(f"{iter_num//12} hours passed")
    if cnt > 30:
        break

batch_size = 2

cmd = f"CUDA_VISIBLE_DEVICES={gpus} python -u inference.py"
print(cmd)
os.system(cmd)
