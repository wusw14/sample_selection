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
parser.add_argument("lm", type=str)
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
# while True:
#     handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpus.split(",")[0]))
#     memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     memory = memoryInfo.used / memoryInfo.total
#     if memory > 0.1:
#         cnt = 0
#         time.sleep(300)
#         iter_num += 1
#     else:
#         cnt += 1
#         print(f"cnt: {cnt}, memory: {memory}")
#         time.sleep(np.random.randint(10, 20))
#     if iter_num % 12 == 0:
#         print(f"{iter_num//12} hours passed")
#     if cnt > 30:
#         break

batch_size = 1
sample_size = 2000
budget = 50
# args.version = "base_debug"
mode = "select"
serialization = "s6"
k = 10

for dataset in dataset_list:
    for selection_method in [
        "cosine_sim",
        "min_entropy",
        "max_entropy",
        "votek",
        "adaicl",
    ]:
        for lm in ["llama2-7b", "llama2-13b", "llama2-70b"]:
            if selection_method == "cosine_sim":
                if lm != "llama2-7b":
                    continue
            if lm != args.lm:
                continue
            if os.path.exists(f"logs/{mode}_{args.version}/{dataset}") is False:
                os.makedirs(f"logs/{mode}_{args.version}/{dataset}")
            if os.path.exists(
                f"logs/{mode}_{args.version}/{dataset}/{selection_method}_{lm}.log"
            ):
                continue
            if mode == "select":
                run_file = "main.py"
            else:
                run_file = "evaluate.py"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpus} "
                f"python -u {run_file} --lm {lm} --gpus {gpus} --dataset {dataset} "
                f"--selection_method {selection_method} "
                f"--budget {budget} --k {k} --batch_size {batch_size} "
                f"--version {args.version} --order o7 "
                f"--serialization {serialization} "
                f"--eval_size 100 --sample_size {sample_size}"
                f" > logs/{mode}_{args.version}/{dataset}/{selection_method}_{lm}.log"
            )
            print(cmd)
            os.system(cmd)
