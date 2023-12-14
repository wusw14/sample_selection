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
#     if memory > 0.05:
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

lm = args.lm
batch_size = 1

for dataset in dataset_list:
    for selection_method in [
        "MFL",
        "fast_votek",
        "min_entropy",
        "max_entropy",
        "cbs_maxIG",
        "votek",
        "adaicl",
        "our",
    ][-3:-1]:
        for budget in [6, 8, 10, 20, 50]:
            if (
                selection_method
                in [
                    "MFL",
                    "fast_votek",
                    "min_entropy",
                    "max_entropy",
                    "cbs_maxIG",
                    "our",
                ]
                and budget < 50
            ):
                continue
            if selection_method in ["MFL", "fast_votek"] and lm != "llama2-70b":
                continue
            if os.path.exists(f"logs/select_{args.version}/{dataset}") is False:
                os.makedirs(f"logs/select_{args.version}/{dataset}")
            if os.path.exists(
                f"logs/select_{args.version}/{dataset}/{selection_method}_{budget}_{lm}.log"
            ):
                continue
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpus} "
                f"python -u main.py --lm {lm} --gpus {gpus} --dataset {dataset} "
                f"--selection_method {selection_method} "
                f"--budget {budget} --batch_size {batch_size} "
                f"--version {args.version} --order o7 "
                f" >> logs/select_{args.version}/{dataset}/{selection_method}_{budget}_{lm}.log"
            )
            print(cmd)
            os.system(cmd)


for dataset in dataset_list:
    for selection_method in [
        "MFL",
        "fast_votek",
        "min_entropy",
        "max_entropy",
        "cbs_maxIG",
        "votek",
        "adaicl",
        "our",
    ][-3:-1]:
        for budget in [6, 8, 10, 20, 50]:
            if os.path.exists(f"logs/inference_{args.version}/{dataset}") is False:
                os.makedirs(f"logs/inference_{args.version}/{dataset}")
            if os.path.exists(
                f"logs/inference_{args.version}/{dataset}/{selection_method}_{budget}_{lm}.log"
            ):
                continue
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpus} "
                f"python -u evaluate.py --lm {lm} --gpus {gpus} --dataset {dataset} "
                f"--selection_method {selection_method} "
                f"--budget {budget} --batch_size {batch_size} "
                f"--version {args.version} --order o7"
                f" > logs/inference_{args.version}/{dataset}/{selection_method}_{budget}_{lm}.log"
            )
            print(cmd)
            os.system(cmd)
