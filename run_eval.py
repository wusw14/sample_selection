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
parser.add_argument("--p", type=float, default=1.0)
args = parser.parse_args()
gpus = args.gpus
p = args.p
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
    if memory > 0.5:
        cnt = 0
        time.sleep(300)
        iter_num += 1
    else:
        cnt += 1
        print(f"cnt: {cnt}, memory: {memory}")
        time.sleep(np.random.randint(10, 20))
    if iter_num % 12 == 0:
        print(f"{iter_num//12} hours passed")
    if cnt > 10:
        break

batch_size = 1

for beam_size in [1, 2, 3, 4, 6, 8]:
    for p in [0.7, 0.8, 0.9, 1.0][-1:]:
        args.version = f"base_IG_ablation_BS{beam_size}"
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
                "our_pairwise",
                "our_progressive",
                "ideal",
            ][-1:]:
                for budget in [6, 8, 10, 20, 30, 40, 50]:
                    if selection_method not in ["votek", "adaicl"] and budget < 50:
                        continue
                    for lm in ["llama2-7b", "llama2-13b", "llama2-70b"]:
                        if lm != args.lm:
                            continue
                        if (
                            selection_method in ["MFL", "fast_votek"]
                            and lm != "llama2-70b"
                        ):
                            continue
                        if (
                            os.path.exists(f"logs/select_{args.version}/{dataset}")
                            is False
                        ):
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
                            f"--serialization s6 "
                            f"--beam_size {beam_size} "
                            f"--p {p} "
                            f" >> logs/select_{args.version}/{dataset}/{selection_method}_{budget}_{lm}.log"
                        )
                        print(cmd)
                        os.system(cmd)
                        # time.sleep(60)

        for budget in [10, 8, 6][:1]:
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
                    "our_pairwise",
                    "our_progressive",
                    "ideal",
                ][-1:]:
                    # for budget in [6, 8, 10]:
                    for lm in ["llama2-7b", "llama2-13b", "llama2-70b"]:
                        if lm != args.lm:
                            continue
                        if (
                            os.path.exists(f"logs/inference_{args.version}/{dataset}")
                            is False
                        ):
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
                            f"--version {args.version} --order o7 "
                            f"--serialization s6 "
                            f" >> logs/inference_{args.version}/{dataset}/{selection_method}_{budget}_{lm}.log"
                        )
                        print(cmd)
                        os.system(cmd)
                        # time.sleep(60)

if args.lm == "llama2-70b":
    cmd = f"CUDA_VISIBLE_DEVICES={gpus} python -u inference.py"
    print(cmd)
    os.system(cmd)
# cmd = f"CUDA_VISIBLE_DEVICES={gpus} python -u inference.py"
# print(cmd)
# os.system(cmd)
