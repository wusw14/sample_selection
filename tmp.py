import sys
import os
import pandas as pd
from utils.misc import evaluate
import numpy as np
from utils.io import load_data
import warnings

warnings.filterwarnings("ignore")

dataset_dict = {
    "AG": "Amazon-Google",
    "BR": "BeerAdvo-RateBeer",
    "DA": "DBLP-ACM",
    "DS": "DBLP-Scholar",
    "FZ": "Fodors-Zagats",
    "IA": "iTunes-Amazon",
    "WA": "Walmart-Amazon",
    "AB": "Abt-Buy",
}

result_dir = f"results_baseline"
for method in ["MFL", "fast_votek"]:
    result_list = []
    for dataset in ["FZ", "WA", "AB", "cameras"]:
        dataset = dataset_dict.get(dataset, dataset)
        filename = f"{result_dir}/{dataset}/{method}_10_llama2-70b.csv"
        df = pd.read_csv(filename)
        pred = df["pred"].values
        label = df["label"].values
        _, _, f1 = evaluate(label, pred)
        result_list.append(f"{f1:.2f}")
    print(f"{method} & {' & '.join(result_list)} \\\\")


# dataset = sys.argv[1]
# if dataset in dataset_dict:
#     filedir = f"../new_data/baseline/ER-Magellan"
#     is_wdc = False
#     train_file = "train.csv"
#     if dataset == "AB":
#         filedir = f"{filedir}/Textual"
#     else:
#         filedir = f"{filedir}/Structured"
# else:
#     filedir = f"../new_data/baseline/wdc"
#     is_wdc = True
#     train_file = "train.txt.small"

# data_dir = filedir.replace("new_data/baseline", "data")
# dataset = dataset_dict.get(dataset, dataset)
# # filename = f"{filedir}/{dataset}/max_entropy_bl_sampling_wm_llama2-70b.csv"
# data_dir = f"{data_dir}/{dataset}"
# filename = f"{data_dir}/train.csv"
# df = pd.read_csv(filename)
# label = df["label"].values
# freq_dict = {k: 0 for k in range(0, 6)}
# for i in range(1000):
#     sampled_labels = np.random.choice(label, size=10, replace=False)
#     if np.sum(sampled_labels) >= 5:
#         freq_dict[5] += 1
#     else:
#         freq_dict[np.sum(sampled_labels)] += 1

# freq_list = [f"{(freq_dict[k]/1000):.3f}" for k in range(0, 6)]
# print(f"{dataset} & {' & '.join(freq_list)} \\\\")
# selected_1 = df[df.budget == 50]["label"].values[:10]
# selected_2 = df[df.budget == 100]["label"].values[:10]
# print(selected_1)
# print(selected_2)
# for i in range(10):
#     if selected_1[i] != selected_2[i]:
#         print(dataset, "update")
#         exit(0)
# print(dataset, "no update")
# print(df.tail(3))
# train_entry_pairs, train_labels = load_data(data_dir, train_file, is_wdc)

# cnt = 0
# last_label = None
# for _, row in df.iterrows():
#     row_num = row["index"]
#     label = row["label"]
#     # if last_label is not None and label == last_label:
#     cnt += 1
#     print("Num:", cnt, "index:", row_num, "label:", label)
#     print(train_entry_pairs[row_num].valsA)
#     print(train_entry_pairs[row_num].valsB)
#     print(train_labels[row_num])
#     print("*******************\n")
#     last_label = label.copy()
