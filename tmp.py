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

dataset = sys.argv[1]
if dataset in dataset_dict:
    filedir = f"../new_data/baseline/ER-Magellan"
    is_wdc = False
    train_file = "train.csv"
    if dataset == "AB":
        filedir = f"{filedir}/Textual"
    else:
        filedir = f"{filedir}/Structured"
else:
    filedir = f"../new_data/baseline/wdc"
    is_wdc = True
    train_file = "train.txt.small"

data_dir = filedir.replace("new_data/baseline", "data")
dataset = dataset_dict.get(dataset, dataset)
filename = f"{filedir}/{dataset}/max_entropy_bl_sampling_wm_llama2-70b.csv"
data_dir = f"{data_dir}/{dataset}"
df = pd.read_csv(filename)
df = df[df.budget == 100]
print(df.tail(3))
train_entry_pairs, train_labels = load_data(data_dir, train_file, is_wdc)

cnt = 0
last_label = None
for _, row in df.iterrows():
    row_num = row["index"]
    label = row["label"]
    if last_label is not None and label == last_label:
        cnt += 1
        print("Num:", cnt, "index:", row_num, "label:", label)
        print(train_entry_pairs[row_num].valsA)
        print(train_entry_pairs[row_num].valsB)
        print(train_labels[row_num])
        print("*******************\n")
    last_label = label.copy()
