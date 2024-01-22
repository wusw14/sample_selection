import os
import pandas as pd
from utils.misc import evaluate
import sys

method = "ideal"
result_dir = f"results/results_{sys.argv[1]}"
dataset_list = [
    "AG",
    "BR",
    "DA",
    "DS",
    "FZ",
    "IA",
    "WA",
    "AB",
    "cameras",
    "computers",
    "shoes",
    "watches",
]

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

for dataset in dataset_list:
    f1_list = []
    dataset = dataset_dict.get(dataset, dataset)
    for llm in ["llama2-7b", "llama2-13b", "llama2-70b"]:
        for k in [6, 8, 10]:
            filename = f"{result_dir}/{dataset}/{method}_{k}_{llm}.csv"
            if os.path.exists(filename) == False:
                f1_list.append("--")
                continue
            df = pd.read_csv(filename)
            pred = df["pred"].values
            label = df["label"].values
            precision, recall, f1 = evaluate(label, pred)
            f1_list.append(f"{f1:.2f}")
    print(f"{dataset} {' '.join(f1_list)}")
