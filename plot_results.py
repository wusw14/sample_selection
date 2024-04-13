from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.mixture import GaussianMixture


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


method = "ideal"
# dataset = sys.argv[1]
llm_size = sys.argv[1]
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

for dataset in dataset_list:
    dataset = dataset_dict.get(dataset, dataset)

    fig = plt.figure(figsize=(25, 5))
    for loc, B in enumerate([20, 40, 60, 80, 100]):
        file_dir = f"results/results_0411_V3_f1_B{B}"
        filename = f"{file_dir}/{dataset}/{method}_llama2-{llm_size}b.csv"
        if os.path.exists(filename) == False:
            continue
        df = pd.read_csv(filename)
        prob = df["prob"].values
        pred = df["pred"].values
        label = df["label"].values
        ax = fig.add_subplot(1, 5, loc + 1)
        ax.hist(prob[label == 0], bins=20, alpha=0.5, range=(0, 1))
        ax.hist(prob[label == 1], bins=20, alpha=0.5, range=(0, 1))
        ax.axvline(0.5, color="red")
        ax.axvline(0.6, color="gray", linestyle="--")
        ax.axvline(0.4, color="gray", linestyle="--")
        ylim = len(df) // 10
        ax.set_ylim(0, ylim)
        ax.set_title(f"B = {B}, F1-score = {f1_score(label, pred)*100:.2f}")
    fig.suptitle(f"Prediction of test data on {dataset} by LLAMA2-{llm_size}b")
    plt.tight_layout()
    fig_dir = f"results_plots/results_0411_V3_f1/{dataset}"
    if os.path.exists(fig_dir) == False:
        os.makedirs(fig_dir)
    plt.savefig(f"{fig_dir}/llama2-{llm_size}b.png")
