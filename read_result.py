import os
import pandas as pd
from utils.misc import evaluate


result_dir = "results_1116"
datasets = os.listdir(result_dir)
for dataset in datasets:
    f1_list = []
    for llm in ["llama2-7b", "llama2-13b", "llama2-70b"]:
        for k in [6, 8, 10, 20]:
            filename = f"{result_dir}/{dataset}/our_{k}_{llm}.csv"
            if os.path.exists(filename) == False:
                f1_list.append("--")
                continue
            df = pd.read_csv(filename)
            pred = df["pred"].values
            label = df["label"].values
            precision, recall, f1 = evaluate(label, pred)
            f1_list.append(f"{f1:.2f}")
    print(f"{dataset} {' '.join(f1_list)}")
