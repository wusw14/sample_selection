from algorithm.MFL import MFL
from algorithm.votek import votek, fast_votek
from algorithm.adaicl import adaicl
from algorithm.entropy import max_entropy, min_entropy, cbs_maxIG, max_entropy_bl
from algorithm.our import (
    select_by_cosine_sim,
    ideal,
)
from utils.llm import init_model
from utils.io import load_data
import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
import torch

import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, default="gpt2")
    parser.add_argument("--order", type=str, default="o5")
    parser.add_argument("--filedir", type=str, default="results0514")
    parser.add_argument("--entry_type", type=str, default="Product")
    parser.add_argument("--question_format", type=str, default="q2")
    parser.add_argument("--serialization", type=str, default="s6")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--torch_half", action="store_true")
    parser.add_argument("--CoT", action="store_true")
    parser.add_argument("--desp", type=str, default="t3")
    parser.add_argument("--output_format", type=str, default="o1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--selection_method", type=str, default="votek")
    parser.add_argument("--sim_func", type=str, default="cosine")
    parser.add_argument("--select_ind", action="store_true")
    parser.add_argument("--dirty", action="store_true")
    parser.add_argument("--gpus", type=str, default="1")
    parser.add_argument("--dataset", type=str, default="FZ")
    parser.add_argument("--path", type=str, default="../data")
    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pos_num", type=int, default=3)
    parser.add_argument("--neg_num", type=int, default=3)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--eval_size", type=int, default=20)
    parser.add_argument("--sample_size", type=int, default=20)
    args = parser.parse_args()

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
    if args.dataset in ["cameras", "computers", "shoes", "watches"]:
        dataset_dir = "wdc"
    elif args.dataset == "AB":
        dataset_dir = "ER-Magellan/Textual"
    elif args.dirty:
        dataset_dir = "ER-Magellan/Dirty"
    else:
        dataset_dir = "ER-Magellan/Structured"
    args.path = os.path.join(args.path, dataset_dir)
    if args.dataset in ["DA", "DS"]:
        args.entry_type = "Paper"
    elif args.dataset == "FZ":
        args.entry_type = "Resturant"
    elif args.dataset == "IA":
        args.entry_type = "Song"
    else:
        args.entry_type = "Product"
    args.dataset = dataset_dict.get(args.dataset, args.dataset)
    args.data_dir = os.path.join(args.path, args.dataset)
    args.is_wdc = dataset_dir == "wdc"
    args.train_file = "train.txt.small" if args.is_wdc else "train.csv"
    args.test_file = "test.txt" if args.is_wdc else "test.csv"
    return args


def initialization(args):
    train_entry_pairs, train_labels = load_data(
        args.data_dir, args.train_file, args.is_wdc
    )
    if len(train_entry_pairs) <= args.sample_size - 100:
        print(f"same as the previous result")
        exit()
    data_dir = args.data_dir.replace("data", "temp_data")
    embeddings = np.loadtxt(os.path.join(data_dir, "train_pair_emb.npy"))
    assert len(train_entry_pairs) == len(embeddings)

    if args.selection_method in ["MFL", "fast_votek", "cosine_sim"]:
        model_name, model, tokenizer = None, None, None
    else:
        model_name, model, tokenizer = init_model(args)
    return model_name, model, tokenizer, train_entry_pairs, train_labels, embeddings


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    print(f"PID {os.getpid()}\n\n")
    print(args, "\n\n")
    np.random.seed(args.seed)
    if args.beam_size == 1:
        exit()

    data_dir = args.data_dir.replace("data", f"new_data/{args.version}")
    if os.path.exists(data_dir) is False:
        os.makedirs(data_dir)
    if args.selection_method in ["MFL", "fast_votek"]:
        output_file = os.path.join(data_dir, f"{args.selection_method}.csv")
    else:
        output_file = os.path.join(data_dir, f"{args.selection_method}_{args.lm}.csv")
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        if len(df[df.budget == args.budget]) > 0:
            print("already got the selected indices")
            exit()

    (model_name, model, tokenizer, entry_pairs, labels, embeddings) = initialization(
        args
    )

    start_time = time.time()
    if args.selection_method == "votek":
        selected_indices = votek(
            model_name, model, tokenizer, entry_pairs, labels, embeddings, args
        )
    elif args.selection_method == "adaicl":
        selected_indices = adaicl(
            model_name, model, tokenizer, entry_pairs, labels, embeddings, args
        )
    elif args.selection_method == "cosine_sim":
        selected_indices = select_by_cosine_sim(
            model_name, model, tokenizer, entry_pairs, labels, embeddings, args
        )
    elif args.selection_method == "ideal":
        selected_indices = ideal(
            model_name, model, tokenizer, entry_pairs, labels, embeddings, args
        )
    elif args.selection_method == "max_entropy":
        selected_indices = max_entropy(
            model_name, model, tokenizer, entry_pairs, labels, embeddings, args
        )
    elif "max_entropy_bl" in args.selection_method:
        selected_indices = max_entropy_bl(
            model_name, model, tokenizer, entry_pairs, labels, embeddings, args
        )
    elif args.selection_method == "min_entropy":
        selected_indices = min_entropy(
            model_name, model, tokenizer, entry_pairs, labels, embeddings, args
        )
    elif args.selection_method == "cbs_maxIG":
        selected_indices = cbs_maxIG(
            model_name, model, tokenizer, entry_pairs, labels, embeddings, args
        )
    elif args.selection_method == "fast_votek":
        selected_indices, _, _ = fast_votek(embeddings, args.budget, args)
    elif args.selection_method == "MFL":
        selected_indices = MFL(embeddings, args)
    else:
        raise NotImplementedError
    print(f"Total Time for Selection: {time.time()-start_time:.2f}s")
    selected_labels = [labels[idx] for idx in selected_indices]

    df_new = pd.DataFrame(
        {
            "k": [args.k] * len(selected_indices),
            "index": selected_indices,
            "label": selected_labels,
        }
    )
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        df = pd.concat([df, df_new])
    else:
        df = df_new
    df.to_csv(output_file, index=False)
