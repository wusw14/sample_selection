import os
import sys
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import torch

import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

from utils.io import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="1")
    parser.add_argument("--dataset", type=str, default="FZ")
    parser.add_argument("--path", type=str, default="../data")
    parser.add_argument("--dirty", action="store_true")
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
    args.dataset = dataset_dict.get(args.dataset, args.dataset)
    args.data_dir = os.path.join(args.path, args.dataset)
    args.is_wdc = dataset_dir == "wdc"
    args.train_file = "train.txt.small" if args.is_wdc else "train.csv"
    args.test_file = "test.txt" if args.is_wdc else "test.csv"
    return args


def get_emb(text):
    EmbModel = SentenceTransformer("stsb-roberta-base")
    embedding = EmbModel.encode(text, batch_size=512)
    return embedding


def serialize(entry_pairs, target_cols=None):
    question = "Are Entry A and Entry B the same?"
    # [cols, valsA, valsB]
    entryA_text, entryB_text, entry_pair_text = [], [], []
    for cols, valsA, valsB in tqdm(entry_pairs):
        serialized_text = []
        for col, val in zip(cols, valsA):
            if target_cols is not None and col not in target_cols:
                continue
            serialized_text.append(f"{col}: {val}")
        entryA_text.append(", ".join(serialized_text))
        serialized_text = []
        for col, val in zip(cols, valsB):
            if target_cols is not None and col not in target_cols:
                continue
            serialized_text.append(f"{col}: {val}")
        entryB_text.append(", ".join(serialized_text))
        entry_pair_text.append(
            f"Entry A is {entryA_text[-1]}. Entry B is {entryB_text[-1]}. {question}"
        )
    print(entry_pair_text[0])
    return entryA_text, entryB_text, entry_pair_text


def preprocess(args, filename, cols=None):
    entry_pairs, _ = load_data(args.data_dir, filename, args.is_wdc, filter_cols=True)
    entryA_text, entryB_text, entry_pair_text = serialize(entry_pairs, cols)
    entryA_emb = get_emb(entryA_text)
    entryB_emb = get_emb(entryB_text)
    entry_diff_emb = np.abs(entryA_emb - entryB_emb)
    entry_pair_emb = get_emb(entry_pair_text)
    entry_pair_emb = entry_pair_emb / np.linalg.norm(
        entry_pair_emb, axis=1, keepdims=True
    )
    filename = filename.split(".")[0]
    data_dir = args.data_dir.replace("data", "temp_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.savetxt(os.path.join(data_dir, filename + "_A_emb.npy"), entryA_emb)
    print(f"Saved {filename}_A_emb.npy, shape = {entryA_emb.shape}")
    np.savetxt(os.path.join(data_dir, filename + "_B_emb.npy"), entryB_emb)
    print(f"Saved {filename}_B_emb.npy, shape = {entryB_emb.shape}")
    np.savetxt(os.path.join(data_dir, filename + "_pair_emb.npy"), entry_pair_emb)
    print(f"Saved {filename}_pair_emb.npy, shape = {entry_pair_emb.shape}")
    np.savetxt(os.path.join(data_dir, filename + "_diff_emb.npy"), entry_diff_emb)
    print(f"Saved {filename}_diff_emb.npy, shape = {entry_diff_emb.shape}")


def filter_cols(args):
    df_A = pd.read_csv(os.path.join(args.data_dir, "tableA.csv"), index_col=0)
    df_B = pd.read_csv(os.path.join(args.data_dir, "tableB.csv"), index_col=0)
    df_train = pd.read_csv(os.path.join(args.data_dir, args.train_file))

    cols_remained = []
    for col in df_A.columns:
        cnt = 0
        for lid, rlid, _ in df_train.values:
            if pd.isna(df_A.loc[lid, col]) or pd.isna(df_B.loc[rlid, col]):
                cnt += 1
        if cnt < 0.2 * len(df_train):
            cols_remained.append(col)
    return cols_remained


if __name__ == "__main__":
    args = parse_args()
    print(f"PID {os.getpid()}\n\n")
    print(args, "\n\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # cols = filter_cols(args)
    cols = None

    preprocess(args, filename=args.train_file, cols=cols)
    preprocess(args, filename=args.test_file, cols=cols)
