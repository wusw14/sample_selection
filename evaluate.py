from utils.llm import init_model
from utils.io import load_data
from utils.prompt import construct_prompt
from utils.inference import inference
from utils.misc import evaluate
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
    parser.add_argument("--eval_size", type=int, default=20)
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--metric", type=str, default="f1")
    parser.add_argument("--sep_sample", action="store_true")
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


def load_selected_indices(args):
    # load selected indices
    data_dir = args.data_dir.replace("data", f"new_data/{args.version}")
    if args.selection_method in ["MFL", "fast_votek"]:
        input_file = os.path.join(data_dir, f"{args.selection_method}.csv")
    else:
        input_file = os.path.join(data_dir, f"{args.selection_method}_{args.lm}.csv")
    df_selected = pd.read_csv(input_file)
    if args.selection_method in ["votek", "adaicl"]:
        df_selected = df_selected[df_selected.k == args.k]
        selected_indices = df_selected["index"].tolist()
    else:
        selected_indices = df_selected["index"].tolist()[: args.budget]
    return selected_indices


def load_in_context_examples(args):
    selected_indices = load_selected_indices(args)
    train_entry_pairs, train_labels = load_data(
        args.data_dir, args.train_file, args.is_wdc
    )
    data_dir = args.data_dir.replace("data", "temp_data")
    embeddings = np.loadtxt(os.path.join(data_dir, "train_pair_emb.npy"))
    assert len(train_entry_pairs) == len(embeddings)

    example_inputs, example_labels, example_embeddings = [], [], []
    for index in selected_indices:
        example_inputs.append(train_entry_pairs[index])
        example_labels.append(train_labels[index])
        example_embeddings.append(embeddings[index])
    return example_inputs, example_labels, example_embeddings


def load_test(args):
    test_entry_pairs, test_labels = load_data(
        args.data_dir, args.test_file, args.is_wdc
    )
    data_dir = args.data_dir.replace("data", "temp_data")
    embeddings = np.loadtxt(os.path.join(data_dir, "test_pair_emb.npy"))
    assert len(test_entry_pairs) == len(embeddings)
    return test_entry_pairs, test_labels, embeddings


def initialization(args):
    args = parse_args()
    print(f"PID {os.getpid()}\n\n")
    print(args, "\n\n")
    np.random.seed(args.seed)

    example_inputs, example_labels, example_embeddings = load_in_context_examples(args)
    print(example_labels)

    test_entry_pairs, test_labels, test_embeddings = load_test(args)
    test_prompts = construct_prompt(
        example_inputs,
        example_labels,
        example_embeddings,
        test_entry_pairs,
        test_embeddings,
        args,
    )
    print(test_prompts[0])

    model_name, model, tokenizer = init_model(args)

    return model_name, model, tokenizer, test_prompts, test_labels


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    output_file = f"results/results_{args.version}/{args.dataset}/{args.selection_method}_{args.lm}.csv"
    if os.path.exists(output_file):
        print("File exists")
        exit()

    (model_name, model, tokenizer, test_prompts, test_labels) = initialization(args)
    start_time = time.time()
    preds, probs = inference(model_name, model, tokenizer, test_prompts, args)
    print(f"Time cost {time.time() - start_time}")

    precision, recall, f1 = evaluate(test_labels, preds)
    print(f"Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f}")

    df_result = pd.DataFrame({"label": test_labels, "pred": preds, "prob": probs})
    if os.path.exists(f"results/results_{args.version}/{args.dataset}") is False:
        os.makedirs(f"results/results_{args.version}/{args.dataset}")

    df_result.to_csv(output_file, index=False)
