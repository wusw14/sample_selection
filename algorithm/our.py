from collections import defaultdict
import numpy as np
from scipy.spatial import distance_matrix
from utils.prompt import construct_prompt
from utils.inference import inference
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math


def select(probs_reps, uncertain_indices):
    """
    probs: [K, T]
    """
    T = len(probs_reps[0])
    # weighted average of prob_reps
    weights = np.array([1.0 / 2 ** (T - i) for i in range(T)])
    weights = weights / np.sum(weights)
    probs = np.sum(np.array(probs_reps) * weights[None,], axis=1)
    print(probs)
    potential_pos = np.where(probs > 0.5)[0]
    potential_neg = np.where(probs <= 0.5)[0]
    print("potential_pos:", potential_pos)
    print("potential_neg:", potential_neg)
    if len(potential_pos) >= len(potential_neg):
        candidate_reps = np.array(probs_reps)[potential_pos]
        uncertain_indices = np.array(uncertain_indices)[potential_pos]
    else:
        candidate_reps = np.array(probs_reps)[potential_neg]
        uncertain_indices = np.array(uncertain_indices)[potential_neg]

    dist_matrix = distance_matrix(candidate_reps, candidate_reps, p=1)
    dist = np.sum(dist_matrix, axis=0)
    idx = np.argmin(dist)
    return uncertain_indices[idx]


def fast_votek(probs_reps, uncertain_indices):
    sim_matrix = distance_matrix(probs_reps, probs_reps, p=1)

    # build the graph
    n = len(probs_reps)
    k = math.ceil(n / 2)
    vote_stat = defaultdict(int)
    for i in range(n):
        cur_scores = sim_matrix[i]
        sorted_indices = np.argsort(cur_scores).tolist()[:k]
        for idx in sorted_indices:
            if idx != i:
                vote_stat[idx] += 1

    # select the samples
    votes = sorted(vote_stat.items(), key=lambda x: x[1], reverse=True)
    cur_selected_idx = votes[0][0]
    return uncertain_indices[int(cur_selected_idx)]


def cal_cosine_sim(args):
    data_dir = args.data_dir.replace("data", "temp_data")
    embeddingsA = np.loadtxt(f"{data_dir}/train_A_emb.npy")
    embeddingsB = np.loadtxt(f"{data_dir}/train_B_emb.npy")
    embeddingsA = embeddingsA / np.linalg.norm(embeddingsA, axis=1, keepdims=True)
    embeddingsB = embeddingsB / np.linalg.norm(embeddingsB, axis=1, keepdims=True)
    cosine_sim = np.sum(embeddingsA * embeddingsB, axis=1)
    return cosine_sim


def stratified_sampling(inputs, labels, embeddings, cosine_of_each_pair, args):
    # by cosine similarity and null values distribution
    missing_value_reps = []
    for entry_pair in inputs:
        valsA = entry_pair.valsA
        valsB = entry_pair.valsB
        missing_valsA = np.array(pd.isna(valsA), int)
        missing_valsB = np.array(pd.isna(valsB), int)
        missing_value_rep = missing_valsA + missing_valsB
        missing_value_rep = "".join([str(v) for v in missing_value_rep])
        missing_value_reps.append(int(missing_value_rep))

    df = pd.DataFrame(
        {
            "id": list(range(len(inputs))),
            "score": cosine_of_each_pair,
            "missing_value_rep": missing_value_reps,
        }
    )
    df["group"] = pd.cut(df["score"], bins=5, labels=[0, 1, 2, 3, 4])
    group_id, missing_value_rep, _ = zip(
        *sorted(
            df.groupby(["group", "missing_value_rep"]).size().reset_index().values,
            key=lambda x: x[-1],
        )
    )
    group_num = len(group_id)
    sample_indices = []
    for i in range(group_num):
        df_sub = df[
            (df.group == group_id[i]) & (df.missing_value_rep == missing_value_rep[i])
        ]
        n = (100 - len(sample_indices)) // (group_num - i)
        if n >= len(df_sub):
            sample_indices += df_sub["id"].tolist()
        else:
            sample_indices += df_sub.sample(n=n, random_state=args.seed)["id"].tolist()
    sample_inputs = [inputs[idx] for idx in sample_indices]
    sample_labels = [labels[idx] for idx in sample_indices]
    sample_embeddings = [embeddings[idx] for idx in sample_indices]
    scores = [cosine_of_each_pair[idx] for idx in sample_indices]
    print(
        f"pos/neg in candidates: {sum(sample_labels)}/{len(sample_labels)-sum(sample_labels)}"
    )
    return sample_inputs, sample_labels, sample_embeddings, sample_indices, scores


def our(model_name, model, tokenizer, inputs, labels, embeddings, args):
    # sample by cosine similarity
    cosine_of_each_pair = cal_cosine_sim(args)
    inputs, labels, embeddings, candidate_indices, scores = stratified_sampling(
        inputs, labels, embeddings, cosine_of_each_pair, args
    )
    print(f"candidates: {len(inputs)}")

    # warm up with the sample with the highest and lowest cosine similarity score
    example_inputs, example_labels, example_embeddings = [], [], []
    selected_indices = [np.argmax(scores), np.argmin(scores)]
    for idx in selected_indices:
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])

    # iterative sampling by confidence
    # conf_decay = 25
    scaler = -np.log(0.5)
    certain_thr = 1 + (0.65 * np.log(0.65) + (1 - 0.65) * np.log(1 - 0.65)) / scaler
    historical_confs, historical_probs = [], []

    while len(selected_indices) < min(args.k, args.budget):
        print(f"****************")
        print(f"iteration {len(selected_indices) + 1}")
        # probs with two orders
        prompts = construct_prompt(
            example_inputs, example_labels, example_embeddings, inputs, embeddings, args
        )
        print(prompts[0])
        print("---------------\n\n")
        _, probs = inference(model_name, model, tokenizer, prompts, args)
        probs = np.array(probs)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        conf = 1 + (probs * np.log(probs) + (1 - probs) * np.log(1 - probs)) / scaler

        cond1 = conf < certain_thr
        cond2 = (
            conf < np.max(historical_confs, axis=0)
            if len(historical_confs) > 0
            else (conf < certain_thr)
        )
        conf[selected_indices] = 1
        if np.sum(cond1 * cond2) > 0:
            uncertain_indices = np.where(cond1 * cond2)[0]
        elif np.sum(cond1) > 0:
            uncertain_indices = np.where(cond1)[0]  # not confident
        elif np.sum(cond2) > 0:
            uncertain_indices = np.where(cond2)[0]  # less confident than before
        else:
            uncertain_indices = [np.argmin(conf)]  # least confident
        print(f"uncertain_indices ({len(uncertain_indices)}): {uncertain_indices}")
        conf = np.clip(conf, 0, certain_thr)
        historical_confs.append(conf)  # [T, N]
        historical_probs.append(probs)  # [T, N]

        if len(uncertain_indices) > 2:
            # calculate similarity between samples based on their predicted probs
            probs_reps = np.array(historical_probs)[:, uncertain_indices].T  # [K, T]
            # idx = fast_votek(probs_reps, uncertain_indices)
            idx = select(probs_reps, uncertain_indices)
        elif len(uncertain_indices) == 2:
            diff1 = (
                np.max(np.array(historical_confs)[:, uncertain_indices[0]])
                - conf[uncertain_indices[0]]
            )
            diff2 = (
                np.max(np.array(historical_confs)[:, uncertain_indices[0]])
                - conf[uncertain_indices[1]]
            )
            if diff1 > diff2:
                idx = uncertain_indices[0]
            else:
                idx = uncertain_indices[1]
        else:
            idx = uncertain_indices[0]
        probs_output = [f"{v:.4f}" for v in probs]

        print(
            f"index: {idx}, label: {labels[idx]}, pred: {probs[idx]:.2f}, conf: {conf[idx]:.2f}"
        )
        result = [v for v in zip(range(len(labels)), labels, probs_output)]
        print(result)
        print(f"****************\n\n")
        selected_indices.append(idx)
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
    while len(selected_indices) < args.budget:
        break
        # TODO
    selected_indices = [candidate_indices[idx] for idx in selected_indices]
    return selected_indices
