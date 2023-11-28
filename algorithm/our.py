from collections import defaultdict
import numpy as np
from scipy.spatial import distance_matrix
from utils.prompt import construct_prompt
from utils.inference import inference
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math
from utils.misc import evaluate


def agg_probs(prob_reps, window=2):
    prob_reps = np.array(prob_reps)[:, -window:]
    T = len(prob_reps[0])
    # weighted average of prob_reps
    weights = np.array([1.0 / 2 ** (T - i) for i in range(T)])
    weights = weights / np.sum(weights)
    wt_prob_reps = prob_reps * weights[None,]
    wt_probs = np.sum(wt_prob_reps, axis=1)
    return prob_reps, wt_prob_reps, wt_probs


def split_candidates(prob_reps, probs, uncertain_indices):
    potential_pos = np.where(probs > 0.5)[0]
    potential_neg = np.where(probs <= 0.5)[0]
    print("potential_pos:")
    for idx in potential_pos:
        print(uncertain_indices[idx], prob_reps[idx])
    print("potential_neg:")
    for idx in potential_neg:
        print(uncertain_indices[idx], prob_reps[idx])
    return uncertain_indices[potential_pos], uncertain_indices[potential_neg]


def get_next_target(selected_labels, potential_pos, potential_neg):
    pos_num, neg_num = np.sum(selected_labels), len(selected_labels) - np.sum(
        selected_labels
    )
    if pos_num > neg_num:
        next_target = "neg_avg"
        if pos_num - neg_num >= 2 or len(potential_neg) <= 2:
            next_target = "neg_most_conf"
    elif pos_num < neg_num:
        next_target = "pos_avg"
        if neg_num - pos_num >= 2 or len(potential_pos) <= 2:
            next_target = "pos_most_conf"
    else:
        if len(potential_pos) >= len(potential_neg):
            next_target = "pos_avg"
        else:
            next_target = "neg_avg"
    return next_target


def select(
    historical_probs,
    prob_reps,
    conf_reps,
    uncertain_indices,
    selected_labels,
    selected_indices,
):
    """
    probs: [K, T]
    """
    prob_reps, wt_prob_reps, wt_probs = agg_probs(prob_reps)
    conf_reps, _, _ = agg_probs(conf_reps)
    potential_pos, potential_neg = split_candidates(
        prob_reps, wt_probs, uncertain_indices
    )
    next_target = get_next_target(selected_labels, potential_pos, potential_neg)
    print(f"next_target: {next_target}")

    if next_target == "pos_most_conf":
        if len(potential_pos) > 0:
            idx = np.argmax(wt_probs)
            return uncertain_indices[idx], next_target
        else:
            return None, next_target
    elif next_target == "neg_most_conf":
        if len(potential_neg) > 0:
            idx = np.argmin(wt_probs)
            return uncertain_indices[idx], next_target
        else:
            return None, next_target

    if next_target == "pos_avg":
        uncertain_indices = potential_pos
    elif next_target == "neg_avg":
        uncertain_indices = potential_neg

    historical_probs = np.array(historical_probs).T  # [N, T]
    historical_probs = np.clip(historical_probs, 0.35, 0.65)
    _, wt_prob_reps, _ = agg_probs(historical_probs)
    # calculate distance between selected indices and all the samples
    dist_selected = distance_matrix(
        wt_prob_reps[selected_indices], wt_prob_reps, p=1
    )  # [K, N]
    dist_min_cur = np.min(dist_selected, axis=0)  # [N]
    # print("selected reps:")
    # print(wt_prob_reps[selected_indices])
    # print("dist_min_cur:")
    # print(list(np.round(dist_min_cur, 3)))
    # calculate the distance between the uncertain indices and all the samples
    dist_uncertain = distance_matrix(
        wt_prob_reps[uncertain_indices], wt_prob_reps, p=1
    )  # [K2, N]
    # print(dist_uncertain.shape)
    # print(list(np.round(dist_uncertain, 3)))
    # calculate the improvement
    improvement = dist_min_cur[None,] - dist_uncertain  # [K2, N]
    improvement = np.clip(improvement, 0, None)
    print(list(np.sum(improvement > 0, axis=1)))
    improvement = np.sum(improvement, axis=1)  # [K2]
    print(list(np.round(improvement, 3)))
    idx = np.argmax(improvement)
    return uncertain_indices[idx], next_target


def fast_votek(prob_reps, uncertain_indices):
    sim_matrix = distance_matrix(prob_reps, prob_reps, p=1)

    # build the graph
    n = len(prob_reps)
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
    budget = int(max(100, np.ceil(len(labels) / (args.k * (args.k + 1) / 2.0))))
    # budget = int(max(100, np.ceil(len(labels) / args.k)))
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
        n = (budget - len(sample_indices)) // (group_num - i)
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


def MFL(historical_probs, uncertain_indices, selected_indices):
    historical_probs = np.array(historical_probs).T  # [N, T]
    T = len(historical_probs[0])
    weights = np.array([1.0 / 2 ** (T - i) for i in range(T)])
    weights = weights / np.sum(weights)
    prob_reps = historical_probs * weights[None,]
    # calculate distance between selected indices and all the samples
    dist_selected = distance_matrix(
        prob_reps[selected_indices], prob_reps, p=1
    )  # [K, N]
    dist_min_cur = np.min(dist_selected, axis=0)  # [N]
    # calculate the distance between the uncertain indices and all the samples
    dist_uncertain = distance_matrix(
        prob_reps[uncertain_indices], prob_reps, p=1
    )  # [K2, N]
    # calculate the improvement
    improvement = dist_min_cur[None,] - dist_uncertain  # [K2, N]
    improvement = np.clip(improvement, 0, None)
    improvement = np.sum(improvement, axis=1)  # [N]
    print(improvement)
    idx = np.argmax(improvement)
    return uncertain_indices[idx]


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
        prompts = construct_prompt(
            example_inputs, example_labels, example_embeddings, inputs, embeddings, args
        )
        print(prompts[0])
        print("---------------\n\n")
        _, probs = inference(model_name, model, tokenizer, prompts, args)
        precision, recall, f1 = evaluate(labels, np.array(probs) > 0.5)
        print(f"Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f}")
        probs = np.array(probs)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        conf = 1 + (probs * np.log(probs) + (1 - probs) * np.log(1 - probs)) / scaler
        conf[selected_indices] = 2

        cond1 = conf < certain_thr
        if np.sum(cond1) > 0:
            uncertain_indices = np.where(cond1)[0]  # not confident
            flag_type = "conf"
        else:
            conf_temp = conf.copy()
            if np.sum(example_labels) * 2 > len(example_labels):  # next: neg
                conf_temp[probs > 0.5] = 1 + conf_temp[probs > 0.5]
            elif np.sum(example_labels) * 2 < len(example_labels):
                conf_temp[probs <= 0.5] = 1 + conf_temp[probs <= 0.5]
            uncertain_indices = [np.argmin(conf_temp)]  # least confident
            flag_type = "select least conf from all confident"
        print(f"flag_type: {flag_type}")
        print(f"uncertain_indices ({len(uncertain_indices)}): {uncertain_indices}")
        # conf = np.clip(conf, 0, certain_thr)
        historical_confs.append(np.clip(conf, 0, certain_thr))  # [T, N]
        historical_probs.append(probs)  # [T, N]

        if flag_type == "conf":
            # calculate similarity between samples based on their predicted probs
            prob_reps = np.array(historical_probs)[:, uncertain_indices].T  # [K, T]
            conf_reps = np.array(historical_confs)[:, uncertain_indices].T  # [K, T]
            # idx = fast_votek(prob_reps, uncertain_indices)
            idx, next_target = select(
                historical_probs,
                prob_reps,
                conf_reps,
                uncertain_indices,
                example_labels,
                selected_indices,
            )
            print(f"just for check: idx={idx}, next_target={next_target}")
            if idx is None:
                if next_target == "pos_most_conf":
                    conf_temp = conf.copy()
                    if np.sum(probs > 0.5) > np.sum(example_labels):  # potential pos
                        conf_temp[probs <= 0.5] = 1 + conf_temp[probs <= 0.5]
                        idx = np.argmin(conf_temp)
                    else:
                        probs_temp = probs.copy()
                        probs_temp[selected_indices] = -1
                        idx = np.argmax(probs_temp)
                elif next_target == "neg_most_conf":
                    conf_temp = conf.copy()
                    if np.sum(probs <= 0.5) > len(example_labels) - np.sum(
                        example_labels
                    ):  # potential neg
                        conf_temp[probs > 0.5] = 1 + conf_temp[probs > 0.5]
                        idx = np.argmin(conf_temp)
                    else:
                        probs_temp = probs.copy()
                        probs_temp[selected_indices] = 1
                        idx = np.argmin(probs)
                else:
                    raise ValueError
            # idx = MFL(historical_probs, uncertain_indices, selected_indices)
        else:
            idx = uncertain_indices[0]
        probs_output = [f"{v:.4f}" for v in probs]

        print(
            f"index: {idx}, label: {labels[idx]}, pred: {probs[idx]:.2f}, conf: {conf[idx]:.2f}"
        )
        result = [v for v in zip(range(len(labels)), labels, probs_output)]
        print(result)
        print(f"selected_indices: {selected_indices}")
        print(f"****************\n\n")
        selected_indices.append(idx)
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
        # if most_conf but still not get expected labeled sample
        while 2 * np.sum(example_labels) <= len(example_labels) - 4:
            print("keep selecting pos...")
            scores_temp = np.array(scores.copy())
            scores_temp[selected_indices] = -1
            idx = np.argmax(scores_temp)
            selected_indices.append(idx)
            example_inputs.append(inputs[idx])
            example_labels.append(labels[idx])
            example_embeddings.append(embeddings[idx])
            print(f"selected_indices: {selected_indices}, labels: {example_labels}")
            if len(selected_indices) >= min(args.k, args.budget):
                break
        while 2 * np.sum(example_labels) >= len(example_labels) + 4:
            print("keep selecting neg...")
            scores_temp = np.array(scores.copy())
            scores_temp[selected_indices] = 1
            idx = np.argmin(scores_temp)
            selected_indices.append(idx)
            example_inputs.append(inputs[idx])
            example_labels.append(labels[idx])
            example_embeddings.append(embeddings[idx])
            print(f"selected_indices: {selected_indices}, labels: {example_labels}")
            if len(selected_indices) >= min(args.k, args.budget):
                break
    while len(selected_indices) < args.budget:
        break
        # TODO
    selected_indices = [candidate_indices[idx] for idx in selected_indices]
    return selected_indices
