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
        print(uncertain_indices[idx], np.round(prob_reps[idx], 3))
    print("potential_neg:")
    for idx in potential_neg:
        print(uncertain_indices[idx], np.round(prob_reps[idx], 3))
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
        # p = np.random.rand()
        # if p < 0.5 and len(potential_pos) > 0 or len(potential_neg) == 0:
        #     next_target = "pos_avg"
        # else:
        #     next_target = "neg_avg"
        if len(potential_pos) >= len(potential_neg):
            next_target = "pos_avg"
        else:
            next_target = "neg_avg"
    return next_target


def select(
    historical_probs,
    prob_reps,
    uncertain_indices,
    selected_labels,
    selected_indices,
):
    """
    probs: [K, T]
    """
    # prob_reps, wt_prob_reps, wt_probs = agg_probs(prob_reps)
    wt_probs = prob_reps[:, -1]
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
    # historical_probs = np.clip(historical_probs, 0.35, 0.65)
    wt_prob_reps = historical_probs
    # _, wt_prob_reps, wt_prob = agg_probs(historical_probs)
    if "pos" in next_target:
        sub_wt_prob_reps = wt_prob_reps[wt_prob_reps[:, -1] > 0.5]
    elif "neg" in next_target:
        sub_wt_prob_reps = wt_prob_reps[wt_prob_reps[:, -1] <= 0.5]
    else:
        sub_wt_prob_reps = wt_prob_reps
    # calculate distance between selected indices and all the samples
    if len(selected_indices) == 0:
        dist_min_cur = np.ones(len(sub_wt_prob_reps)) * 100
    else:
        dist_selected = distance_matrix(
            wt_prob_reps[selected_indices], sub_wt_prob_reps, p=1
        )  # [K, N]
        dist_min_cur = np.min(dist_selected, axis=0)  # [N]
    # calculate the distance between the uncertain indices and all the samples
    dist_uncertain = distance_matrix(
        wt_prob_reps[uncertain_indices], sub_wt_prob_reps, p=1
    )  # [K2, N]
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


def stratified_sampling2(inputs, labels, embeddings, cosine_of_each_pair, args):
    budget = 100
    # budget = int(max(100, np.ceil(len(labels) / (args.k * (args.k + 1) / 2.0))))
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


def stratified_sampling(inputs, labels, embeddings, cosine_of_each_pair, args):
    budget = 100
    # budget = int(max(100, np.ceil(len(labels) / (args.k * (args.k + 1) / 2.0))))
    # budget = int(max(100, np.ceil(len(labels) / args.k)))
    # by cosine similarity and null values distribution
    df = pd.DataFrame({"id": list(range(len(inputs))), "score": cosine_of_each_pair})
    df["group"] = pd.cut(df["score"], bins=budget, labels=list(range(budget)))
    group_id, _ = zip(
        *sorted(
            df.groupby(["group"]).size().reset_index().values,
            key=lambda x: x[-1],
        )
    )
    group_num = len(group_id)
    sample_indices = []
    for i in range(group_num):
        df_sub = df[df.group == i]
        if len(df_sub) == 0:
            continue
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


def MFL(selected_reps, candidate_reps, sample_reps):
    dist_selected = distance_matrix(selected_reps, sample_reps, p=1)
    dist_min_cur = np.min(dist_selected, axis=0)  # [N]
    dist_candidate = distance_matrix(candidate_reps, sample_reps, p=1)
    improvement = dist_min_cur[None,] - dist_candidate  # [K2, N]
    improvement = np.clip(improvement, 0, None)
    improvement = np.sum(improvement, axis=1)  # [N]
    return np.argmax(improvement)


def centroid(prob_reps, uncertain_indices, selected_indices, flag="pos"):
    probs = prob_reps[:, -1]
    probs[selected_indices] = 2
    if flag == "pos":
        sub_reps = prob_reps[(probs > 0.5) & (probs < 2)]
    elif flag == "neg":
        sub_reps = prob_reps[probs <= 0.5]
    else:
        sub_reps = prob_reps[probs < 2]
    # calculate the distance between the uncertain indices and all the samples
    dist_uncertain = distance_matrix(
        prob_reps[uncertain_indices], prob_reps[uncertain_indices], p=1
    )  # [K2, N]
    dist_uncertain = np.sum(dist_uncertain, axis=1)  # [N]
    idx = np.argmin(dist_uncertain)
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
    p = np.clip(args.p, 1e-9, 1 - 1e-9)
    certain_thr = 1 + (p * np.log(p) + (1 - p) * np.log(1 - p)) / scaler
    historical_confs, historical_probs = [], []

    while len(selected_indices) < min(args.k, args.budget):
        print(f"****************")
        print(f"iteration {len(selected_indices) + 1}")
        if len(example_inputs) > 0:
            example_inputs, example_labels, example_embeddings = reorder(
                model_name,
                model,
                tokenizer,
                example_inputs,
                example_labels,
                example_embeddings,
                args,
            )
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
        # historical_confs.append(np.clip(conf, 0, certain_thr))  # [T, N]
        historical_probs.append(probs)  # [T, N]

        if flag_type == "conf":
            # calculate similarity between samples based on their predicted probs
            prob_reps = np.array(historical_probs)[:, uncertain_indices].T  # [K, T]
            # idx = fast_votek(prob_reps, uncertain_indices)
            idx, next_target = select(
                historical_probs,
                prob_reps,
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


def reorder(
    model_name,
    model,
    tokenizer,
    example_inputs,
    example_labels,
    example_embeddings,
    args,
):
    # iterative decide the order the samples
    inputs, labels, embeddings = [], [], []
    while len(example_inputs) > 1:
        test_prompts = construct_prompt(
            inputs, labels, embeddings, example_inputs, example_embeddings, args
        )
        preds, probs = inference(model_name, model, tokenizer, test_prompts, args)
        probs = np.array(probs)
        flags = np.array(preds == np.array(example_labels), int)
        if np.sum(labels) * 2 > len(labels):
            flags[np.array(example_labels) == 1] = 0
        elif np.sum(labels) * 2 < len(labels):
            flags[np.array(example_labels) == 0] = 0
        confs = (flags * 2 - 1) * np.maximum(probs, 1 - probs)
        most_conf_index = np.argmax(confs)
        inputs.append(example_inputs[most_conf_index])
        labels.append(example_labels[most_conf_index])
        embeddings.append(example_embeddings[most_conf_index])
        del example_inputs[most_conf_index]
        del example_labels[most_conf_index]
        del example_embeddings[most_conf_index]
    inputs += example_inputs
    labels += example_labels
    embeddings += example_embeddings
    print(labels)

    return inputs, labels, embeddings


def our_base(model_name, model, tokenizer, inputs, labels, embeddings, args):
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
            uncertain_indices = [np.argmin(conf_temp)]  # least confident
            flag_type = "select least conf from all confident"
        print(f"flag_type: {flag_type}")
        print(f"uncertain_indices ({len(uncertain_indices)}): {uncertain_indices}")
        # conf = np.clip(conf, 0, certain_thr)
        historical_confs.append(np.clip(conf, 0, certain_thr))  # [T, N]
        historical_probs.append(probs)  # [T, N]

        if flag_type == "conf":
            prob_reps = np.array(historical_probs).T  # [K, T]
            # prob_reps = np.clip(prob_reps, 0.35, 0.65)

            # calculate similarity between samples based on their predicted probs
            if len(selected_indices) > 0:
                dist_selected = distance_matrix(
                    prob_reps[selected_indices], prob_reps, p=1
                )  # [K, N]
            else:
                dist_selected = np.ones((1, len(prob_reps))) * 100
            dist_min_cur = np.min(dist_selected, axis=0)  # [N]
            dist_uncertain = distance_matrix(
                prob_reps[uncertain_indices], prob_reps, p=1
            )  # [K2, N]
            improvement = dist_min_cur[None,] - dist_uncertain  # [K2, N]
            improvement = np.clip(improvement, 0, None)
            print(list(np.sum(improvement > 0, axis=1)))
            improvement = np.sum(improvement, axis=1)  # [K2]
            print(list(np.round(improvement, 3)))
            idx = uncertain_indices[np.argmax(improvement)]
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
    while len(selected_indices) < args.budget:
        break
        # TODO
    selected_indices = [candidate_indices[idx] for idx in selected_indices]
    return selected_indices


def our_pairwise(model_name, model, tokenizer, inputs, labels, embeddings, args):
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
    p = np.clip(args.p, 1e-9, 1 - 1e-9)
    certain_thr = 1 + (p * np.log(p) + (1 - p) * np.log(1 - p)) / scaler
    historical_confs, historical_probs = [], []

    while len(selected_indices) < min(args.k, args.budget):
        pos_neg_diff = 2 * np.sum(example_labels) - len(example_labels)
        print(f"****************")
        print(f"iteration {len(selected_indices) + 1}")
        prompts = construct_prompt(
            example_inputs, example_labels, example_embeddings, inputs, embeddings, args
        )
        print(prompts[0])
        print("---------------\n\n")
        _, probs = inference(model_name, model, tokenizer, prompts, args)

        # calculate unselcted pairs
        unselected_indices = list(set(range(len(labels))) - set(selected_indices))
        unselected_probs = np.array(probs)[unselected_indices]
        unselected_labels = np.array(labels.copy())[unselected_indices]
        precision, recall, f1 = evaluate(unselected_labels, unselected_probs > 0.5)
        print(f"Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f}")

        probs = np.array(probs)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        historical_probs.append(probs)  # [T, N]
        reps = np.array(historical_probs).T  # [N, T]
        conf = 1 + (probs * np.log(probs) + (1 - probs) * np.log(1 - probs)) / scaler
        conf[selected_indices] = 2
        historical_confs.append(conf)

        if pos_neg_diff < 0:
            next = "pos"
        elif pos_neg_diff > 0:
            next = "neg"
        else:
            low_conf_pos_num = np.sum((conf < certain_thr) * (probs > 0.5))
            low_conf_neg_num = np.sum((conf < certain_thr) * (probs <= 0.5))
            if low_conf_pos_num >= low_conf_neg_num:
                next = "pos"
            else:
                next = "neg"

        idx = select_next(
            historical_probs,
            historical_confs,
            selected_indices,
            pos_neg_diff,
            conf,
            args,
            next=next,
        )
        selected_indices.append(idx)
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
        print(f"index: {idx}, label: {labels[idx]}, pred: {probs[idx]:.2f}")

        probs_output = [f"{v:.4f}" for v in probs]
        result = [v for v in zip(range(len(labels)), labels, probs_output)]
        print(result)
        print(f"****************\n\n")
        print("selected indices:", selected_indices)
        print("labels of selected:", example_labels)

    while len(selected_indices) < args.budget:
        break
        # TODO
    selected_indices = [candidate_indices[idx] for idx in selected_indices]
    return selected_indices


def our_progressive(model_name, model, tokenizer, inputs, labels, embeddings, args):
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
    p = np.clip(args.p, 1e-9, 1 - 1e-9)
    certain_thr = 1 + (p * np.log(p) + (1 - p) * np.log(1 - p)) / scaler
    historical_confs, historical_probs = [], []

    pos_thr, neg_thr = 0.9, 0.1
    while len(selected_indices) < min(args.k, args.budget):
        print(f"****************")
        print(f"iteration {len(selected_indices) + 1}")
        prompts = construct_prompt(
            example_inputs, example_labels, example_embeddings, inputs, embeddings, args
        )
        print(prompts[0])
        print("---------------\n\n")
        _, probs = inference(model_name, model, tokenizer, prompts, args)
        # calculate unselcted pairs
        unselected_indices = list(set(range(len(labels))) - set(selected_indices))
        unselected_probs = np.array(probs)[unselected_indices]
        unselected_labels = np.array(labels.copy())[unselected_indices]
        precision, recall, f1 = evaluate(unselected_labels, unselected_probs > 0.5)
        print(f"Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f}")
        probs = np.array(probs)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        historical_probs.append(probs)  # [T, N]
        reps = np.array(historical_probs).T  # [N, T]
        conf = 1 + (probs * np.log(probs) + (1 - probs) * np.log(1 - probs)) / scaler
        conf[selected_indices] = 2

        selected_cur1, selected_cur2 = None, None
        while True:
            cond1 = (conf < 2) * (probs > pos_thr)  # potential pos with low conf
            if np.sum(cond1) > 0:
                conf_temp = conf.copy()
                conf_temp[cond1 == False] = 2
                selected_cur1 = np.argmin(conf_temp)
                flag = f"select the least conf pos with pred > {pos_thr:.2f}"
            else:
                probs_temp = probs.copy()
                probs_temp[selected_indices] = 0
                selected_cur1 = np.argmax(probs_temp)
                flag = "select the most likely pos"
            print(
                f"flag = {flag}, selected_potential_pos = {selected_cur1}, "
                f"label = {labels[selected_cur1]}, "
                f"pred = {probs[selected_cur1]:.2f}"
            )
            conf[selected_cur1] = 2
            if labels[selected_cur1] == 1:
                pos_thr = max(pos_thr - 0.1, 0.5)
                break
            elif selected_cur2 is None:
                selected_cur2 = selected_cur1.copy()
            else:
                break

        if selected_cur2 is None:
            cond2 = (conf < 2) * (probs < neg_thr)
            if np.sum(cond2) > 0:
                conf_temp = conf.copy()
                conf_temp[cond2 == False] = 2
                selected_cur2 = np.argmin(conf_temp)
                flag = f"select the least conf neg with pred < {neg_thr:.2f}"
            else:
                probs_temp = probs.copy()
                probs_temp[selected_indices] = 1
                selected_cur2 = np.argmin(probs_temp)
                flag = "select the most likely neg"
            print(
                f"flag = {flag}, selected_potential_neg = {selected_cur2}, "
                f"label = {labels[selected_cur2]}, "
                f"pred = {probs[selected_cur2]:.2f}"
            )
            conf[selected_cur2] = 2
            if labels[selected_cur2] == 0:
                neg_thr = min(neg_thr + 0.1, 0.5)

        probs_output = [f"{v:.4f}" for v in probs]
        result = [v for v in zip(range(len(labels)), labels, probs_output)]
        print(result)
        print(f"****************\n\n")
        print("selected indices:", selected_indices)
        print("labels of selected:", example_labels)
        if np.random.rand() < 0.5:
            cur_selected_indices = [selected_cur1, selected_cur2]
        else:
            cur_selected_indices = [selected_cur2, selected_cur1]
        for idx in cur_selected_indices:
            selected_indices.append(idx)
            example_inputs.append(inputs[idx])
            example_labels.append(labels[idx])
            example_embeddings.append(embeddings[idx])

    while len(selected_indices) < args.budget:
        break
        # TODO
    selected_indices = [candidate_indices[idx] for idx in selected_indices]
    return selected_indices


def cal_conf(p):
    scaler = -np.log(0.5)
    return 1 + (p * np.log(p) + (1 - p) * np.log(1 - p)) / scaler


def select_next(
    model_name,
    model,
    tokenizer,
    inputs,
    labels,
    embeddings,
    historical_probs,
    historical_confs,
    selected_indices,
    pos_neg_diff,
    conf,
    args,
    next="pos",
):
    # if next == "pos" and pos_neg_diff < -1 or next == "neg" and pos_neg_diff > 1:
    #     p_upper = np.clip(args.p + abs(pos_neg_diff) / 10.0, 1e-9, 1 - 1e-9)
    #     p_lower = np.clip(0.5 + abs(pos_neg_diff) / 10.0, 1e-9, 1 - 1e-9)
    # else:
    p_lower, p_upper = 0.5, np.clip(args.p, 1e-9, 1 - 1e-9)
    p_upper2 = np.clip(1 - 1e-9, 1e-9, 1 - 1e-9)
    conf_upper = cal_conf(p_upper)
    conf_upper2 = cal_conf(p_upper2)
    conf_lower = cal_conf(p_lower)
    cond1 = (conf < conf_upper) & (conf > conf_lower)
    cond1_up = (conf > conf_upper) & (conf < 2)
    cond1_lw = (conf < conf_lower) & (conf >= 0)

    probs = np.array(historical_probs[-1])
    if next == "pos":
        cond2 = probs > 0.5
    else:
        cond2 = probs <= 0.5

    if np.sum(cond1 * cond2) > 0:
        uncertain_indices = np.where(cond1 * cond2)[0]
        flag = f"select from potential {next} within the conf range"
    elif np.sum(cond1_up * cond2) > 0:
        conf_temp = conf.copy()
        conf_temp[(cond1_up * cond2) == False] = 2
        uncertain_indices = [np.argmin(conf_temp)]
        flag = f"select from least conf potential {next} above the conf range"
    elif np.sum(cond1_lw * cond2) > 0:
        conf_temp = conf.copy()
        conf_temp[(cond1_lw * cond2) == False] = -1
        uncertain_indices = [np.argmax(conf_temp)]
        flag = f"select from most conf potential {next} below the conf range"
    else:
        uncertain_indices = [np.argmin(conf)]
        flag = f"no candidate, select from most likely potential {next}"

    if len(uncertain_indices) == 1:
        index = uncertain_indices[0]
    elif len(uncertain_indices) == 2:
        index = uncertain_indices[np.argmax(conf[uncertain_indices])]
    else:
        # reps = np.array(historical_probs).T
        # selected_reps = reps[selected_indices]
        # candidate_reps = reps[uncertain_indices]
        # sample_reps = reps[(conf > conf_lower) * (cond2)]
        # index = MFL(selected_reps, candidate_reps, sample_reps)
        # index = uncertain_indices[index]
        sample_indices = np.where((conf < conf_upper2) * cond2)[0]
        if len(uncertain_indices) > 10:
            uncertain_indices = sampling(historical_probs, uncertain_indices, probs)
        if len(sample_indices) > 10:
            sample_indices = sampling(historical_probs, sample_indices, probs)
        print(f"uncertain_indices: {uncertain_indices}")
        print(f"sample_indices: {sample_indices}")
        index = max_info_gain(
            model_name,
            model,
            tokenizer,
            inputs,
            labels,
            embeddings,
            selected_indices,
            uncertain_indices,
            sample_indices,
            probs,
            args,
            next,
        )
    print(f"flag = {flag}, selected_index = {index}")
    return index


def sampling(historical_probs, indices, probs, n=10):
    # probs = probs[indices]
    # indices, probs = zip(*sorted(zip(indices, probs), key=lambda x: x[-1]))
    # start_idx = 0
    # interval = len(indices) // n
    # new_indices = []
    # while len(new_indices) < n:
    #     interval = (len(indices) - start_idx) // (n - len(new_indices))
    #     new_indices.append(indices[start_idx])
    #     start_idx += interval
    reps = np.array(historical_probs).T
    reps = reps[indices]
    sim_matrix = distance_matrix(reps, reps, p=1)
    sim_matrix = 1 - sim_matrix / np.max(sim_matrix)
    similarity_to_labeled = np.array([-1] * len(indices))
    selected_indices, scores = [], []
    while len(selected_indices) < n:
        max_score, idx = -1, -1
        for i in range(len(indices)):
            if i in selected_indices:
                continue
            value = sim_matrix[i] - similarity_to_labeled
            score = np.sum(value[value > 0])
            if score > max_score:
                max_score = score
                idx = i
        selected_indices.append(idx)
        similarity_to_labeled = np.maximum(similarity_to_labeled, sim_matrix[idx])
        scores.append(max_score)
    selected_indices = np.array(indices)[selected_indices]
    return selected_indices


def max_info_gain(
    model_name,
    model,
    tokenizer,
    inputs,
    labels,
    embeddings,
    selected_indices,
    uncertain_indices,
    sample_indices,
    probs,
    args,
    next,
):
    example_inputs, example_labels, example_embeddings = [], [], []
    for idx in selected_indices:
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
    sample_inputs, sample_embeddings = [], []
    for idx in sample_indices:
        sample_inputs.append(inputs[idx])
        sample_embeddings.append(embeddings[idx])
    # calculate information gain of each uncertain sample
    info_gain = []
    print(f"uncertain_indices: {uncertain_indices}")
    for idx in uncertain_indices:
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
        prompts = construct_prompt(
            example_inputs,
            example_labels,
            example_embeddings,
            sample_inputs,
            sample_embeddings,
            args,
        )
        _, probs_temp = inference(model_name, model, tokenizer, prompts, args)
        probs_temp = np.array(probs_temp)
        conf_temp = np.maximum(probs_temp, 1 - probs_temp)
        conf_org = np.maximum(probs, 1 - probs)
        conf_diff = conf_temp - conf_org[sample_indices]
        if idx in sample_indices:
            sample_idx = list(sample_indices).index(idx)
            conf_diff[sample_idx] = 0
            improvement = np.sum(conf_diff) / (len(sample_indices) - 1)
        else:
            improvement = np.sum(conf_diff) / len(sample_indices)
        info_gain.append(improvement)
        print(
            f"newly added index: {idx}, predicted_prob: {probs[idx]}, "
            f"label: {labels[idx]}, info_gain: {info_gain[-1]}"
        )
        del example_inputs[-1]
        del example_labels[-1]
        del example_embeddings[-1]
    return uncertain_indices[np.argmax(info_gain)]


def select_by_cosine_sim(
    model_name, model, tokenizer, inputs, labels, embeddings, args
):
    # sample by cosine similarity
    cosine_of_each_pair = cal_cosine_sim(args)
    inputs, labels, embeddings, candidate_indices, scores = stratified_sampling(
        inputs, labels, embeddings, cosine_of_each_pair, args
    )
    print(f"candidates: {len(inputs)}")
    labels, candidate_indices, scores = zip(
        *sorted(zip(labels, candidate_indices, scores), key=lambda x: x[-1])
    )
    left_index, right_index = 0, len(candidate_indices) - 1
    target = "pos"
    selected_indices = []
    while len(selected_indices) < args.budget:
        if target == "pos":
            index = right_index
            right_index -= 1
            if labels[index] == 1:
                target = "neg"
        else:
            index = left_index
            left_index += 1
            if labels[index] == 0:
                target = "pos"
        selected_indices.append(candidate_indices[index])
    return selected_indices


def ideal(model_name, model, tokenizer, inputs, labels, embeddings, args):
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
    p = np.clip(args.p, 1e-9, 1 - 1e-9)
    certain_thr = 1 + (p * np.log(p) + (1 - p) * np.log(1 - p)) / scaler
    historical_confs, historical_probs = [], []

    while len(selected_indices) < min(args.k, args.budget):
        pos_neg_diff = 2 * np.sum(example_labels) - len(example_labels)
        print(f"****************")
        print(f"iteration {len(selected_indices) + 1}")
        prompts = construct_prompt(
            example_inputs, example_labels, example_embeddings, inputs, embeddings, args
        )
        print(prompts[0])
        print("---------------\n\n")
        _, probs = inference(model_name, model, tokenizer, prompts, args)

        # calculate unselcted pairs
        unselected_indices = list(set(range(len(labels))) - set(selected_indices))
        unselected_probs = np.array(probs)[unselected_indices]
        unselected_labels = np.array(labels.copy())[unselected_indices]
        precision, recall, f1 = evaluate(unselected_labels, unselected_probs > 0.5)
        print(f"Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f}")

        probs = np.array(probs)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        historical_probs.append(probs)  # [T, N]
        reps = np.array(historical_probs).T  # [N, T]
        conf = 1 + (probs * np.log(probs) + (1 - probs) * np.log(1 - probs)) / scaler
        conf[selected_indices] = 2
        historical_confs.append(conf)

        if pos_neg_diff < 0:
            next = "pos"
        elif pos_neg_diff > 0:
            next = "neg"
        else:
            if np.sum((conf < 2) * (probs > 0.5)) == 0:
                next = "pos"
            elif np.sum((conf < 2) * (probs <= 0.5)) == 0:
                next = "neg"
            else:
                low_conf_pos_num = np.sum((conf < certain_thr) * (probs > 0.5))
                low_conf_neg_num = np.sum((conf < certain_thr) * (probs <= 0.5))
                if low_conf_pos_num >= low_conf_neg_num:
                    next = "pos"
                else:
                    next = "neg"

        idx = select_next(
            model_name,
            model,
            tokenizer,
            inputs,
            labels,
            embeddings,
            historical_probs,
            historical_confs,
            selected_indices,
            pos_neg_diff,
            conf,
            args,
            next=next,
        )
        selected_indices.append(idx)
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
        print(f"index: {idx}, label: {labels[idx]}, pred: {probs[idx]:.2f}")

        probs_output = [f"{v:.4f}" for v in probs]
        result = [v for v in zip(range(len(labels)), labels, probs_output)]
        print(result)
        print(f"****************\n\n")
        print("selected indices:", selected_indices)
        print("labels of selected:", example_labels)

    while len(selected_indices) < args.budget:
        break
        # TODO
    selected_indices = [candidate_indices[idx] for idx in selected_indices]
    return selected_indices
