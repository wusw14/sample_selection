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
from scipy.stats import rankdata


def agg_probs(prob_reps, window=2):
    prob_reps = np.array(prob_reps)[:, -window:]
    T = len(prob_reps[0])
    # weighted average of prob_reps
    weights = np.array([1.0 / 2 ** (T - i) for i in range(T)])
    weights = weights / np.sum(weights)
    wt_prob_reps = prob_reps * weights[None,]
    wt_probs = np.sum(wt_prob_reps, axis=1)
    return prob_reps, wt_prob_reps, wt_probs


def cal_cosine_sim(args):
    data_dir = args.data_dir.replace("data", "temp_data")
    embeddingsA = np.loadtxt(f"{data_dir}/train_A_emb.npy")
    embeddingsB = np.loadtxt(f"{data_dir}/train_B_emb.npy")
    embeddingsA = embeddingsA / np.linalg.norm(embeddingsA, axis=1, keepdims=True)
    embeddingsB = embeddingsB / np.linalg.norm(embeddingsB, axis=1, keepdims=True)
    cosine_sim = np.sum(embeddingsA * embeddingsB, axis=1)
    return cosine_sim


def stratified_sampling(inputs, labels, embeddings, cosine_of_each_pair, args):
    budget = 100
    # budget = int(max(100, np.ceil(len(labels) / (args.k * (args.k + 1) / 2.0))))
    # budget = int(max(100, np.ceil(len(labels) / args.k)))
    # by cosine similarity and null values distribution
    df = pd.DataFrame({"id": list(range(len(inputs))), "score": cosine_of_each_pair})
    # bin_num = budget // 5
    bin_num = budget
    df["group"] = pd.cut(df["score"], bins=bin_num, labels=list(range(bin_num)))
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
    selected_indices,
    labeled_set,
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
    conf_upper = conf_func(p_upper)
    conf_upper2 = conf_func(p_upper2)
    conf_lower = conf_func(p_lower)
    cond1 = (conf < conf_upper) & (conf > conf_lower)
    cond1_up = (conf > conf_upper) & (conf < 2)
    cond1_lw = (conf < conf_lower) & (conf >= 0)
    labeled_eval = labeled_set - set(selected_indices)
    print("labeled_eval", labeled_eval)

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
        uncertain_indices = [index]
    else:
        candidate_indices = np.where((conf < conf_upper2))[0]
        if len(uncertain_indices) > 5:
            uncertain_indices, _ = sampling(
                historical_probs,
                uncertain_indices,
                uncertain_indices,
                [],
                n=args.beam_size,
                type="Kmeans",
            )
        print(f"uncertain_indices: {uncertain_indices}")
        sample_indices, _ = sampling(
            historical_probs,
            candidate_indices,
            candidate_indices,
            [],
            n=20,
            type="Kmeans",
        )

        print("sample indices", sample_indices)
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
            labeled_eval,
            probs,
            args,
        )
    print(f"flag = {flag}, selected_index = {index}")
    return index, uncertain_indices


def sampling(
    historical_probs,
    indices,
    indices_all,
    labeled_indices,
    n=10,
    type="MFL_sampling",
):
    # MFL
    probs = historical_probs[-1]
    reps = np.array(historical_probs).T  # [N, T]
    all_reps = reps[indices_all]
    candidate_reps = reps[indices]
    if type == "Kmeans":
        kmeans = KMeans(n_clusters=n, random_state=0).fit(candidate_reps)
        selected_indices = []
        for i in range(n):
            dist = distance_matrix([kmeans.cluster_centers_[i]], candidate_reps, p=1)
            dist = dist[0]
            selected_indices.append(indices[np.argmin(dist)])
        return selected_indices, np.ones(len(selected_indices))
    if len(labeled_indices) > 0:
        labeled_reps = reps[labeled_indices]
        dist_selected = distance_matrix(labeled_reps, all_reps, p=1)  # [N1, N]
        dist_min_cur = np.min(dist_selected, axis=0)
    else:
        dist_min_cur = np.ones(len(all_reps)) * 1e9
    dist_matrix = distance_matrix(candidate_reps, all_reps, p=1)  # [N2, N]
    selected_indices, scores = [], []
    if type == "MFL_sampling":
        while len(selected_indices) < n:
            max_score, idx = -1, -1
            for i in range(len(indices)):
                if i in selected_indices:
                    continue
                value = dist_min_cur - dist_matrix[i]
                score = np.sum(value[value > 0])
                if score > max_score:
                    max_score = score
                    idx = i
            selected_indices.append(idx)
            dist_min_cur = np.minimum(dist_min_cur, dist_matrix[idx])
            scores.append(max_score)
    elif type == "MFL_rank":
        value_list = []
        for i in range(len(indices)):
            value = dist_min_cur - dist_matrix[i]
            value_list.append(np.sum(value[value > 0]))
        value_list = np.array(value_list)
        selected_indices = np.argsort(value_list)[::-1][:n]
    selected_indices = np.array(indices)[selected_indices]
    print("selected_indices", selected_indices)
    print("probs", np.round(probs[selected_indices], 4))
    return selected_indices, np.ones(len(selected_indices))


def conf_func(prob):
    # v1
    conf = np.maximum(prob, 1 - prob)
    # v2
    # conf = 1 - 1 / (1 + np.exp(5 * np.abs(prob - 0.5)))
    # v3
    # conf = 1 + (prob * np.log(prob) + (1 - prob) * np.log(1 - prob))
    return conf


def cal_conf_avg(conf, added_idx, sample_indices, probs, target):
    if target == "pos":
        cond = probs > 0.5
    elif target == "neg":
        cond = probs < 0.5
    else:
        cond = np.ones(len(conf), dtype=bool)
    sample_indices = np.array(sample_indices)[cond]
    conf = conf[cond]
    if added_idx in sample_indices:
        index = list(sample_indices).index(added_idx)
        conf_avg = (np.sum(conf) - conf[index]) / (len(conf) - 1)
    else:
        conf_avg = np.mean(conf)
    return conf_avg


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
    labeled_eval,
    probs,
    args,
):
    if len(uncertain_indices) == 1:
        return uncertain_indices[0]
    example_inputs, example_labels, example_embeddings = [], [], []
    for idx in selected_indices:
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
    sample_inputs, sample_embeddings = [], []
    for idx in sample_indices:
        sample_inputs.append(inputs[idx])
        sample_embeddings.append(embeddings[idx])
    labeled_inputs, labeled_embeddings, labeled_labels = [], [], []
    labeled_eval = list(labeled_eval)
    for idx in labeled_eval:
        labeled_inputs.append(inputs[idx])
        labeled_embeddings.append(embeddings[idx])
        labeled_labels.append(labels[idx])
    labeled_labels = np.array(labeled_labels)
    print("labeled_labels", labeled_labels)

    # calculate information gain of each uncertain sample
    probs_org = np.array(probs)[sample_indices]
    conf_org = conf_func(probs_org)
    conf_p1_avg = cal_conf_avg(conf_org, -1, sample_indices, probs_org, "pos")
    conf_p2_avg = cal_conf_avg(conf_org, -1, sample_indices, probs_org, "neg")
    print(f"sampled_probs: {np.round(probs_org, 4)}")
    print(f"Original conf_p1: {conf_p1_avg:.4f}, conf_p2: {conf_p2_avg:.4f}")
    info_gain = []
    for i, idx in enumerate(uncertain_indices):
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
        # part 1: uncertainty calculation
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
        conf_temp = conf_func(probs_temp)
        conf_p1_avg = cal_conf_avg(conf_temp, idx, sample_indices, probs_temp, "pos")
        conf_p2_avg = cal_conf_avg(conf_temp, idx, sample_indices, probs_temp, "neg")
        conf_avg = cal_conf_avg(conf_temp, idx, sample_indices, probs_temp, "all")
        score1 = conf_avg / 2 + min(conf_p1_avg, conf_p2_avg) / 2
        # part 2: informativeness calculation
        if len(labeled_eval) > 0:
            prompts = construct_prompt(
                example_inputs,
                example_labels,
                example_embeddings,
                labeled_inputs,
                labeled_embeddings,
                args,
            )
            _, labeled_probs = inference(model_name, model, tokenizer, prompts, args)
            labeled_probs = np.array(labeled_probs)
            labeled_probs = np.clip(labeled_probs, 1e-6, 1 - 1e-6)
            cross_entropy = labeled_labels * np.log(labeled_probs) + (
                1 - labeled_labels
            ) * np.log(1 - labeled_probs)
            print(cross_entropy)
            cross_entropy = np.clip(cross_entropy, -10, np.log(0.5))
            score2 = cal_conf_avg(cross_entropy, idx, labeled_eval, probs_temp, "all")
        else:
            score2 = 0
        score = score1 + score2
        info_gain.append(score)
        print(
            f"newly added index: {idx}, predicted_prob: {probs[idx]:.4f}, "
            f"ground truth label: {labels[idx]}, info_gain: {info_gain[-1]:.4f} "
            f"conf_p1: {conf_p1_avg:.4f}, conf_p2: {conf_p2_avg:.4f} "
            f"conf_avg: {conf_avg:.4f}, score1: {score1:.4f}, score2: {score2:.4f}"
        )
        del example_inputs[-1]
        del example_labels[-1]
        del example_embeddings[-1]
    print(f"uncertain_indices: {uncertain_indices}")
    print(np.round(np.array(info_gain), 4))
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
    labeled_set = set(selected_indices)

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
        conf = conf_func(probs)
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
                pos_conf = np.sum(conf * (probs > 0.5) * (conf < 2)) / np.sum(
                    (conf < 2) * (probs > 0.5)
                )
                neg_conf = np.sum(conf * (probs <= 0.5) * (conf < 2)) / np.sum(
                    (conf < 2) * (probs <= 0.5)
                )
                print(f"pos_conf: {pos_conf:.2f}, neg_conf: {neg_conf:.2f}")
                if pos_conf > neg_conf:
                    next = "neg"
                else:
                    next = "pos"
        idx, labeled_indices = select_next(
            model_name,
            model,
            tokenizer,
            inputs,
            labels,
            embeddings,
            historical_probs,
            selected_indices,
            labeled_set,
            conf,
            args,
            next=next,
        )
        selected_indices.append(idx)
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
        labeled_set = labeled_set.union(set(labeled_indices))
        print(f"index: {idx}, label: {labels[idx]}, pred: {probs[idx]:.2f}")

        probs_output = [f"{v:.4f}" for v in probs]
        result = [v for v in zip(range(len(labels)), labels, probs_output)]
        print(result)
        print(f"****************\n\n")
        print("selected indices:", selected_indices)
        print("labels of selected:", example_labels)

    labeled_eval = set(labeled_set) - set(selected_indices)
    print("labeled_eval", labeled_eval)
    if len(labeled_eval) > 0:
        labeled_inputs, labeled_embeddings, labeled_labels = [], [], []
        for idx in labeled_eval:
            labeled_inputs.append(inputs[idx])
            labeled_embeddings.append(embeddings[idx])
            labeled_labels.append(labels[idx])
        prompts = construct_prompt(
            example_inputs,
            example_labels,
            example_embeddings,
            labeled_inputs,
            labeled_embeddings,
            args,
        )
        _, labeled_probs = inference(model_name, model, tokenizer, prompts, args)
        labeled_probs = np.array(labeled_probs)
        # find cut-off point to maximize F1 on labeled
        optimal_p = 0.5
        _, _, f1_best = evaluate(labeled_labels, labeled_probs > 0.5)
        for p in np.arange(0, 1, 0.1):
            precision, recall, f1 = evaluate(labeled_labels, labeled_probs > p)
            print(
                f"p: {p:.2f}, Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f}"
            )
            if f1 > f1_best:
                f1_best = f1
                optimal_p = p
        print(f"optimal cut-off point: {optimal_p}")

    while len(selected_indices) < args.budget:
        break
        # TODO
    selected_indices = [candidate_indices[idx] for idx in selected_indices]
    return selected_indices
