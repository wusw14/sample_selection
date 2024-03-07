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
from sklearn.metrics import f1_score


def cal_cosine_sim(args):
    data_dir = args.data_dir.replace("data", "temp_data")
    embsA = np.loadtxt(f"{data_dir}/train_A_emb.npy")
    embsB = np.loadtxt(f"{data_dir}/train_B_emb.npy")
    embsA = embsA / np.linalg.norm(embsA, axis=1, keepdims=True)
    embsB = embsB / np.linalg.norm(embsB, axis=1, keepdims=True)
    cosine_sim = np.sum(embsA * embsB, axis=1)
    return cosine_sim


def stratified_sampling(inputs, labels, embs, cosine_of_each_pair, args):
    budget = 250
    if len(labels) <= budget:
        return inputs, labels, embs, list(range(len(inputs))), cosine_of_each_pair
    # by cosine similarity and null values distribution
    df = pd.DataFrame({"id": list(range(len(inputs))), "score": cosine_of_each_pair})
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
    for i, gid in enumerate(group_id):
        df_sub = df[df.group == gid]
        if len(df_sub) == 0:
            continue
        n = (budget - len(sample_indices)) // (group_num - i)
        if n >= len(df_sub):
            sample_indices += df_sub["id"].tolist()
        else:
            sample_indices += df_sub.sample(n=n, random_state=args.seed)["id"].tolist()
    sample_inputs = [inputs[idx] for idx in sample_indices]
    sample_labels = [labels[idx] for idx in sample_indices]
    sample_embs = [embs[idx] for idx in sample_indices]
    scores = [cosine_of_each_pair[idx] for idx in sample_indices]
    print(
        f"pos/neg in candidates: {sum(sample_labels)}/{len(sample_labels)-sum(sample_labels)}"
    )
    return sample_inputs, sample_labels, sample_embs, sample_indices, scores


def select_next(
    model_name,
    model,
    tokenizer,
    inputs,
    labels,
    embs,
    historical_probs,
    selected_indices,
    labeled_set,
    historical_info,  # [{}, {}] 0: pos, 1: neg
    args,
    next="pos",
    no_improvement=0,
):
    probs = np.array(historical_probs[-1])
    labels = np.array(labels)
    conf = conf_func(probs)
    cond1 = (probs > 0.5) & (probs <= 1)
    cond2 = probs <= 0.5
    if next == 1:
        cond = cond1
    else:
        cond = cond2

    if np.sum(cond) > 0:
        uncertain_indices = np.where(cond)[0]
        flag = f"select from potential {next} within the conf range"
    else:
        uncertain_indices = [np.argmin(conf)]
        flag = f"no candidate, select from most likely potential {next}"

    if len(uncertain_indices) == 1:
        index = uncertain_indices[0]
    elif len(uncertain_indices) == 2:
        index = uncertain_indices[np.argmax(conf[uncertain_indices])]
        uncertain_indices = [index]
    else:
        if len(uncertain_indices) > args.beam_size:
            # false_indices = []
            # if next == 1:
            #     info_dict = historical_info[0]
            # else:
            #     info_dict = historical_info[1]
            # # get top from info_dict
            # if len(info_dict) > 0:
            #     top_indices, _ = zip(
            #         *sorted(
            #             info_dict.items(),
            #             key=lambda x: np.mean(x[-1]),
            #             reverse=True,
            #         )
            #     )
            #     for idx in top_indices[args.beam_size :]:
            #         if int(probs[idx] > 0.5) != int(next == 1):
            #             false_indices.append(idx)
            #         if len(false_indices) >= args.beam_size:
            #             break
            #     top_indices = list(top_indices)[: args.beam_size] + false_indices
            # else:
            #     top_indices = []
            # print(f"*** top_indices: {top_indices}, false_indices: {false_indices}")
            uncertain_indices, _ = sampling(
                historical_probs,
                uncertain_indices,
                [[], labeled_set],
                n=args.beam_size,
                type="covered_by_rep",
            )
            # uncertain_indices = list(uncertain_indices) + top_indices
        print(f"uncertain_indices: {uncertain_indices}")
        eval_indices, _ = sampling(
            historical_probs,
            np.where(cond1 | cond2)[0],
            None,
            n=args.eval_size,
            type="covered_by_rep",
        )
        print(f"### eval indices: {list(eval_indices)}")
        labeled_eval = labeled_set.union(set(uncertain_indices)) - set(selected_indices)
        print("labeled_eval", labeled_eval)
        if next == 1:
            info_dict = historical_info[0]
        else:
            info_dict = historical_info[1]
        if len(info_dict) > 0:
            hist_indices, _ = zip(
                *sorted(info_dict.items(), key=lambda x: np.mean(x[-1]), reverse=True)
            )
            if no_improvement >= 2:
                hist_indices = hist_indices[:2]
        else:
            hist_indices = []
        index, info_dict = max_info_gain(
            model_name,
            model,
            tokenizer,
            inputs,
            labels,
            embs,
            selected_indices,
            uncertain_indices,
            hist_indices,
            eval_indices,
            labeled_eval,
            probs,
            args,
        )
        # delete the selected index from historical info
        if index is not None:
            if labels[index] == 1:
                if index in historical_info[0]:
                    del historical_info[0][index]
            else:
                if index in historical_info[1]:
                    del historical_info[1][index]
        # update historical info
        for idx, value in info_dict.items():
            if idx == index:
                continue
            if labels[idx] == 1:
                historical_info[0][idx].append(value)
            else:
                historical_info[1][idx].append(value)
    print(f"flag = {flag}, selected_index = {index}")
    return index, uncertain_indices, historical_info


def sampling(
    historical_probs,
    indices,
    labeled_info,  # [top_indices, labeled_set]
    n=10,
    type="MFL_whole",
):
    labeled_indices_new = []
    selected_indices = []
    if labeled_info is not None:
        for idx in labeled_info[1]:
            if idx in indices:
                labeled_indices_new.append(list(indices).index(idx))
                if idx in labeled_info[0]:
                    selected_indices.append(list(indices).index(idx))
    if n >= len(indices) - len(labeled_indices_new):
        return list(set(indices) - set(labeled_indices_new)), np.ones(len(indices))
    probs = historical_probs[-1]
    reps = np.array(historical_probs).T  # [N, T]
    # reweighting the reps
    weights = np.array([1.0 / 2 ** (len(reps[0]) - i) for i in range(len(reps[0]))])
    reps = reps * weights[None,]
    candidate_reps = reps[indices]
    candidate_probs = np.array(probs)[indices]
    if type == "Kmeans":
        kmeans = KMeans(n_clusters=n, random_state=0).fit(candidate_reps)
        selected_indices = []
        for i in range(n):
            dist = distance_matrix([kmeans.cluster_centers_[i]], candidate_reps, p=1)
            dist = dist[0]
            selected_indices.append(indices[np.argmin(dist)])
        return selected_indices, np.ones(len(selected_indices))
    elif type == "coverage":
        selected_indices = coverage(indices, candidate_probs, candidate_reps, n)
    elif type == "coverage_by_conf":
        selected_indices = coverage_by_conf(candidate_probs, candidate_reps, n)
    elif type == "covered_by_rep":
        selected_indices = coverage_by_rep(
            candidate_probs,
            candidate_reps,
            n + len(labeled_indices_new),
            selected_indices,
            labeled_indices_new,
        )
    if type == "coverage":
        selected_indices = np.array(selected_indices)
    else:
        selected_indices = np.array(indices)[selected_indices]
    print(f"[{type}] selected_indices: {list(selected_indices)}")
    print(f"[{type}] probs: {list(np.round(probs[selected_indices], 4))}")
    return selected_indices, np.ones(len(selected_indices))


def coverage_by_conf(probs, reps, n):
    # partition into bins based on the recent confs
    confs = conf_func(probs)
    print(
        f"#Coverage Func# max conf: {np.max(confs):.4f}, min conf: {np.min(confs):.4f}"
    )
    df_confs = pd.DataFrame({"id": list(range(len(probs))), "conf": confs})
    df_confs["group"] = pd.cut(
        df_confs["conf"], bins=n // 2, labels=list(range(n // 2))
    )
    # allocate the budget to each bin
    budget = np.zeros(n // 2)
    for i in range(n // 2):
        df_sub = df_confs[df_confs.group == i]
        budget[i] = len(df_sub)
    budget = np.array(budget)
    # # second max
    # budget_2nd_max = np.sort(budget)[-2]
    # budget = np.minimum(budget, budget_2nd_max)  # [n/2]
    budget = budget / np.sum(budget) * n
    group_id = np.argsort(budget)
    # select the centroid one
    selected_indices = []
    for i in group_id:
        df_sub = df_confs[df_confs.group == i]
        if len(df_sub) == 0:
            continue
        num = max(1, int(round(budget[i], 0)))
        # update budget
        budget[i] = 0
        if np.sum(budget) > 0:
            budget = budget / np.sum(budget) * (n - len(selected_indices) - num)
        df_sub["sub_group"] = pd.cut(df_sub["conf"], bins=num, labels=list(range(num)))
        for j in range(num):
            cand_indices = df_sub[df_sub.sub_group == j].id.tolist()
            if len(cand_indices) == 0:
                continue
            cand_reps = reps[cand_indices]
            # select the centroid one
            dist_bw_cand = np.sum(distance_matrix(cand_reps, cand_reps, p=1), 1)
            selected_indices.append(cand_indices[np.argmin(dist_bw_cand)])
    return selected_indices


def coverage_by_rep(probs, reps, n, selected_indices_org, labeled_indices):
    dist = distance_matrix(reps, reps, p=1)  # [N, N]
    m = n
    m_list = []
    sample_list = []
    max_score, max_cover_range = 0, 0
    confs = conf_func(probs)
    org_conf_range = np.max(confs) - np.min(confs)
    print(
        f"budget: {n}, labeled_indices: {labeled_indices}, "
        f"org_conf_range: {org_conf_range:.4f}, "
        f"max_prob: {np.max(probs):.4f}, min_prob: {np.min(probs):.4f}"
    )
    while True:
        m_list.append(m)
        dist_thr = np.percentile(dist, 100.0 / m)
        graph = {}
        for i in range(len(probs)):
            graph[i] = []
            for j in range(len(probs)):
                if dist[i, j] <= dist_thr:
                    graph[i].append(j)
        covered_set = set(list(labeled_indices))
        selected_indices = list(labeled_indices)
        for idx in selected_indices:
            covered_set = covered_set.union(set(graph[idx]))
        if len(sample_list) == 0:
            print(f"org covered_set: {covered_set}")
        while len(covered_set) < len(probs) and len(selected_indices) < n:
            max_covered = -1
            idx = -1
            max_dist = 0
            for i in range(len(probs)):
                if i in covered_set:
                    continue
                new_covered = len(set(graph[i]) - covered_set)
                if len(selected_indices) == 0:
                    new_dist = 0
                else:
                    new_dist = np.sum([dist[i, j] for j in selected_indices])
                if (
                    new_covered > max_covered
                    or (new_covered == max_covered)
                    and (new_dist > max_dist)
                ):
                    max_covered = new_covered
                    max_dist = new_dist
                    idx = i
            covered_set = covered_set.union(set(graph[idx]))
            selected_indices.append(idx)
        covered_conf_range = np.max(confs[selected_indices]) - np.min(
            confs[selected_indices]
        )
        score = (
            10 * len(selected_indices) / n
            + covered_conf_range / org_conf_range
            + len(covered_set) / len(probs)
        )
        if max_score < score:
            max_score = score
            sample_list = list(selected_indices)
            max_cover_range = covered_conf_range
            max_cover_ratio = len(covered_set) / len(probs)
            best_dist_thr = dist_thr
            best_m = m
        if len(selected_indices) < n and len(covered_set) >= len(probs):
            m += 1
        elif len(selected_indices) >= n and len(covered_set) < len(probs):
            m -= 1
        else:
            break
        if m in m_list or m < 1 or m > len(probs):
            break
    print(
        f"covered_conf_range: {max_cover_range:.4f}, best_m: {best_m}, "
        f"covered_ratio: {max_cover_ratio:.4f}, dist_thr: {best_dist_thr:.4f}, "
        f"m_list: {m_list}"
    )
    if len(sample_list) < n:
        print(f"remaining budget: {n - len(sample_list)}")
        print(f"sample_list (before): {len(sample_list)}")
        sample_list += np.random.choice(
            list(set(range(len(probs))) - set(sample_list)),
            n - len(sample_list),
            replace=False,
        ).tolist()
        print(f"sample_list (after): {len(sample_list)}")
    sample_list = list(set(sample_list) - set(labeled_indices))
    return sample_list


def coverage(indices, candidate_probs, reps, n):
    # partition into bins based on the recent confs
    candidate_confs = conf_func(candidate_probs)
    print(
        f"max conf: {np.max(candidate_confs):.4f}, "
        f"min conf: {np.min(candidate_confs):.4f}"
    )
    df_confs = pd.DataFrame({"id": indices, "conf": candidate_confs})
    df_confs["group"] = pd.cut(df_confs["conf"], bins=n, labels=list(range(n)))
    i = 0
    while i < n:
        if len(df_confs) <= n - i:
            selected_indices += df_confs.id.tolist()
            break
        df_sub = df_confs[df_confs.group == i]
        if len(df_sub) == 0:
            df_confs = df_confs[df_confs.group > i]
            if len(df_confs) == 0:
                break
            df_confs["group"] = pd.cut(
                df_confs["conf"], bins=n - i, labels=list(np.arange(i, n))
            )
            continue
        cand_indices = df_sub.id.tolist()
        cand_reps = reps[cand_indices]
        # select the centroid one
        dist_bw_cand = np.sum(distance_matrix(cand_reps, cand_reps, p=1), 1)
        selected_indices.append(cand_indices[np.argmin(dist_bw_cand)])
        i += 1
    return selected_indices


def conf_func(prob):
    # v1
    conf = np.maximum(prob, 1 - prob)
    # v2
    # conf = 1 - 1 / (1 + np.exp(5 * np.abs(prob - 0.5)))
    # v3
    # conf = 1 + (prob * np.log(prob) + (1 - prob) * np.log(1 - prob))
    return conf


def cal_avg(conf, added_idx, sample_indices, probs, target):
    if target == "pos":
        cond = probs > 0.5
    elif target == "neg":
        cond = probs < 0.5
    else:
        cond = np.ones(len(conf), dtype=bool)
    sample_indices = np.array(sample_indices)[cond]
    conf = conf[cond]
    if len(conf) == 0:
        return 1
    if added_idx in sample_indices:
        index = list(sample_indices).index(added_idx)
        if len(conf) == 1:
            return 1
        conf_avg = (np.sum(conf) - conf[index]) / (len(conf) - 1)
    else:
        conf_avg = np.mean(conf)
    return conf_avg


def cal_info_score(
    probs1, indices1, probs2, indices2, labels2, selected_index=-1, p=-1, label=-1
):
    # cal certainty score
    probs1 = np.array(probs1)
    conf = conf_func(probs1)
    conf_p1_avg = cal_avg(conf, selected_index, indices1, probs1, "pos")
    conf_p2_avg = cal_avg(conf, selected_index, indices1, probs1, "neg")
    conf_avg = cal_avg(conf, selected_index, indices1, probs1, "all")
    score1 = conf_avg / 2 + min(conf_p1_avg, conf_p2_avg) / 2

    # cal accuracy
    probs2 = np.array(probs2)
    CE_scores = labels2 == (probs2 > 0.5)
    CE_pos = cal_avg(CE_scores, selected_index, indices2, labels2, "pos")
    CE_neg = cal_avg(CE_scores, selected_index, indices2, labels2, "neg")
    CE_avg = cal_avg(CE_scores, selected_index, indices2, labels2, "all")
    score2 = CE_avg / 2 + min(CE_pos, CE_neg) / 2
    score = score1 + score2 / 2

    if selected_index == -1:
        printinfo = f"[Base] "
    else:
        printinfo = (
            f"[New] newly added index: {selected_index}, predicted_prob: {p:.4f}, "
            f"ground truth label: {label}, "
        )
    print(
        f"{printinfo}"
        f"score_all: {score:.4f}, score1: {score1:.4f}, score2: {score2:.4f}, "
        f"conf_p1: {conf_p1_avg:.4f}, conf_p2: {conf_p2_avg:.4f}, conf_avg: {conf_avg:.4f}, "
        f"CE_p1: {CE_pos:.4f}, CE_p2: {CE_neg:.4f}, CE_avg: {CE_avg:.4f} "
    )
    return score


def max_info_gain(
    model_name,
    model,
    tokenizer,
    inputs,
    labels,
    embs,
    selected_indices,
    uncertain_indices,
    hist_indices,
    sample_indices,
    labeled_eval,
    probs,
    args,
):
    candidate_indices = list(uncertain_indices) + list(hist_indices)

    inputs_of_E, labels_of_E, embs_of_E = [], [], []
    for idx in selected_indices:
        inputs_of_E.append(inputs[idx])
        labels_of_E.append(labels[idx])
        embs_of_E.append(embs[idx])

    sample_inputs, sample_embs = [], []
    for idx in sample_indices:
        sample_inputs.append(inputs[idx])
        sample_embs.append(embs[idx])

    labeled_inputs, labeled_embs, labeled_labels = [], [], []
    labeled_eval = list(labeled_eval)
    for idx in labeled_eval:
        labeled_inputs.append(inputs[idx])
        labeled_embs.append(embs[idx])
        labeled_labels.append(labels[idx])
    labeled_labels = np.array(labeled_labels)
    print("labeled_labels", labeled_labels)

    # calculate information gain of each uncertain sample
    score_max = cal_info_score(
        probs[sample_indices],
        sample_indices,
        probs[labeled_eval],
        labeled_eval,
        labeled_labels,
    )
    best_sample = None
    info_gain = []
    for i, idx in enumerate(candidate_indices):
        inputs_of_E.append(inputs[idx])
        labels_of_E.append(labels[idx])
        embs_of_E.append(embs[idx])
        # part 1: predictions on the unlabeled data
        prompts = construct_prompt(
            inputs_of_E,
            labels_of_E,
            embs_of_E,
            sample_inputs,
            sample_embs,
            args,
        )
        _, probs1 = inference(model_name, model, tokenizer, prompts, args)
        # part 2: predictions on the labeled data
        if len(labeled_eval) > 0:
            prompts = construct_prompt(
                inputs_of_E,
                labels_of_E,
                embs_of_E,
                labeled_inputs,
                labeled_embs,
                args,
            )
            _, labeled_probs = inference(model_name, model, tokenizer, prompts, args)
        else:
            labeled_probs = []
        # calculate information gain
        score = cal_info_score(
            probs1,
            sample_indices,
            labeled_probs,
            labeled_eval,
            labeled_labels,
            idx,
            probs[idx],
            labels[idx],
        )
        if score_max < score:
            score_max = score
            best_sample = idx
        info_gain.append(score)
        del inputs_of_E[-1]
        del labels_of_E[-1]
        del embs_of_E[-1]
        if best_sample is not None and len(info_gain) >= len(uncertain_indices):
            break
    print(f"candidate_indices: {uncertain_indices}")
    print(f"info_gain: {np.round(np.array(info_gain), 4)}")
    # normalize info_dict
    info_rank = np.array(rankdata(info_gain)) / len(info_gain)
    info_dict = {}
    for i, rank in enumerate(info_rank):
        info_dict[candidate_indices[i]] = rank
    print(f"info_dict: {info_dict}")
    return best_sample, info_dict


def select_by_cosine_sim(model_name, model, tokenizer, inputs, labels, embs, args):
    # sample by cosine similarity
    cosine_of_each_pair = cal_cosine_sim(args)
    inputs, labels, embs, candidate_indices, scores = stratified_sampling(
        inputs, labels, embs, cosine_of_each_pair, args
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


def determine_next_target(labels_of_E, selected_indices, probs, labeled_set, labels):
    pos_neg_diff = 2 * np.sum(labels_of_E) - len(labels_of_E)
    if pos_neg_diff < 0:
        next = 1
    elif pos_neg_diff > 0:
        next = 0
    else:
        if np.sum(probs > 0.5) == np.sum(labels_of_E):
            next = 1
        elif np.sum(probs <= 0.5) == len(labels_of_E) - np.sum(labels_of_E):
            next = 0
        else:
            labeled_eval = list(labeled_set - set(selected_indices))
            if len(labeled_eval) == 0:
                next = 1
            else:
                eval_labels = np.array(labels)[labeled_eval]
                eval_pseudo = np.array(probs[labeled_eval] > 0.5, dtype=int)
                scores = (eval_labels == eval_pseudo).astype(int)
                pos_avg = np.mean(scores[eval_labels == 1])
                neg_avg = np.mean(scores[eval_labels == 0])
                if np.sum(eval_labels) == 0:
                    pos_avg = 0.5
                if np.sum(eval_labels) == len(eval_labels):
                    neg_avg = 0.5
                if pos_avg > neg_avg:
                    next = 0
                else:
                    next = 1
                print(f"pos_avg: {pos_avg:.4f}, neg_avg: {neg_avg:.4f} => next: {next}")
    return next


def ideal(model_name, model, tokenizer, inputs, labels, embs, args):
    # sample by cosine similarity
    cosine_of_each_pair = cal_cosine_sim(args)
    inputs, labels, embs, candidate_indices, scores = stratified_sampling(
        inputs, labels, embs, cosine_of_each_pair, args
    )
    unselected_indices = list(range(len(labels)))
    print(f"candidates: {len(inputs)}")
    # warm up with the sample with the highest and lowest cosine similarity score
    inputs_of_E, labels_of_E, embs_of_E = [], [], []
    selected_indices = [np.argmax(scores), np.argmin(scores)]
    for idx in selected_indices:
        inputs_of_E.append(inputs[idx])
        labels_of_E.append(labels[idx])
        embs_of_E.append(embs[idx])
        del unselected_indices[unselected_indices.index(idx)]

    # iterative sampling by confidence
    historical_info = [defaultdict(list), defaultdict(list)]
    historical_probs = []
    labeled_set = set(selected_indices)

    no_improvement = 0
    while len(selected_indices) < args.k:
        print(f"\n\n****************iteration {len(selected_indices) + 1}")

        if no_improvement == 0:
            # LLM's predictions based on selected examples $\mathbf{E}$
            inputs_of_U, embs_of_U, labels_of_U = [], [], []
            for idx in unselected_indices:
                inputs_of_U.append(inputs[idx])
                embs_of_U.append(embs[idx])
                labels_of_U.append(labels[idx])
            prompts = construct_prompt(
                inputs_of_E, labels_of_E, embs_of_E, inputs_of_U, embs_of_U, args
            )
            print(prompts[0])
            print("---------------\n\n")
            _, probs = inference(model_name, model, tokenizer, prompts, args)
            probs_output = [f"{v:.4f}" for v in probs]
            result = [v for v in zip(unselected_indices, labels_of_U, probs_output)]
            print(result)

            # evaluate the performance of the current model
            precision, recall, f1 = evaluate(labels_of_U, np.array(probs) > 0.5)
            print(f"[Eval]: Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f}")

            # update the historical information
            probs = np.clip(np.array(probs), 1e-6, 1 - 1e-6)
            probs_N = np.ones(len(labels)) * 2
            probs_N[unselected_indices] = probs
            historical_probs.append(probs_N)  # [T, N]

        # Selection of the next example
        ## budget allocation
        # if len(selected_indices) < 4:
        #     args.beam_size = 2 * args.budget // args.k - 1
        # else:
        ### next target
        target = determine_next_target(
            labels_of_E, selected_indices, probs_N, labeled_set, labels
        )
        for next in [target, 1 - target]:
            args.beam_size = np.ceil(
                (args.budget - len(labeled_set)) / (args.k - len(selected_indices))
            ).astype(int)
            print(
                f"remaining budget: {args.budget - len(labeled_set)}, "
                f"beam size: {args.beam_size}, next: {next}"
            )
            idx, labeled_indices, historical_info = select_next(
                model_name,
                model,
                tokenizer,
                inputs,
                labels,
                embs,
                historical_probs,
                selected_indices,
                labeled_set,
                historical_info,
                args,
                next=next,
                no_improvement=no_improvement,
            )
            labeled_set = labeled_set.union(set(labeled_indices))
            if idx is not None:
                break

        # update the variables
        if idx is not None:
            del unselected_indices[unselected_indices.index(idx)]
            selected_indices.append(idx)
            inputs_of_E.append(inputs[idx])
            labels_of_E.append(labels[idx])
            embs_of_E.append(embs[idx])

            print(f"index: {idx}, label: {labels[idx]}, pred: {probs_N[idx]:.2f}")
            print(
                f"****************\n"
                f"selected indices: {selected_indices}\n"
                f"labels of selected: {labels_of_E}"
            )
            no_improvement = 0
        else:
            no_improvement += 1
            if len(labeled_set) == args.budget:
                print(f"!!! Run out of the budget before selecting the enough examples")
                break

        # unselected_indices, _ = sampling(
        #     historical_probs,
        #     unselected_indices,
        #     None,
        #     n=max(args.budget * 5, len(unselected_indices) // 2),
        #     type="covered_by_rep",
        # )
        # unselected_indices = list(
        #     set(unselected_indices).union(set(labeled_set)) - set(selected_indices)
        # )
        # unselected_indices = list(unselected_indices)

    selected_indices = [candidate_indices[idx] for idx in selected_indices]
    return selected_indices
