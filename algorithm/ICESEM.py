from collections import defaultdict
import numpy as np
from scipy.spatial import distance_matrix
from utils.prompt import construct_prompt
from utils.inference import inference
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math
from utils.misc import evaluate, MFL_l1
from scipy.stats import rankdata
from sklearn.metrics import f1_score
import time
from copy import deepcopy


def select_next(
    model_name,
    model,
    tokenizer,
    inputs,
    labels,
    embs,
    historical_probs,
    probs_N,
    selected_indices,
    labeled_set,
    eval_indices,
    args,
    next=1,
    no_improvement=False,
    weights_dict={},
):
    probs = np.array(historical_probs[-1])
    for idx in selected_indices:
        probs[idx] = 2
    labels = np.array(labels)
    cond1 = (probs > 0.5) & (probs <= 1)
    cond2 = probs <= 0.5
    cond3 = probs <= 1
    imp_rate = 1.1
    pred_dict = {}

    if args.sep_sample:
        uncertain_indices_p1 = np.where(cond1)[0]
        uncertain_indices_p2 = np.where(cond2)[0]
        if len(uncertain_indices_p1) > args.beam_size // 2:
            uncertain_indices_p1, _ = sampling(
                historical_probs,
                uncertain_indices_p1,
                [[], labeled_set],
                n=args.beam_size // 2,
                sample_type="covered_by_rep",
            )
        if len(uncertain_indices_p2) > args.beam_size - len(uncertain_indices_p1):
            uncertain_indices_p2, _ = sampling(
                historical_probs,
                uncertain_indices_p2,
                [[], labeled_set],
                n=args.beam_size - len(uncertain_indices_p1),
                sample_type="covered_by_rep",
            )
        uncertain_indices = np.concatenate([uncertain_indices_p1, uncertain_indices_p2])
        uncertain_indices = list(uncertain_indices.astype(int))
    else:
        uncertain_indices = np.where(cond3)[0]
        if len(uncertain_indices) > args.beam_size and args.beam_size > 0:
            uncertain_indices_p1, weights_dict = sampling(
                historical_probs,
                uncertain_indices,
                [[], labeled_set],
                n=args.beam_size,
                sample_type="covered_by_rep",
            )
            labeled_set = labeled_set.union(set(uncertain_indices_p1))
            if (
                np.sum(labels[list(labeled_set - set(selected_indices))]) < 2
                and args.beam_size > 0
            ):
                uncertain_indices_p2, weights_dict = sampling(
                    historical_probs,
                    np.where(cond1)[0],
                    [[], labeled_set],
                    n=(args.beam_size // 2),
                    sample_type="covered_by_rep",
                )
                uncertain_indices = np.concatenate(
                    [uncertain_indices_p1, uncertain_indices_p2]
                ).astype(int)
            else:
                uncertain_indices = uncertain_indices_p1
        elif args.beam_size == 0:
            uncertain_indices = []
    print(f"uncertain_indices: {list(uncertain_indices)}")
    labeled_eval = labeled_set.union(set(uncertain_indices)) - set(selected_indices)
    print("labeled_eval", labeled_eval)
    # print(f"weights_dict {weights_dict}")
    print(f"### eval indices: {list(eval_indices)}")

    index, imp_rate, pred_dict = max_info_gain(
        model_name,
        model,
        tokenizer,
        inputs,
        labels,
        embs,
        selected_indices,
        list(uncertain_indices),
        weights_dict,
        eval_indices,
        labeled_eval,
        probs_N,
        next,
        no_improvement,
        args,
    )
    print(f"selected_index = {index}, imp_rate = {imp_rate:.4f}")
    return index, uncertain_indices, imp_rate, pred_dict, weights_dict


def sampling(
    historical_probs,
    indices,
    labeled_info,  # [top_indices, labeled_set]
    n=10,
    sample_type="MFL_whole",
):
    if n == 0:
        return [], []
    weights = np.ones(len(indices))
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
    if sample_type == "Kmeans":
        kmeans = KMeans(n_clusters=n, random_state=0).fit(candidate_reps)
        selected_indices = []
        for i in range(n):
            dist = distance_matrix([kmeans.cluster_centers_[i]], candidate_reps, p=1)
            dist = dist[0]
            selected_indices.append(indices[np.argmin(dist)])
        return selected_indices, np.ones(len(selected_indices))
    elif sample_type == "MFL":
        selected_indices = MFL_l1(candidate_reps, n)
    elif sample_type == "coverage":
        selected_indices = coverage(indices, candidate_probs, candidate_reps, n)
    elif sample_type == "coverage_by_conf":
        selected_indices = coverage_by_conf(candidate_probs, candidate_reps, n)
    elif sample_type == "covered_by_rep":
        selected_indices, weights = coverage_by_rep(
            candidate_probs,
            candidate_reps,
            n + len(labeled_indices_new),
            selected_indices,
            labeled_indices_new,
        )
    if sample_type == "coverage":
        selected_indices = np.array(selected_indices)
    else:
        selected_indices = np.array(indices)[selected_indices]
    if type(weights) == dict:
        weights_dict = {indices[idx]: weights[idx] for idx in weights}
    else:
        weights_dict = {}
    selected_indices = np.array(selected_indices).astype(int)
    print(f"[{type}] selected_indices: {list(selected_indices)}")
    print(f"[{type}] probs: {list(np.round(probs[selected_indices], 4))}")
    return selected_indices, weights_dict


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
    dist = distance_matrix(reps, reps, p=2)  # [N, N]
    sample_list = []
    max_score, max_cover_range = 0, 0
    confs = conf_func(probs)
    org_conf_range = np.max(confs) - np.min(confs)
    print(
        f"budget: {n}, labeled_indices: {labeled_indices}, "
        f"org_conf_range: {org_conf_range:.4f}, "
        f"max_prob: {np.max(probs):.4f}, min_prob: {np.min(probs):.4f}"
    )
    m_lower, m_upper = 0, 1
    best_m = 1
    m_list, dist_list = [], []
    while m_lower <= m_upper:
        m = (m_lower + m_upper) / 2
        dist_thr = np.percentile(dist, 100.0 * m)
        m_list.append(f"{m:.4f}")
        dist_list.append(f"{dist_thr:.4f}")
        graph = {}
        for i in range(len(probs)):
            graph[i] = []
            for j in range(len(probs)):
                if dist[i, j] <= dist_thr:
                    graph[i].append(j)
        covered_set = set(list(labeled_indices))
        selected_indices = list(labeled_indices)
        covered_num = {}
        for idx in selected_indices:
            # covered_num[idx] = len(set(graph[idx]) - set(covered_set))
            covered_num[idx] = len(set(graph[idx]))
            covered_set = covered_set.union(set(graph[idx]))
        if len(sample_list) == 0:
            print(f"selected_indices: {selected_indices}")
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
            # covered_num[idx] = len(set(graph[idx]) - set(covered_set))
            covered_num[idx] = len(set(graph[idx]))
            covered_set = covered_set.union(set(graph[idx]))
            selected_indices.append(idx)
        score = len(selected_indices) / n + len(covered_set) / len(probs)
        if max_score < score or max_score == score and m < best_m:
            max_score = score
            max_cover_ratio = len(covered_set) / len(probs)
            sample_list = list(selected_indices)
            best_m = m
            best_dist_thr = dist_thr
            covered_of_best = deepcopy(covered_num)
        if len(selected_indices) < n:
            m_upper = m - 0.001
        elif len(covered_set) < len(probs):
            m_lower = m + 0.001
        else:
            m_upper = m - 0.001
    median_thr = np.median(list(covered_of_best.values()))
    for idx in sample_list:
        covered_set = set()
        for j in sample_list:
            if j != idx:
                covered_set = covered_set.union(set(graph[j]))
        margin_covered = len(set(graph[idx]) - set(covered_set))
        covered_of_best[idx] = int(
            (min(covered_of_best[idx], median_thr) + max(1, margin_covered)) // 2
        )
    print(f"m_list: {m_list}, dist_list: {dist_list}")
    print(
        f"best_m: {best_m:.4f}, best_dist_thr: {best_dist_thr:.4f}, "
        f"covered_ratio: {max_cover_ratio:.4f} "
        f" sample_list: {sample_list}"
        f" covered_of_best: {covered_of_best}"
    )
    # if len(sample_list) < n:
    #     print(f"remaining budget: {n - len(sample_list)}")
    #     print(f"sample_list (before): {len(sample_list)}")
    #     sample_list += np.random.choice(
    #         list(set(range(len(probs))) - set(sample_list)),
    #         n - len(sample_list),
    #         replace=False,
    #     ).tolist()
    sample_list = list(set(sample_list) - set(labeled_indices))
    return sample_list, covered_of_best


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
    # if added_idx in sample_indices:
    #     index = list(sample_indices).index(added_idx)
    #     if len(conf) == 1:
    #         return 1
    #     conf_avg = (np.sum(conf) - conf[index]) / (len(conf) - 1)
    # else:
    conf_avg = np.mean(conf)
    return conf_avg


def cal_score1(probs1, indices1, selected_index=-1):
    # cal certainty score
    if len(probs1) == 0:
        conf_p1_avg, conf_p2_avg, conf_avg, score1 = 1.0
    else:
        probs1 = np.array(probs1)
        conf = conf_func(probs1)
        conf_p1_avg = cal_avg(conf, selected_index, indices1, probs1, "pos")
        conf_p2_avg = cal_avg(conf, selected_index, indices1, probs1, "neg")
        conf_avg = cal_avg(conf, selected_index, indices1, probs1, "all")
        score1 = conf_avg / 2 + min(conf_p1_avg, conf_p2_avg) / 2
    return score1, conf_p1_avg, conf_p2_avg, conf_avg


def cal_score2(probs, indices, labels, selected_index=-1, metric="f1", weights_dict={}):
    probs2, indices2, labels2 = [], [], []
    for i, idx in enumerate(indices):
        probs2.extend([probs[i]] * weights_dict.get(idx, 1))
        indices2.extend([idx] * weights_dict.get(idx, 1))
        labels2.extend([labels[i]] * weights_dict.get(idx, 1))
    if metric == "acc":
        # cal accuracy
        if len(probs2) == 0:
            CE_pos, CE_neg, CE_avg, score2 = 1, 1, 1, 1
        else:
            probs2 = np.array(probs2)
            CE_scores = labels2 == (probs2 > 0.5)
            CE_pos = cal_avg(CE_scores, selected_index, indices2, labels2, "pos")
            CE_neg = cal_avg(CE_scores, selected_index, indices2, labels2, "neg")
            CE_avg = cal_avg(CE_scores, selected_index, indices2, labels2, "all")
            score2 = CE_avg / 2 + min(CE_pos, CE_neg) / 2
            if CE_pos == 0:
                score2 = 0
        return score2, CE_pos, CE_neg, CE_avg
    else:
        # cal precision, recall, f1
        if len(probs2) == 0:
            pre, rec, f1, score2 = 1, 1, 1, 1
        else:
            probs2 = np.array(probs2)
            if np.sum(labels2) > 0:
                pred = probs2 > 0.5
            else:
                pred = probs2 <= 0.5
                labels2 = 1 - np.array(labels2)
            pre = np.sum(pred * labels2) / (np.sum(pred) + 1e-6)
            rec = np.sum(pred * labels2) / (np.sum(labels2) + 1e-6)
            f1 = 2 * pre * rec / (pre + rec + 1e-6)
            if metric == "f1":
                score2 = f1
            else:
                score2 = f1 / 2 + min(pre, rec) / 2
        return score2, pre, rec, f1


def cal_info_score(
    probs1,
    indices1,
    probs2,
    indices2,
    labels2,
    selected_index=-1,
    p=-1,
    label=-1,
    weights_dict={},
    metric="f1",
):
    score1, conf_p1_avg, conf_p2_avg, conf_avg = cal_score1(probs1, indices1)
    score2, p1, p2, avg = cal_score2(
        probs2, indices2, labels2, selected_index, metric, weights_dict
    )
    score = 0.5 * score1 + score2

    if selected_index == -1:
        printinfo = f"[Base] "
    else:
        printinfo = (
            f"[New] newly added index: {selected_index}, predicted_prob: {p:.4f}, "
            f"ground truth label: {label}, "
        )
    printinfo += f"score_all: {score:.4f}, score1: {score1:.4f}, score2: {score2:.4f}, "
    printinfo += f"conf_p1: {conf_p1_avg:.4f}, conf_p2: {conf_p2_avg:.4f}, conf_avg: {conf_avg:.4f}, "
    printinfo += f"avg: {avg:.4f}, p1: {p1:.4f}, p2: {p2:.4f}, "
    print(f"{printinfo}")
    return score, score1, score2


def sort_labeled_eval(labeled_eval_org, probs, labels):
    def intertwine(a, b):
        if len(a) == 0:
            return list(b)
        if len(b) == 0:
            return list(a)
        num = len(a) + len(b)
        if len(a) > len(b):
            a, b = b, a
        rate = min(round(num / len(a)), 3)
        c = []
        a, b = list(a), list(b)
        for i in range(num):
            if i % rate == 0 and len(a) > 0:
                c.append(a.pop(0))
            elif len(b) > 0:
                c.append(b.pop(0))
        c = c + a + b
        return c

    # sort labeled_eval_org by difficulty
    if len(labeled_eval_org) > 0:
        labeled_pos_easy, labeled_pos_hard = [], []
        labeled_neg_easy, labeled_neg_hard = [], []
        for idx in labeled_eval_org:
            if labels[idx] == 1:
                if probs[idx] > 0.5:
                    labeled_pos_easy.append(idx)
                else:
                    labeled_pos_hard.append(idx)
            else:
                if probs[idx] <= 0.5:
                    labeled_neg_easy.append(idx)
                else:
                    labeled_neg_hard.append(idx)
        # sort by difficulty
        labeled_eval_org = []
        labeled_pos_easy = np.array(labeled_pos_easy)[
            np.argsort(probs[labeled_pos_easy])
        ].astype(int)
        labeled_pos_hard = np.array(labeled_pos_hard)[
            np.argsort(-probs[labeled_pos_hard])
        ].astype(int)
        labeled_neg_easy = np.array(labeled_neg_easy)[
            np.argsort(-probs[labeled_neg_easy])
        ].astype(int)
        labeled_neg_hard = np.array(labeled_neg_hard)[
            np.argsort(probs[labeled_neg_hard])
        ].astype(int)
        print(
            f"debug!!! labeled_eval_org: {labeled_eval_org} "
            f"labeled_pos_easy: {labeled_pos_easy} labeled_neg_easy: {labeled_neg_easy} "
            f"labeled_pos_hard: {labeled_pos_hard} labeled_neg_hard: {labeled_neg_hard}"
        )
        labeled_pos = intertwine(labeled_pos_easy, labeled_pos_hard)
        labeled_neg = intertwine(labeled_neg_easy, labeled_neg_hard)
        labeled_eval_org = intertwine(labeled_pos, labeled_neg)
        print(
            f"labeled_eval_org: {labeled_eval_org}, prob: {list(probs[labeled_eval_org])}, "
            f"label: {list(labels[labeled_eval_org])}"
        )
    else:
        labeled_eval_org = []
    return labeled_eval_org


def get_weights_dict(probs, candidate_indices):
    probs = np.array(probs)
    probs_N = deepcopy(probs)
    probs = probs[probs < 2]
    # # partition probs into bins
    # bins = np.linspace(np.min(probs), np.max(probs) + 1e-6, 6)
    # org_num, labeled_num = {}, {}
    # for i in range(len(bins) - 1):
    #     org_num[i] = np.sum((probs >= bins[i]) & (probs < bins[i + 1]))
    #     labeled_num[i] = 0
    #     for idx in candidate_indices:
    #         if probs_N[idx] >= bins[i] and probs_N[idx] < bins[i + 1]:
    #             labeled_num[i] += 1
    # weights_dict = {}
    # for i in range(len(bins) - 1):
    #     for idx in candidate_indices:
    #         if probs_N[idx] >= bins[i] and probs_N[idx] < bins[i + 1]:
    #             weights_dict[idx] = round(org_num[i] / (labeled_num[i] + 1e-6))
    indices_probs = list(zip(candidate_indices, probs_N[candidate_indices]))
    indices, probs_of_indices = zip(*sorted(indices_probs, key=lambda x: x[1]))
    probs_bounds = [0]
    for i in range(len(probs_of_indices) - 1):
        probs_bounds.append((probs_of_indices[i] + probs_of_indices[i + 1]) / 2)
    probs_bounds.append(1)
    weights_dict = {}
    for i, idx in enumerate(indices):
        weights_dict[idx] = np.sum(
            (probs >= probs_bounds[i]) & (probs < probs_bounds[i + 1])
        )
    return weights_dict


def max_info_gain(
    model_name,
    model,
    tokenizer,
    inputs,
    labels,
    embs,
    selected_indices,
    uncertain_indices,
    weights_dict,
    sample_indices_org,
    labeled_eval_org,
    probs_N,
    next,
    no_improvement,
    args,
):
    if type(weights_dict) != dict:
        weights_dict = {}
    pos_num = np.sum(labels[selected_indices])
    neg_num = len(selected_indices) - pos_num
    pos_scalar = (2 * args.k + neg_num) / (2 * args.k + pos_num)
    neg_scalar = 1
    scalar_dict = {1: pos_scalar, 0: neg_scalar}
    # scalar_dict = {1: 0.5, 0: 0.5}
    print(f"scalar_dict: {scalar_dict}")
    pred_dict = {}
    labeled_eval_org = list(labeled_eval_org)
    hist_indices = []
    if no_improvement == False:
        for idx in labeled_eval_org:
            if (
                next == 2 or labels[idx] == next
            ) and idx not in selected_indices + uncertain_indices:
                hist_indices.append(idx)
    else:
        hist_indices = []
    candidate_indices = list(uncertain_indices) + list(hist_indices)
    candidate_indices = [int(v) for v in candidate_indices]
    if len(candidate_indices) == 0:
        return None, -1, {}

    inputs_of_E, labels_of_E, embs_of_E = [], [], []
    for idx in selected_indices:
        inputs_of_E.append(inputs[idx])
        labels_of_E.append(labels[idx])
        embs_of_E.append(embs[idx])

    sample_inputs, sample_embs = [], []
    for idx in range(len(probs_N)):
        if idx in candidate_indices and probs_N[idx] == 2:
            sample_inputs.append(inputs[idx])
            sample_embs.append(embs[idx])

    # pred the newly selected examples
    probs = np.array(probs_N)
    print(f"debug!!! predict newly added: {len(sample_inputs)}")
    if len(sample_inputs) > 0:
        prompts = construct_prompt(
            inputs_of_E,
            labels_of_E,
            embs_of_E,
            sample_inputs,
            sample_embs,
            args,
        )
        _, probs1 = inference(model_name, model, tokenizer, prompts, args)
        for idx in range(len(probs_N)):
            if idx in candidate_indices and probs_N[idx] == 2:
                probs[idx] = probs1.pop(0)
    weights_dict = get_weights_dict(probs, labeled_eval_org)
    print(f"[weights_dict]: {weights_dict}")
    labeled_eval_org = sort_labeled_eval(labeled_eval_org, probs, labels)

    if len(candidate_indices) > 3:
        T = 3
    else:
        T = 1
    tau = round(len(candidate_indices) ** (1.0 / T))
    print(f"********* tau: {tau}")
    unlabeled_pred, labeled_pred = {}, {}
    sample_indices_last, labeled_eval_last = [], []
    for t in range(1, T + 1):
        pred_dict_collections, imp_rate_collections, score_collections = {}, {}, {}
        un_eval_size = max(10, np.round(args.eval_size / tau ** (T - t)).astype(int))
        labeled_eval_size = max(
            10, np.round(len(labeled_eval_org) / tau ** (T - t)).astype(int)
        )
        sample_indices = list(sample_indices_org)[:un_eval_size]
        labeled_eval = list(labeled_eval_org)[:labeled_eval_size]
        labeled_labels = np.array(labels)[labeled_eval]
        print(
            f"\n**********[Eval] iteration {t} candidates: {len(candidate_indices)} "
            f" un_eval_size: {un_eval_size}, labeled_eval_size: {labeled_eval_size}"
            f" labeled_labels: {np.sum(labeled_labels)}/{len(labeled_labels)}"
        )

        sample_inputs, sample_embs = [], []
        for idx in sample_indices[len(sample_indices_last) :]:
            sample_inputs.append(inputs[idx])
            sample_embs.append(embs[idx])

        labeled_inputs, labeled_embs = [], []
        for idx in labeled_eval[len(labeled_eval_last) :]:
            labeled_inputs.append(inputs[idx])
            labeled_embs.append(embs[idx])

        sample_indices_last = list(sample_indices)
        labeled_eval_last = list(labeled_eval)

        # calculate information gain of each uncertain sample
        score_base, score1_base, score2_base = cal_info_score(
            probs[sample_indices],
            sample_indices,
            probs[labeled_eval],
            labeled_eval,
            labeled_labels,
            weights_dict=weights_dict,
            metric=args.metric,
        )

        # part 2: predictions on the labeled data
        score2_dict, score2_info = {}, {}
        for i, idx in enumerate(candidate_indices):
            inputs_of_E.append(inputs[idx])
            labels_of_E.append(labels[idx])
            embs_of_E.append(embs[idx])
            if len(labeled_inputs) > 0:
                prompts = construct_prompt(
                    inputs_of_E,
                    labels_of_E,
                    embs_of_E,
                    labeled_inputs,
                    labeled_embs,
                    args,
                )
                _, labeled_probs = inference(
                    model_name, model, tokenizer, prompts, args
                )
            else:
                labeled_probs = []
            if t > 1:
                labeled_probs = np.concatenate([labeled_pred[idx], labeled_probs])
            labeled_pred[idx] = deepcopy(labeled_probs)
            score2_p1, p1, p2, avg = cal_score2(
                labeled_probs,
                labeled_eval,
                labeled_labels,
                idx,
                metric=args.metric,
                weights_dict=weights_dict,
            )
            if idx in labeled_eval:
                labeled_probs[labeled_eval.index(idx)] = probs[idx]
            score2_p2, p1, p2, avg = cal_score2(
                labeled_probs,
                labeled_eval,
                labeled_labels,
                idx,
                metric=args.metric,
                weights_dict=weights_dict,
            )
            score2_dict[idx] = (score2_p1 + score2_p2) / 2
            score2_info[idx] = (f"{p1:.4f}", f"{p2:.4f}", f"{avg:.4f}")
            del inputs_of_E[-1]
            del labels_of_E[-1]
            del embs_of_E[-1]

        pruned_ratio = 1 - 2.0 / tau
        if len(candidate_indices) * pruned_ratio > 1:
            score2_thr = np.percentile(list(score2_dict.values()), pruned_ratio * 100)
            score2_thr = score2_thr - 1e-3
        else:
            score2_thr = 0
        if t == T:
            score2_thr = max(score2_base * 0.99, score2_thr)
        print(f"pruned_ratio: {pruned_ratio:.4f}, score2_thr: {score2_thr:.4f}")

        for i, idx in enumerate(candidate_indices):
            if score2_dict[idx] < score2_thr:
                score_collections[idx] = (
                    score2_dict[idx] * scalar_dict[labels[idx]] * 0.5
                )
                imp_rate_collections[idx] = score2_dict[idx] / score2_base - 1
                print(f"pruned index {idx}, score: {score_collections[idx]:.4f}")
                continue
            inputs_of_E.append(inputs[idx])
            labels_of_E.append(labels[idx])
            embs_of_E.append(embs[idx])
            # part 1: predictions on the unlabeled data
            if len(sample_inputs) > 0:
                prompts = construct_prompt(
                    inputs_of_E,
                    labels_of_E,
                    embs_of_E,
                    sample_inputs,
                    sample_embs,
                    args,
                )
                _, probs1 = inference(model_name, model, tokenizer, prompts, args)
            else:
                probs1 = []
            if t > 1:
                probs1 = np.concatenate([unlabeled_pred[idx], probs1])
            unlabeled_pred[idx] = probs1
            score1 = cal_score1(probs1, sample_indices)[0]
            score = 0.5 * score1 + score2_dict[idx]
            score = score * scalar_dict[labels[idx]]

            pred_dict_collections[idx] = {}
            for j, index in enumerate(sample_indices):
                pred_dict_collections[idx][index] = probs1[j]
            for j, index in enumerate(labeled_eval):
                pred_dict_collections[idx][index] = labeled_pred[idx][j]
            score1_imp_rate = (score1 + 1e-6) / (score1_base + 1e-6) - 1
            score2_imp_rate = (score2_dict[idx] + 1e-6) / (score2_base + 1e-6) - 1
            score_imp_rate = score / scalar_dict[labels[idx]] / score_base - 1
            if score2_imp_rate < -0.01:
                imp_rate = score2_imp_rate
            elif score1_imp_rate < -0.02:
                imp_rate = score1_imp_rate * 0.5
            else:
                imp_rate = score_imp_rate
            imp_rate_collections[idx] = imp_rate
            if imp_rate < -0.01 and t == T:
                score_collections[idx] = -1
            else:
                score_collections[idx] = score
            print(
                f"added index {idx}, prob: {probs[idx]:.4f}, label: {labels[idx]}, "
                f"scalar: {scalar_dict[labels[idx]]:.4f}, score: {score:.4f}, score1: {score1:.4f}, "
                f"score2: {score2_dict[idx]:.4f}, score2_info: {list(score2_info[idx])} "
                f"imp_rate: {imp_rate:.4f}, score_final: {score_collections[idx]:.4f}"
            )
            del inputs_of_E[-1]
            del labels_of_E[-1]
            del embs_of_E[-1]
        # top 5
        top_indices, _ = zip(
            *sorted(score_collections.items(), key=lambda x: x[1], reverse=True)
        )
        candidate_indices = top_indices[: np.ceil(len(top_indices) / tau).astype(int)]
        candidate_indices = list(candidate_indices)
        score2_thr = score2_dict[candidate_indices[0]]
        for idx in score2_dict:
            if (
                score2_dict[idx] >= score2_thr
                and imp_rate_collections.get(idx, 0) > 0
                and idx not in candidate_indices
                and idx not in labeled_eval
                and score2_thr < 1
            ):
                candidate_indices.append(idx)
        if len(candidate_indices) < 2:
            print(f"Not too much candidates left for comparison!!!")
            break
    candidate_indices, scaled_imp_rates = [], []
    for idx, imp_rate in imp_rate_collections.items():
        if imp_rate < 0:
            continue
        candidate_indices.append(idx)
        scaled_imp_rates.append(imp_rate * scalar_dict[labels[idx]])
    if (
        len(scaled_imp_rates) > 1
        and len(selected_indices) < args.k - 1
        and args.argmax == False
    ):
        scaled_imp_rates = np.array(scaled_imp_rates)
        scaled_imp_rates = scaled_imp_rates / max(np.min(scaled_imp_rates), 0.01)
        scaled_imp_rates = np.exp(scaled_imp_rates)
        selected_probs = np.array(scaled_imp_rates) / np.sum(scaled_imp_rates)
        best_sample = np.random.choice(candidate_indices, p=selected_probs)
        print(
            f"indices with imp_rate > 0: {candidate_indices}, "
            f"scaled_imp_rates: {list(scaled_imp_rates)}, "
            f"selected_probs: {list(selected_probs)}"
        )
    else:
        best_sample = max(score_collections, key=score_collections.get)
    best_imp_rate = imp_rate_collections.get(best_sample, -1)
    pred_dict = pred_dict_collections.get(best_sample, {})
    return best_sample, best_imp_rate, pred_dict


def determine_next_target(labels_of_E, selected_indices, probs, labeled_set, labels):
    pos_neg_diff = 2 * np.sum(labels_of_E) - len(labels_of_E)
    if pos_neg_diff < 0:
        next = 1
    elif pos_neg_diff > 0:
        next = 0
    else:
        next = 2
    return next


def warm_up(unselected_indices, scores, labels):
    # # warm up with the sample with the highest and lowest cosine similarity score
    # labeled_set = set()
    # indices_by_scores = np.argsort(scores)[::-1]
    # for i in range(len(scores)):
    #     labeled_set.add(indices_by_scores[i])
    #     if labels[indices_by_scores[i]] == 1:
    #         selected_indices = [indices_by_scores[i]]
    #         break
    # for i in range(len(scores) - 1, -1, -1):
    #     labeled_set.add(indices_by_scores[i])
    #     if labels[indices_by_scores[i]] == 0:
    #         selected_indices.append(indices_by_scores[i])
    #         break
    # warm up with cosine similarity score
    labeled_set = set()
    unselected_indices, scores = zip(
        *sorted(zip(unselected_indices, scores), key=lambda x: x[1])
    )
    # select the pos
    l, r = int(len(unselected_indices) / 50 * 49), len(unselected_indices) - 1
    while l <= r:
        labeled_set.add(unselected_indices[r])
        if labels[unselected_indices[r]] == 1:
            selected_indices = [unselected_indices[r]]
            break
        l = (l + r) // 2 + 1
    # select the neg
    l, r = 0, int(len(unselected_indices) / 5)
    while l <= r:
        labeled_set.add(unselected_indices[r])
        if labels[unselected_indices[r]] == 0:
            selected_indices.append(unselected_indices[r])
            break
        r = (l + r) // 2
    print(
        f"[warmp up], selected_indices: {selected_indices}, "
        f"labeled_set: {labeled_set}, labels: {list(np.array(labels)[list(labeled_set)])}"
    )
    return selected_indices, labeled_set


def ICESEM(model_name, model, tokenizer, inputs, labels, embs, scores, args):
    unselected_indices = list(range(len(labels)))
    print(f"candidates: {len(inputs)}")
    selected_indices, labeled_set = warm_up(unselected_indices, scores, labels)
    inputs_of_E, labels_of_E, embs_of_E = [], [], []
    for idx in selected_indices:
        inputs_of_E.append(inputs[idx])
        labels_of_E.append(labels[idx])
        embs_of_E.append(embs[idx])
        del unselected_indices[unselected_indices.index(idx)]

    # iterative sampling by confidence
    historical_probs = []
    imp_thr = 0.01
    last_pred = {}
    no_improvement = False
    start_time = time.time()
    no_sig_imp_list = []
    beam_size = np.ceil((args.budget - len(labeled_set)) / (args.k - 2)).astype(int)
    # beam_size = max(10, beam_size)
    weights_dict = {}
    while len(selected_indices) < args.k:
        print(f"\n\n****************iteration {len(selected_indices) + 1}")

        if len(no_sig_imp_list) == 0 and no_improvement == False:
            # LLM's predictions based on selected examples $\mathbf{E}$
            pred_indices, inputs_of_U, embs_of_U, labels_of_U = [], [], [], []
            print(f"debug+++ last_pred: {list(last_pred.keys())}")
            for idx in unselected_indices:
                if idx in last_pred:
                    continue
                pred_indices.append(idx)
                inputs_of_U.append(inputs[idx])
                embs_of_U.append(embs[idx])
                labels_of_U.append(labels[idx])
            print(f"debug+++ pred ratio {len(inputs_of_U)}/{len(unselected_indices)}")
            if len(inputs_of_U) > 0:
                prompts = construct_prompt(
                    inputs_of_E, labels_of_E, embs_of_E, inputs_of_U, embs_of_U, args
                )
                print(prompts[0])
                print("---------------\n\n")
                _, probs = inference(model_name, model, tokenizer, prompts, args)
                # update the historical information
                probs = list(np.clip(np.array(probs), 1e-6, 1 - 1e-6))
            else:
                probs = []
            probs_N = np.ones(len(labels)) * 2
            for idx in unselected_indices:
                if idx in last_pred:
                    probs_N[idx] = last_pred[idx]
                else:
                    probs_N[idx] = probs.pop(0)
            historical_probs.append(probs_N)  # [T, N]
            # evaluate the performance of the current model
            precision, recall, f1 = evaluate(
                np.array(labels)[unselected_indices],
                (probs_N[unselected_indices]) > 0.5,
            )
            print(f"[Eval]: Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f}")
            eval_indices, _ = sampling(
                historical_probs,
                np.where(probs_N < 2)[0],
                None,
                n=args.eval_size,
                sample_type="MFL",
            )
        else:
            probs_N = np.ones(len(labels)) * 2
            for idx in last_pred:
                probs_N[idx] = last_pred[idx]
            # update the predictions on the eval_indices
            sample_inputs, sample_embs = [], []
            for idx in range(len(probs_N)):
                if idx in eval_indices and probs_N[idx] == 2:
                    sample_inputs.append(inputs[idx])
                    sample_embs.append(embs[idx])

            # pred the newly selected examples
            print(f"debug!!! predict newly added: {len(sample_inputs)}")
            if len(sample_inputs) > 0:
                prompts = construct_prompt(
                    inputs_of_E,
                    labels_of_E,
                    embs_of_E,
                    sample_inputs,
                    sample_embs,
                    args,
                )
                _, probs1 = inference(model_name, model, tokenizer, prompts, args)
                for idx in range(len(probs_N)):
                    if idx in eval_indices and probs_N[idx] == 2:
                        probs_N[idx] = probs1.pop(0)
        probs_print = [f"{v:.4f}" for v in probs_N]
        print(f"{probs_print}")
        next = 2
        args.beam_size = min(beam_size, args.budget - len(labeled_set))
        print(
            f"remaining budget: {args.budget - len(labeled_set)}, "
            f"beam size: {args.beam_size}, next: {next}"
        )
        idx, labeled_indices, imp_rate, pred_dict, weights_dict = select_next(
            model_name,
            model,
            tokenizer,
            inputs,
            labels,
            embs,
            historical_probs,
            probs_N,
            selected_indices,
            labeled_set,
            eval_indices,
            args,
            next=next,
            no_improvement=no_improvement,
            weights_dict=weights_dict,
        )
        labeled_set = labeled_set.union(set(labeled_indices))
        print(
            f"==== index: {idx}, imp_rate: {imp_rate:.4f}, "
            f"Elapsed: {time.time()-start_time:.2f}s"
        )
        # update the variables
        no_sig_imp_list.append(imp_rate)
        cul_product = np.prod(np.array(no_sig_imp_list) + 1) - 1
        if (imp_rate > -0.01) and (len(no_sig_imp_list) == 1 or cul_product > 0):
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
            no_improvement = False
            # pred_diff = []
            # for k, v in pred_dict.items():
            #     if probs_N[k] == 2:
            #         continue
            #     if v > 0.5 and probs_N[k] <= 0.5 or v <= 0.5 and probs_N[k] > 0.5:
            #         pred_diff.append(1)
            #     else:
            #         pred_diff.append(0)
            # pred_diff = np.mean(pred_diff)
            # print(f"debug pred diff: {pred_diff:.4f}")
            pred_diff = 0
            last_pred = dict(pred_dict)
            if imp_rate > imp_thr or cul_product > imp_thr or pred_diff > imp_thr:
                no_sig_imp_list = []
        else:
            pred_diff = 0
            no_sig_imp_list.pop(-1)
            no_improvement = True
            if len(labeled_set) == args.budget:
                print(f"!!! Run out of the budget before selecting the enough examples")
                break
        print(
            f"debug!!! no_sig_imp_list: {list(no_sig_imp_list)} "
            f"cul_product: {cul_product:.4f} \n\n"
        )
    while len(no_sig_imp_list) > 0:
        cul_product = np.prod(np.array(no_sig_imp_list) + 1) - 1
        if cul_product < 0 or no_sig_imp_list[-1] < 0:
            selected_indices.pop(-1)
            no_sig_imp_list.pop(-1)
        else:
            break
    print(
        f"final selected indices: {selected_indices} "
        f"labels: {list(np.array(labels)[selected_indices])}"
    )
    return selected_indices
