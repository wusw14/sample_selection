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
import time


def cal_cosine_sim(args):
    data_dir = args.data_dir.replace("data", "temp_data")
    embsA = np.loadtxt(f"{data_dir}/train_A_emb.npy")
    embsB = np.loadtxt(f"{data_dir}/train_B_emb.npy")
    embsA = embsA / np.linalg.norm(embsA, axis=1, keepdims=True)
    embsB = embsB / np.linalg.norm(embsB, axis=1, keepdims=True)
    cosine_sim = np.sum(embsA * embsB, axis=1)
    return cosine_sim


def stratified_sampling(inputs, labels, embs, cosine_of_each_pair, args):
    budget = args.sample_size
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
    args,
    next=1,
    no_improvement=False,
):
    probs = np.array(historical_probs[-1])
    for idx in selected_indices:
        probs[idx] = 2
    labels = np.array(labels)
    conf = conf_func(probs)
    cond1 = (probs > 0.5) & (probs <= 1)
    cond2 = probs <= 0.5
    imp_rate = 1.1
    pred_dict = {}
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
        uncertain_indices_all = []
        for j in range(2):
            if len(uncertain_indices) > args.beam_size:
                uncertain_indices_part, _ = sampling(
                    historical_probs,
                    uncertain_indices,
                    [[], labeled_set],
                    n=args.beam_size,
                    type="covered_by_rep",
                )
                # uncertain_indices = list(uncertain_indices) + top_indices
                labeled_set = labeled_set.union(set(uncertain_indices_part))
                uncertain_indices_all.extend(list(uncertain_indices_part))
                xor_sum = np.sum(
                    labels[uncertain_indices_part] * next
                    + (1 - labels[uncertain_indices_part]) * (1 - next)
                )
                print(
                    f"debug: uncertain_indices_part: {uncertain_indices_part} "
                    f"next: {next} "
                    f"args.budget: {args.budget} num of labeled set: {len(labeled_set)}"
                    f" labels: {labels[uncertain_indices_part]} "
                    f" xor_sum: {xor_sum}"
                )
                if xor_sum < min(
                    2, len(uncertain_indices_part) // 2
                ) and args.budget > len(labeled_set):
                    args.beam_size = min(args.beam_size, args.budget - len(labeled_set))
                else:
                    break
            else:
                uncertain_indices_all.extend(list(uncertain_indices))
                break
        uncertain_indices = list(uncertain_indices_all)
        print(f"uncertain_indices: {uncertain_indices}")
        eval_indices, _ = sampling(
            historical_probs,
            np.where(cond1 | cond2)[0],
            None,
            n=args.eval_size,
            type="MFL",
        )
        print(f"### eval indices: {list(eval_indices)}")
        labeled_eval = labeled_set.union(set(uncertain_indices)) - set(selected_indices)
        print("labeled_eval", labeled_eval)

        index, imp_rate, pred_dict = max_info_gain(
            model_name,
            model,
            tokenizer,
            inputs,
            labels,
            embs,
            selected_indices,
            list(uncertain_indices),
            eval_indices,
            labeled_eval,
            probs,
            next,
            no_improvement,
            args,
        )
    print(f"flag = {flag}, selected_index = {index}, imp_rate = {imp_rate:.4f}")
    return index, uncertain_indices, imp_rate, pred_dict


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
    elif type == "MFL":
        selected_indices = MFL(candidate_reps, n)
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
    m = np.round(1.0 / n, 1)
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
    for t, digit in zip([0.1, 0.01, 0.001], [1, 2, 3]):
        m_list = []
        while True:
            m_list.append(m)
            dist_thr = np.percentile(dist, 100.0 * m)
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
                len(selected_indices) / n
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
                m -= t
            elif len(selected_indices) >= n and len(covered_set) < len(probs):
                m += t
            if m in m_list or m <= m_lower or m >= m_upper:
                m_lower = np.round(max(0, best_m - t), digit)
                m_upper = np.round(min(1, best_m + t), digit)
                m = best_m
                break
    print(
        f"covered_conf_range: {max_cover_range:.4f}, best_m: {best_m}, "
        f"covered_ratio: {max_cover_ratio:.4f}, dist_thr: {best_dist_thr:.4f}, "
        f"m_list: {m_list}, stepwise: {t}"
    )
    if len(sample_list) < n:
        print(f"remaining budget: {n - len(sample_list)}")
        print(f"sample_list (before): {len(sample_list)}")
        sample_list += np.random.choice(
            list(set(range(len(probs))) - set(sample_list)),
            n - len(sample_list),
            replace=False,
        ).tolist()
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


def MFL(reps, n):
    dist = distance_matrix(reps, reps, p=1)  # [N, N]
    sim_matrix = 1 - dist / np.max(dist)
    similarity_to_labeled = np.array([-1.0] * len(dist))
    selected_indices, scores = [], []
    while len(selected_indices) < n:
        max_score, idx = -1, -1
        for i in range(len(reps)):
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


def cal_info_score(
    probs1, indices1, probs2, indices2, labels2, selected_index=-1, p=-1, label=-1
):
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
    sample_indices_org,
    labeled_eval_org,
    probs,
    next,
    no_improvement,
    args,
):
    pred_dict = {}
    labeled_eval_org = list(labeled_eval_org)
    hist_indices = []
    if no_improvement == False:
        for idx in labeled_eval_org:
            if labels[idx] == next and idx not in selected_indices + uncertain_indices:
                hist_indices.append(idx)
    else:
        hist_indices = []
    candidate_indices = list(uncertain_indices) + list(hist_indices)

    inputs_of_E, labels_of_E, embs_of_E = [], [], []
    for idx in selected_indices:
        inputs_of_E.append(inputs[idx])
        labels_of_E.append(labels[idx])
        embs_of_E.append(embs[idx])

    # sort labeled_eval_org by difficulty
    if len(labeled_eval_org) > 0:
        diff_scores, labeled_eval_tmp = [], []
        for idx in labeled_eval_org:
            diff_s = probs[idx] * labels[idx] + (1 - probs[idx]) * (1 - labels[idx])
            # if diff_s < 0.8:
            diff_scores.append(diff_s)
            labeled_eval_tmp.append(idx)
        print(
            f"debug+++ diff_scores: {diff_scores} labeled_eval_org: {labeled_eval_org}"
        )
        labeled_eval_org = np.array(labeled_eval_tmp)[np.argsort(diff_scores)].tolist()
    else:
        labeled_eval_org = []

    tau = np.ceil(len(candidate_indices) ** (1.0 / 3)).astype(int)
    print(f"********* tau: {tau}")
    unlabeled_pred, labeled_pred = {}, {}
    sample_indices_last, labeled_eval_last = [], []
    for t in range(1, 4):
        un_eval_size = max(10, np.round(args.eval_size / tau ** (3 - t)).astype(int))
        labeled_eval_size = max(
            5, np.round(len(labeled_eval_org) / tau ** (3 - t)).astype(int)
        )
        print(
            f"\n**********[Eval] iteration {t} candidates: {len(candidate_indices)} "
            f" un_eval_size: {un_eval_size}, labeled_eval_size: {labeled_eval_size}"
        )
        sample_indices = list(sample_indices_org)[:un_eval_size]
        labeled_eval = list(labeled_eval_org)[:labeled_eval_size]
        labeled_labels = np.array(labels)[labeled_eval]

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
        score_base = cal_info_score(
            probs[sample_indices],
            sample_indices,
            probs[labeled_eval],
            labeled_eval,
            labeled_labels,
        )
        best_sample, score_max = None, 0
        info_gain = []
        for i, idx in enumerate(candidate_indices):
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

            # part 2: predictions on the labeled data
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
            labeled_pred[idx] = labeled_probs
            if idx in labeled_eval:
                labeled_probs[labeled_eval.index(idx)] = probs[idx]
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
                pred_dict = {}
                for j, index in enumerate(sample_indices):
                    pred_dict[index] = probs1[j]
                for j, index in enumerate(labeled_eval):
                    pred_dict[index] = labeled_probs[j]
            info_gain.append(score)
            del inputs_of_E[-1]
            del labels_of_E[-1]
            del embs_of_E[-1]
        # normalize info_dict
        info_rank = np.array(rankdata(info_gain)) / len(info_gain)
        info_dict = {}
        for i, rank in enumerate(info_rank):
            info_dict[candidate_indices[i]] = rank
        if len(info_dict) > 0:
            # top 5
            top_indices, _ = zip(
                *sorted(info_dict.items(), key=lambda x: x[1], reverse=True)
            )
            candidate_indices = top_indices[: np.ceil(len(info_dict) / tau).astype(int)]
        else:
            candidate_indices = []
        if len(candidate_indices) < 2:
            print(f"Not too much candidates left for comparison!!!")
            break
    return best_sample, score_max / score_base - 1, pred_dict


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


def get_samples(inputs, labels, embs, candidate_indices, scores):
    inputs = [inputs[idx] for idx in candidate_indices]
    labels = [labels[idx] for idx in candidate_indices]
    embs = [embs[idx] for idx in candidate_indices]
    scores = [scores[idx] for idx in candidate_indices]
    print(f"candiates: {len(inputs)}, pos/neg: {sum(labels)}/{len(labels)-sum(labels)}")
    return inputs, labels, embs, scores


def ideal(model_name, model, tokenizer, inputs, labels, embs, args):
    # sample by cosine similarity
    cosine_of_each_pair = cal_cosine_sim(args)
    candidate_indices = MFL(np.reshape(cosine_of_each_pair, (-1, 1)), args.sample_size)
    inputs, labels, embs, scores = get_samples(
        inputs, labels, embs, candidate_indices, cosine_of_each_pair
    )
    # inputs, labels, embs, candidate_indices, scores = stratified_sampling(
    #     inputs, labels, embs, cosine_of_each_pair, args
    # )
    unselected_indices = list(range(len(labels)))
    print(f"candidates: {len(inputs)}")
    # warm up with the sample with the highest and lowest cosine similarity score
    labeled_set = set()
    indices_by_scores = np.argsort(scores)[::-1]
    for i in range(len(scores)):
        labeled_set.add(indices_by_scores[i])
        if labels[indices_by_scores[i]] == 1:
            selected_indices = [indices_by_scores[i]]
            break
    for i in range(len(scores) - 1, -1, -1):
        labeled_set.add(indices_by_scores[i])
        if labels[indices_by_scores[i]] == 0:
            selected_indices.append(indices_by_scores[i])
            break
    inputs_of_E, labels_of_E, embs_of_E = [], [], []
    # selected_indices = [np.argmax(scores), np.argmin(scores)]
    for idx in selected_indices:
        inputs_of_E.append(inputs[idx])
        labels_of_E.append(labels[idx])
        embs_of_E.append(embs[idx])
        del unselected_indices[unselected_indices.index(idx)]

    # iterative sampling by confidence
    historical_probs = []
    # labeled_set = set(selected_indices)

    imp_rate = 1
    imp_thr = 0.05
    last_pred = {}
    no_improvement = False
    start_time = time.time()
    while len(selected_indices) < args.k:
        print(f"\n\n****************iteration {len(selected_indices) + 1}")

        if imp_rate > imp_thr:
            # LLM's predictions based on selected examples $\mathbf{E}$
            pred_indices, inputs_of_U, embs_of_U, labels_of_U = [], [], [], []
            for idx in unselected_indices:
                if idx in last_pred:
                    continue
                pred_indices.append(idx)
                inputs_of_U.append(inputs[idx])
                embs_of_U.append(embs[idx])
                labels_of_U.append(labels[idx])
            print(f"debug+++ pred ratio {len(inputs_of_U)}/{len(unselected_indices)}")
            prompts = construct_prompt(
                inputs_of_E, labels_of_E, embs_of_E, inputs_of_U, embs_of_U, args
            )
            print(prompts[0])
            print("---------------\n\n")
            _, probs = inference(model_name, model, tokenizer, prompts, args)
            # update the historical information
            probs = list(np.clip(np.array(probs), 1e-6, 1 - 1e-6))
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

        ### next target
        target = determine_next_target(
            labels_of_E, selected_indices, probs_N, labeled_set, labels
        )
        info_dict = {}
        for next in [target, 1 - target]:
            args.beam_size = np.ceil(
                (args.budget - len(labeled_set)) / (args.k - len(selected_indices))
            ).astype(int)
            print(
                f"remaining budget: {args.budget - len(labeled_set)}, "
                f"beam size: {args.beam_size}, next: {next}"
            )
            idx, labeled_indices, imp_rate, pred_dict = select_next(
                model_name,
                model,
                tokenizer,
                inputs,
                labels,
                embs,
                historical_probs,
                selected_indices,
                labeled_set,
                args,
                next=next,
                no_improvement=no_improvement,
            )
            labeled_set = labeled_set.union(set(labeled_indices))
            info_dict[idx] = imp_rate
            if imp_rate > 0:
                break
        idx = max(info_dict, key=info_dict.get)
        print(
            f"==== index: {idx}, imp_rate: {info_dict[idx]:.4f}, "
            f"Elapsed: {time.time()-start_time:.2f}s"
        )
        # update the variables
        if info_dict[idx] > -imp_thr:
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
            imp_rate = info_dict[idx]
            no_improvement = False
            last_pred = dict(pred_dict)
        else:
            no_improvement = True
            if len(labeled_set) == args.budget:
                print(f"!!! Run out of the budget before selecting the enough examples")
                break

    selected_indices = [candidate_indices[idx] for idx in selected_indices]
    return selected_indices
