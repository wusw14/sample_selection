from collections import defaultdict
import numpy as np
from tqdm import tqdm
from math import ceil
from utils.misc import cal_similarity_matrix
from utils.prompt import construct_prompt
from utils.inference import inference


def fast_votek(embeddings, select_num, args):
    sim_matrix = cal_similarity_matrix(embeddings, args)
    k = 150

    # build the graph
    n = len(embeddings)
    vote_stat = defaultdict(list)
    for i in tqdm(range(n)):
        cur_scores = 1 - sim_matrix[i]
        sorted_indices = np.argsort(cur_scores).tolist()[:k]
        for idx in sorted_indices:
            if idx != i:
                vote_stat[idx].append(i)

    # select the samples
    votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices) < select_num:
        cur_scores = defaultdict(int)
        for idx, candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices, vote_stat, selected_times


def votek(model_name, model, tokenizer, inputs, labels, embeddings, args):
    # 1st: fast_votek
    select_num = ceil(args.budget / 10.0)
    selected_indices, vote_stat, selected_times = fast_votek(
        embeddings, select_num, args
    )
    # 2nd: uncertainty sampling
    args.k = min(args.k, len(selected_indices))
    example_inputs, example_labels, example_embeddings = [], [], []
    for idx in selected_indices:
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
    prompts = construct_prompt(
        example_inputs, example_labels, example_embeddings, inputs, embeddings, args
    )
    print(prompts[0], "\n\n")
    _, predictions = inference(model_name, model, tokenizer, prompts, args)
    predictions = np.array(predictions)
    conf = np.maximum(predictions, 1 - predictions)
    sorted_indices = np.argsort(conf).tolist()
    bin_size = int(len(sorted_indices) * 0.9) // (args.budget - len(selected_indices))

    count_t = 0
    while len(selected_indices) < args.budget and count_t * bin_size < len(
        sorted_indices
    ):
        cur_scores = defaultdict(int)
        for idx in sorted_indices[count_t * bin_size : (count_t + 1) * bin_size]:
            if not str(idx) in vote_stat:
                cur_scores[idx] = 0
                continue
            candidates = vote_stat[str(idx)]
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
        selected_indices.append(cur_selected_idx)
        if cur_selected_idx in vote_stat:
            for idx_support in vote_stat[cur_selected_idx]:
                selected_times[idx_support] += 1
        count_t += 1
    return selected_indices  # list of indices
