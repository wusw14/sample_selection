from collections import defaultdict
import numpy as np
from utils.misc import cal_similarity_matrix
from utils.prompt import construct_prompt
from utils.inference import inference


def adaicl(model_name, model, tokenizer, inputs, labels, embeddings, args):
    sim_matrix = cal_similarity_matrix(embeddings, args)
    phases = 2
    selected_indices = []
    example_inputs, example_labels, example_embeddings = [], [], []
    for i in range(phases):
        if i == phases - 1:
            select_num = args.budget - len(selected_indices)
        else:
            select_num = args.budget // phases
        # 1st filter out those easy samples
        n = len(inputs)
        prompts = construct_prompt(
            example_inputs, example_labels, example_embeddings, inputs, embeddings, args
        )
        print(prompts[0])
        _, predictions = inference(model_name, model, tokenizer, prompts, args)
        predictions = np.array(predictions)
        conf = np.maximum(predictions, 1 - predictions)
        sorted_indices = np.argsort(conf).tolist()
        candidate_indices = sorted_indices[: n // 2]

        # 2nd: diversity sampling
        vote_stat = defaultdict(list)
        for i in range(n):
            cur_scores = sim_matrix[i]
            sorted_indices = np.argsort(cur_scores).tolist()[-16:-1]
            for idx in sorted_indices:
                if idx != i:
                    vote_stat[idx].append(i)
        votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
        new_selected_indices = []
        selected_times = defaultdict(int)
        egonet = defaultdict(list)
        # Create egonets
        for idx, candidates in votes:
            for idx_support in candidates:
                if (idx_support in candidate_indices) and (
                    idx_support not in egonet[idx]
                ):
                    egonet[idx].append(idx_support)
                    selected_times[idx] += 1

        egonet_greedy = sorted(egonet.items(), key=lambda x: len(x[1]), reverse=True)
        selected_weight = defaultdict(int)
        while len(new_selected_indices) < select_num:
            cur_scores = defaultdict(int)
            for idx, candidates in egonet_greedy:
                if idx in selected_indices or idx in new_selected_indices:
                    cur_scores[idx] = -100  # sanity check
                    continue
                for idx_support in candidates:
                    if idx_support in candidate_indices:  # sanity check
                        cur_scores[idx] += 10 ** (-selected_weight[idx_support])

            cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
            new_selected_indices.append(int(cur_selected_idx))

            for idx_support in egonet[cur_selected_idx]:
                selected_weight[idx_support] += 1
        selected_indices += new_selected_indices
        for idx in new_selected_indices:
            example_inputs.append(inputs[idx])
            example_labels.append(labels[idx])
            example_embeddings.append(embeddings[idx])
        print(f"selected indices {len(selected_indices)}: {selected_indices}")
    return selected_indices
