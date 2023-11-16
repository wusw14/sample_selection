import numpy as np
from utils.misc import cal_similarity_matrix


def MFL(embeddings, args):
    select_num = args.budget
    sim_matrix = cal_similarity_matrix(embeddings, args)
    similarity_to_labeled = np.array([-1.0] * len(embeddings))
    selected_indices, scores = [], []
    while len(selected_indices) < select_num:
        max_score, idx = -1, -1
        for i in range(len(embeddings)):
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
