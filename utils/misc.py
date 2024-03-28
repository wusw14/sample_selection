import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial import distance_matrix
import os
import pandas as pd


def evaluate(y_truth, y_pred):
    """Evaluate model."""
    precision = precision_score(y_truth, y_pred)
    recall = recall_score(y_truth, y_pred)
    f1 = f1_score(y_truth, y_pred)
    return precision * 100, recall * 100, f1 * 100


def softmax(s1, s2):
    s0 = np.min([np.min(s1), np.min(s2)])
    s1 = np.sum(np.exp(s1 - s0))
    s2 = np.sum(np.exp(s2 - s0))
    return s1 / (s1 + s2), s2 / (s1 + s2)


def cal_similarity_matrix(embeddings, args):
    data_dir = args.data_dir.replace("data", "temp_data")
    if args.sim_func == "cosine":
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sim_matrix = np.dot(embeddings, embeddings.T)
    elif args.sim_func in ["L1norm", "L2norm"]:
        dist_matrix = distance_matrix(
            embeddings, embeddings, p=1 if args.sim_func == "L1norm" else 2
        )
        dist_matrix = np.array(dist_matrix) / np.max(dist_matrix)
        sim_matrix = 1 - dist_matrix
    elif args.sim_func == "cosine_sim_diff":
        embeddingsA = np.loadtxt(os.path.join(data_dir, "train_A_emb.npy"))
        embeddingsB = np.loadtxt(os.path.join(data_dir, "train_B_emb.npy"))
        embeddingsA = embeddingsA / np.linalg.norm(embeddingsA, axis=1, keepdims=True)
        embeddingsB = embeddingsB / np.linalg.norm(embeddingsB, axis=1, keepdims=True)
        cosine_sim = np.sum(embeddingsA * embeddingsB, axis=1)
        sim_matrix = np.abs(cosine_sim.reshape(-1, 1) - cosine_sim.reshape(1, -1))
        sim_matrix = 1 - sim_matrix / np.max(sim_matrix)
    else:
        raise NotImplementedError
    return sim_matrix


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


def MFL_l1(reps, n):
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


def get_samples(inputs, labels, embs, candidate_indices, scores):
    inputs = [inputs[idx] for idx in candidate_indices]
    labels = [labels[idx] for idx in candidate_indices]
    embs = [embs[idx] for idx in candidate_indices]
    scores = [scores[idx] for idx in candidate_indices]
    print(f"candiates: {len(inputs)}, pos/neg: {sum(labels)}/{len(labels)-sum(labels)}")
    return inputs, labels, embs, scores
