import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial import distance_matrix
import os

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
