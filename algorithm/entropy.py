from collections import defaultdict
import numpy as np
from tqdm import tqdm
from math import ceil
from utils.misc import cal_similarity_matrix
from utils.prompt import construct_prompt
from utils.inference import inference
from collections import namedtuple
import pandas as pd

EntryPair = namedtuple("EntryPair", ["cols", "valsA", "valsB"])


def cal_entropy(
    model_name, model, tokenizer, inputs, labels, embeddings, args, prekg=None
):
    if prekg == None:
        prompts = construct_prompt([], [], [], inputs, embeddings, args)
    else:
        prompts = construct_prompt(
            prekg[0], prekg[1], prekg[2], inputs, embeddings, args
        )
    print(prompts[0])
    _, predictions = inference(model_name, model, tokenizer, prompts, args)
    predictions = np.array(predictions)
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
    entropys = -predictions * np.log(predictions) - (1 - predictions) * np.log(
        1 - predictions
    )
    return entropys, predictions


def max_entropy(model_name, model, tokenizer, inputs, labels, embeddings, args):
    entropys, _ = cal_entropy(
        model_name, model, tokenizer, inputs, labels, embeddings, args
    )
    sorted_indices = np.argsort(entropys).tolist()[::-1]
    return sorted_indices[: args.budget]


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


def max_entropy_bl(model_name, model, tokenizer, inputs, labels, embeddings, args):
    # 1st sampling
    # 2nd max entropy
    cosine_of_each_pair = cal_cosine_sim(args)
    inputs, labels, embeddings, candidate_indices, scores = stratified_sampling(
        inputs, labels, embeddings, cosine_of_each_pair, args
    )
    indices = [np.argmax(scores), np.argmin(scores)]
    prekg = [
        [inputs[indices[0]], inputs[indices[1]]],
        [labels[indices[0]], labels[indices[1]]],
        [embeddings[indices[0]], embeddings[indices[1]]],
    ]
    entropys, predictions = cal_entropy(
        model_name, model, tokenizer, inputs, labels, embeddings, args, prekg
    )
    sorted_indices = np.argsort(predictions).tolist()[::-1]
    left_index, right_index = 0, len(sorted_indices) - 1
    next_target = "pos"
    while len(indices) < args.budget:
        if next_target == "pos":
            while sorted_indices[left_index] in indices:
                left_index += 1
            indices.append(sorted_indices[left_index])
            if labels[indices[-1]] == 1:
                next_target = "neg"
        else:
            while sorted_indices[right_index] in indices:
                right_index -= 1
            indices.append(sorted_indices[right_index])
            if labels[indices[-1]] == 0:
                next_target = "pos"
    indices = indices[: args.budget]
    indices = [candidate_indices[idx] for idx in indices]
    return indices


def min_entropy(model_name, model, tokenizer, inputs, labels, embeddings, args):
    entropys, _ = cal_entropy(
        model_name, model, tokenizer, inputs, labels, embeddings, args
    )
    sorted_indices = np.argsort(entropys).tolist()
    return sorted_indices[: args.budget]


def cbs_maxIG(model_name, model, tokenizer, inputs, labels, embeddings, args):
    # calibration tokens
    cf_tokens = ["[MASK]", "N/A"]
    cols = inputs[0].cols
    cf_inputs = []
    for token in cf_tokens:
        cf_inputs.append(EntryPair(cols, [token] * len(cols), [token] * len(cols)))
    prompts = construct_prompt([], [], [], cf_inputs, [1] * len(cf_inputs), args)
    _, cf_predictions = inference(model_name, model, tokenizer, prompts, args)
    print(f"cf_predictions = {cf_predictions}")
    cf_predictions = np.clip(np.mean(cf_predictions), 1e-6, 1 - 1e-6)
    # original predictions
    _, predictions = cal_entropy(
        model_name, model, tokenizer, inputs, labels, embeddings, args
    )
    predictions = np.array(predictions)  # [N]
    # calibration
    pos_prob = predictions / cf_predictions
    neg_prob = (1 - predictions) / (1 - cf_predictions)
    cbs_prob = np.exp(pos_prob) / (np.exp(pos_prob) + np.exp(neg_prob))
    entropys = -cbs_prob * np.log(cbs_prob) - (1 - cbs_prob) * np.log(1 - cbs_prob)
    sorted_indices = np.argsort(entropys).tolist()
    return sorted_indices[: args.budget]
