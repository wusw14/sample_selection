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


def min_entropy_bl(model_name, model, tokenizer, inputs, labels, embeddings, args):
    indices = []
    entropys, predictions = cal_entropy(
        model_name, model, tokenizer, inputs, labels, embeddings, args
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
    return indices


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
