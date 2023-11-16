from collections import defaultdict
import numpy as np
from scipy.spatial import distance_matrix
from utils.prompt import construct_prompt
from utils.inference import inference
import pandas as pd


def cal_cosine_sim(args):
    data_dir = args.data_dir.replace("data", "temp_data")
    embeddingsA = np.loadtxt(f"{data_dir}/train_A_emb.npy")
    embeddingsB = np.loadtxt(f"{data_dir}/train_B_emb.npy")
    embeddingsA = embeddingsA / np.linalg.norm(embeddingsA, axis=1, keepdims=True)
    embeddingsB = embeddingsB / np.linalg.norm(embeddingsB, axis=1, keepdims=True)
    cosine_sim = np.sum(embeddingsA * embeddingsB, axis=1)
    return cosine_sim


def stratified_sampling(inputs, labels, embeddings, cosine_of_each_pair, args):
    # by cosine similarity and null values distribution
    missing_value_reps = []
    for entry_pair in inputs:
        valsA = entry_pair.valsA
        valsB = entry_pair.valsB
        missing_valsA = np.array(pd.isna(valsA), int)
        missing_valsB = np.array(pd.isna(valsB), int)
        missing_value_rep = missing_valsA + missing_valsB
        missing_value_rep = "".join([str(v) for v in missing_value_rep])
        missing_value_reps.append(int(missing_value_rep))

    df = pd.DataFrame(
        {
            "id": list(range(len(inputs))),
            "score": cosine_of_each_pair,
            "missing_value_rep": missing_value_reps,
        }
    )
    df["group"] = pd.cut(df["score"], bins=5, labels=[0, 1, 2, 3, 4])
    group_id, missing_value_rep, _ = zip(
        *sorted(
            df.groupby(["group", "missing_value_rep"]).size().reset_index().values,
            key=lambda x: x[-1],
        )
    )
    group_num = len(group_id)
    sample_indices = []
    for i in range(group_num):
        df_sub = df[
            (df.group == group_id[i]) & (df.missing_value_rep == missing_value_rep[i])
        ]
        n = (100 - len(sample_indices)) // (group_num - i)
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


def our(model_name, model, tokenizer, inputs, labels, embeddings, args):
    # sample by cosine similarity
    cosine_of_each_pair = cal_cosine_sim(args)
    inputs, labels, embeddings, candidate_indices, scores = stratified_sampling(
        inputs, labels, embeddings, cosine_of_each_pair, args
    )
    print(f"candidates: {len(inputs)}")

    # warm up with the sample with the highest and lowest cosine similarity score
    example_inputs, example_labels, example_embeddings = [], [], []
    selected_indices = [np.argmax(scores), np.argmin(scores)]
    for idx in selected_indices:
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])

    # iterative sampling by confidence
    # conf_decay = 25
    scaler = -np.log(0.5)
    certain_thr = 1 + (0.65 * np.log(0.65) + (1 - 0.65) * np.log(1 - 0.65)) / scaler
    historical_confs, historical_probs = [], []

    while len(selected_indices) < min(args.k, args.budget):
        print(f"****************")
        print(f"iteration {len(selected_indices) + 1}")
        # probs with two orders
        prompts = construct_prompt(
            example_inputs, example_labels, example_embeddings, inputs, embeddings, args
        )
        print(prompts[0])
        print("---------------\n\n")
        _, probs = inference(model_name, model, tokenizer, prompts, args)
        probs = np.array(probs)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        conf = 1 + (probs * np.log(probs) + (1 - probs) * np.log(1 - probs)) / scaler
        if len(example_inputs) > 2:
            # infer again for the examples with low confidence
            inputs2 = [
                inputs[idx] for idx in range(len(inputs)) if conf[idx] < certain_thr
            ]
            if len(inputs2) > 0:
                embeddings2 = [
                    embeddings[idx]
                    for idx in range(len(inputs))
                    if conf[idx] < certain_thr
                ]
                prompts2 = construct_prompt(
                    example_inputs,
                    example_labels,
                    example_embeddings,
                    inputs2,
                    embeddings2,
                    args,
                )
                print(prompts2[0])
                print("---------------\n\n")
                _, probs2 = inference(model_name, model, tokenizer, prompts2, args)
                for idx in range(len(inputs)):
                    if conf[idx] < certain_thr:
                        probs[idx] = (probs2.pop(0) + probs[idx]) / 2.0
                        conf[idx] = (
                            1
                            + (
                                probs[idx] * np.log(probs[idx])
                                + (1 - probs[idx]) * np.log(1 - probs[idx])
                            )
                            / scaler
                        )

        conf = np.clip(conf, 0, certain_thr)
        historical_confs.append(conf)  # [T, N]
        historical_probs.append(probs)  # [T, N]
        cond1 = conf < certain_thr
        cond2 = conf < np.max(historical_confs, axis=0)
        if np.sum(cond1 * cond2) > 0:
            uncertain_indices = np.where(cond1 * cond2)[0]
        elif np.sum(cond1) > 0:
            uncertain_indices = np.where(cond1)[0]
        else:
            uncertain_indices = np.arange(len(inputs))

        # calculate similarity between samples based on their predicted probs
        probs_reps = np.array(historical_probs)[:, uncertain_indices].T  # [K, T]
        dist_matrix = distance_matrix(probs_reps, probs_reps, p=1)  # [K, K]
        dist_score = np.sum(dist_matrix, axis=1)  # [K]
        idx = uncertain_indices[np.argmin(dist_score)]
        probs_output = [f"{v:.4f}" for v in probs]

        print(f"uncertain_indices ({len(uncertain_indices)}): {uncertain_indices}")
        print(f"dist_score: {dist_score}")
        print(
            f"index: {idx}, label: {labels[idx]}, pred: {probs[idx]:.2f}, conf: {conf[idx]:.2f}"
        )
        result = [v for v in zip(range(len(labels)), labels, probs_output)]
        print(result)
        print(f"****************\n\n")
        selected_indices.append(idx)
        example_inputs.append(inputs[idx])
        example_labels.append(labels[idx])
        example_embeddings.append(embeddings[idx])
    while len(selected_indices) < args.budget:
        idx = np.argmax(scores)
        if idx in selected_indices:
            scores[idx] = -100
        else:
            selected_indices.append(idx)
    selected_indices = [candidate_indices[idx] for idx in selected_indices]
    return selected_indices
