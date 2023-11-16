import numpy as np
import pandas as pd
import os
from .transforms import serialize, get_serialize_func
from .prefix import PREFIXES, ORDER_OF_EXAMPLES

Q_FORMAT = {
    "q1": "Do {item} A and {item} B refer to the same entity?",
    "q2": "Are {item} A and {item} B the same?",
    "q3": "Do these two {item}s refer to the same entity?",
    "q4": "Are these two {item}s the same?",
}

TASK_DESP = {
    "t1": "The task is to identify whether the two {item}s refer to the same entity based on the attribute values.\n\n",
    "t2": "The task is to identify whether the two {item}s refer to the same entity.\n\n",
    "t3": "This is an entity matching task.\n\n",
}

INSTRUCT = {
    "i1": "Determine whether {item} A and {item} B refer to the same entity. ",
    "i2": "Determine whether the two {item}s refer to the same entity. ",
    "i3": "Determine whether the two {item}s are the same. ",
}

OUT_FORMAT = {
    "o1": "Give your answer as either yes or no.\n\n",
    "o2": "First give your answer as either yes or no, then briefly explain your thoughts.\n\n",
}


def get_prompt_parts(args):
    return {
        "question": Q_FORMAT.get(args.question_format, "").format(item=args.entry_type),
        "desc": TASK_DESP.get(args.desp, "").format(item=args.entry_type),
        "out": OUT_FORMAT.get(args.output_format, ""),
    }


def construct_cases(
    question, pos_cases, neg_cases, cases_org, labels, order, args=None
):
    cases, pos_cases_post, neg_cases_post = [], [], []
    if len(cases_org) > 0 and (order == "o6" or order == "o7"):
        if order == "o7":
            cases_org = cases_org[::-1]
            labels = labels[::-1]
        for case, label in zip(cases_org, labels):
            case1 = f"{case}\n{question} {label}."
            cases.append(case1)
    else:
        for i, pos in enumerate(pos_cases):
            case1 = f"{pos}\n{question} Yes."
            pos_cases_post.append(case1)
        for i, neg in enumerate(neg_cases):
            case2 = f"{neg}\n{question} No."
            neg_cases_post.append(case2)

    if order == "o1":
        for case1, case2 in zip(pos_cases_post, neg_cases_post):
            cases.append(case1)
            cases.append(case2)
    elif order == "o2":
        for case1, case2 in zip(pos_cases_post, neg_cases_post):
            cases.append(case2)
            cases.append(case1)
    elif order == "o3":
        order_index = int(args.version[-1]) - 1
        order_list = ORDER_OF_EXAMPLES[(len(pos_cases), len(neg_cases))][order_index]
        print(f"order_list = {order_list}")
        for o in order_list:
            if o == 1:
                cases.append(pos_cases_post.pop(0))
            else:
                cases.append(neg_cases_post.pop(0))
    elif order == "o5":
        cases = pos_cases_post + neg_cases_post
        np.random.shuffle(cases)
    examples = "\n\n".join(cases)
    if len(cases) > 0:
        examples += "\n\n"
    return examples


def get_fewshots(
    example_inputs,
    example_labels,
    example_embeddings,
    entry_emb,
    serialize_func,
    args,
):
    # print(f"len(example_inputs) = {len(example_inputs)}, {len(example_labels)}")
    # print("matches in examples: ", np.sum(example_labels))
    pos_cases, neg_cases = [], []
    pos_indices, neg_indices = [], []
    example_labels = np.array(example_labels)

    k = min(args.k, len(example_inputs))
    args.pos_num = k // 2
    args.neg_num = k - args.pos_num
    # print(f"args.pos/neg: {args.pos_num}/{args.neg_num}")
    if args.select_ind:
        example_embeddings = np.array(example_embeddings)
        cosine_sim = np.dot(example_embeddings, entry_emb)
        indices = np.argsort(cosine_sim)[::-1]
    else:
        indices = list(range(len(example_labels)))

    indices_selected = []
    # split the annotated data into pos/neg
    for i, idx in enumerate(indices):
        if example_labels[idx] == 1:
            pos_indices.append(idx)
            indices_selected.append(idx)
        elif example_labels[idx] == 0:
            neg_indices.append(idx)
            indices_selected.append(idx)
        if (
            len(pos_indices) >= args.pos_num
            and len(neg_indices) >= args.neg_num
            and (len(pos_indices) + len(neg_indices) >= k)
        ):
            break
    if len(pos_indices) + len(neg_indices) > k:
        if len(pos_indices) <= args.pos_num:
            neg_indices = neg_indices[: k - len(pos_indices)]
        elif len(neg_indices) <= args.neg_num:
            pos_indices = pos_indices[: k - len(neg_indices)]

    for idx in pos_indices[::-1]:
        pos_cases.append(serialize_func(example_inputs[idx]))
    for idx in neg_indices[::-1]:
        neg_cases.append(serialize_func(example_inputs[idx]))
    cases, labels = [], []
    for idx in indices_selected[::-1]:
        if idx in pos_indices or idx in neg_indices:
            cases.append(serialize_func(example_inputs[idx]))
            if idx in pos_indices:
                labels.append("Yes")
            else:
                labels.append("No")
    return pos_cases, neg_cases, cases, labels


def get_prefix(prefixes, dataset):
    for key, examples in prefixes.items():
        if dataset in key:
            return examples


def construct_prompt(
    example_inputs, example_labels, example_embeddings, inputs, embeddings, args
):
    serialize_func = get_serialize_func(args.entry_type, args.serialization)
    prompt_parts = get_prompt_parts(args)
    question = prompt_parts.get("question", "")
    desc = prompt_parts.get("desc", "")

    prompt_list = []
    if args.k >= len(example_inputs):
        args.select_ind = False
    else:
        args.select_ind = True
    k = min(args.k, len(example_inputs))
    for i, (entry_pair, entry_emb) in enumerate(zip(inputs, embeddings)):
        if k > 0 and (i == 0 or args.select_ind):
            (pos_cases, neg_cases, cases, labels) = get_fewshots(
                example_inputs,
                example_labels,
                example_embeddings,
                entry_emb,
                serialize_func,
                args,
            )
            if i == 0:
                if len(pos_cases) + len(neg_cases) < k:
                    print("Not enough in-context examples")
                    exit()
            examples = construct_cases(
                prompt_parts.get("question", ""),
                pos_cases,
                neg_cases,
                cases,
                labels,
                args.order,
                args=args,
            )
        elif k == 0:
            examples = ""

        text = serialize_func(entry_pair)
        if args.selection_method == "manual":
            prefix = get_prefix(PREFIXES, args.dataset)
            prompt = f"{prefix}\n{text}\n{question}"
        else:
            prompt = f"{desc}{examples}{text}\n{question}"
        prompt_list.append(prompt)
    return prompt_list
