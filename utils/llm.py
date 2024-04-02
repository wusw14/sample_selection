import numpy as np
import openai
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .misc import softmax

LLMS = [
    "gpt2",
    "llama7",
    "llama13",
    "llama30",
    "llama65",
    "vicuna",
    "gpt3.5",
    "gpt4",
    "ada",
]


def get_model_name(name):
    # if name not in LLMS:
    #     raise ValueError(f"Unknown model name: {name}")
    if name == "gpt2":
        return "gpt2"
    elif name == "gpt3.5":
        return "gpt-3.5-turbo-0301"
    elif name == "gpt4":
        return "gpt-4-0314"
    elif name == "vicuna":
        return "eachadea/vicuna-13b-1.1"
    elif name == "ada":
        return "text-ada-001"
    elif "llama2-" in name:
        if "bf" in name:
            version = name.split("-")[-1][:-2]
            return f"/ssddata/liuyue/github/llama_models/llama-2-{version}b-chat-hf"
        else:
            version = name.split("-")[-1][:-1]
            return f"/ssddata/liuyue/github/llama_models/llama-2-{version}b-hf"
    elif "llama" in name:  # llama7, llama13, llama30, llama65
        size = int(name[5:])
        if size == 65:
            return "huggyllama/llama-65b"
        else:
            return f"decapoda-research/llama-{size}b-hf"
    else:
        raise ValueError(f"Unknown model name: {name}")


def init_model(args):
    """model initialization"""
    model, tokenizer = None, None
    model_name = get_model_name(args.lm)
    if args.lm == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, device_map="auto")
        model = GPT2LMHeadModel.from_pretrained(
            model_name, torch_dtype=torch.half, device_map="auto"
        )
    elif args.lm.startswith("llama"):  # LLaMa
        tokenizer = LlamaTokenizer.from_pretrained(model_name, device_map="auto")
        model = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.half, device_map="auto"
        )
    elif args.lm == "vicuna":
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.half, device_map="auto"
        )
    return model_name, model, tokenizer


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def ask_chatgpt(text, model):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    org = os.getenv("OPENAI_ORGANIZATION")
    if org:
        openai.organization = org

    messages = [{"role": "user", "content": text}]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0
    )
    return (
        response["choices"][0]["message"]["content"],
        response["usage"]["prompt_tokens"],
        response["usage"]["completion_tokens"],
    )


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def ask_ada(model_name, text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    org = os.getenv("OPENAI_ORGANIZATION")
    if org:
        openai.organization = org

    response = openai.Completion.create(
        model=model_name,
        prompt=text,
        temperature=0,
        max_tokens=1,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logprobs=2,
    )

    top_tokens = response["choices"][0]["logprobs"]["top_logprobs"][0]
    pos_score, neg_score = None, None
    for k, v in top_tokens.items():
        if "yes" in k.lower():
            pos_score = v
        elif "no" in k.lower():
            neg_score = v
    if pos_score is not None and neg_score is not None:
        answer = np.exp(pos_score) / (np.exp(pos_score) + np.exp(neg_score))
        pred = int(answer > 0.5)
        return answer, pred
    elif pos_score is not None:
        return 1, 1
    elif neg_score is not None:
        return 0, 0
    else:
        return 0.5, 1


def get_api_cost(model_name, token_prompt, token_completion):
    if "gpt-4" in model_name:
        return token_prompt * 0.03 + token_completion * 0.06
    elif "gpt-3.5" in model_name:
        return (token_prompt + token_completion) * 0.002
    return 0.0


def score_to_prob(score, tokenizer):
    idxs, score = zip(
        *sorted(
            zip(list(range(len(score))), score),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    pos_score, neg_score = [], []
    for idx, s in zip(idxs, score):
        if len(pos_score) == 0 and "yes" in tokenizer.decode(idx).lower():
            pos_score = [s]
        elif len(neg_score) == 0 and "no" in tokenizer.decode(idx).lower():
            neg_score = [s]
        if len(pos_score) > 0 and len(neg_score) > 0:
            break
    pos_score, neg_score = softmax(pos_score, neg_score)
    return pos_score


def answer_to_pred(answer, CoT=False):
    answer = answer.lower().strip()
    if CoT:
        try:
            idx = answer.index("therefore, the answer is")
            answer_label = answer.split()[-1]
            answer = answer.replace("\n", " ")
            return 1 - int("no" in answer_label), answer
        except:
            answer = answer.replace("\n", " ")
            different_cnt, same_cnt = 0, 0
            for word in answer.split():
                if word == "different":
                    different_cnt += 1
                elif word == "same":
                    same_cnt += 1
            return int(different_cnt <= same_cnt), answer
    if "yes" in answer and "no" in answer:
        index1 = answer.index("yes")
        index2 = answer.index("no")
        return int(index1 < index2)
    else:
        return 1 - int("no" in answer)
    # TODO: What if both yes and no are not in the answer?


def get_answer_space(model_name, tokenizer):
    if model_name == "gpt2":
        pos_words = [" Yes"]
        neg_words = [" No"]
        pos_words_ids = tokenizer(pos_words).input_ids
        neg_words_ids = tokenizer(neg_words).input_ids
        print(pos_words_ids)
        print(neg_words_ids)
        pos_words_ids = np.array([w[0] for w in pos_words_ids])
        neg_words_ids = np.array([w[0] for w in neg_words_ids])
    else:
        pos_words = ["Yes"]
        neg_words = ["No"]
        pos_words_ids = tokenizer(pos_words).input_ids
        neg_words_ids = tokenizer(neg_words).input_ids
        print(pos_words_ids)
        print(neg_words_ids)
        pos_words_ids = np.array([w[1] for w in pos_words_ids])
        neg_words_ids = np.array([w[1] for w in neg_words_ids])
    print(pos_words_ids)
    print(neg_words_ids)
    return pos_words_ids, neg_words_ids
