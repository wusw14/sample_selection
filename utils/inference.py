import time
from utils.llm import score_to_prob
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from utils.llm import ask_chatgpt, ask_ada, get_api_cost, answer_to_pred
import numpy as np


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop[1:].to("cuda") for stop in stops]
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False


def inference(model_name, model, tokenizer, test_prompts, args):
    start_time = time.time()
    start = 0
    pred_list, ans_list = [], []
    if args.lm in ["gpt3.5", "gpt4", "ada"]:
        max_new_token = 100 if args.output_format == "o2" else 5
        for i in range(start, len(test_prompts)):
            prompt = test_prompts[i]
            if args.lm in ["gpt3.5", "gpt4"]:
                answer, token_prompt, token_completion = ask_chatgpt(prompt, model_name)
                cost += get_api_cost(model_name, token_prompt, token_completion)
            elif args.lm == "ada":
                answer, pred = ask_ada(model_name, prompt)
            else:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                out = model.generate(
                    inputs=input_ids.cuda(),
                    max_new_tokens=max_new_token,
                )
                out = out[0][len(input_ids[0]) :]
                output_text = tokenizer.decode(
                    out,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )
                answer = output_text.replace("\n", " ").strip()
            if args.lm == "ada":
                pred_list.append(pred)
            else:
                pred_list.append(answer_to_pred(answer))
            ans_list.append(answer)
            print(f"Case {i}: pred {pred_list[-1]} answer {answer}")
    else:
        if args.CoT:
            max_new_token = 200
        else:
            max_new_token = 1
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.bos_token_id
        if args.lm.startswith("llama2-"):
            max_len = 4096
        else:
            max_len = 2048
        for i in range(start, len(test_prompts), args.batch_size):
            inputs = tokenizer(
                test_prompts[i : i + args.batch_size],
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if args.CoT:
                stop_words_ids = [
                    tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
                    for stop_word in ["the answer is yes", "the answer is no"]
                ]
                stopping_criteria = StoppingCriteriaList(
                    [StoppingCriteriaSub(stops=stop_words_ids)]
                )
                outputs = model.generate(
                    inputs.input_ids.cuda(),
                    max_new_tokens=max_new_token,
                    stopping_criteria=stopping_criteria,
                )
                answer = tokenizer.decode(
                    outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
                )
                pred, answer = answer_to_pred(answer, CoT=True)
                pred_list.append(pred)
                ans_list.append(answer)
                print(f"Case {i}, label: {pred}, answer: {answer}")
            else:
                outputs = model.generate(
                    inputs.input_ids.cuda(),
                    max_new_tokens=max_new_token,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
                scores = outputs["scores"][0].detach().cpu().numpy()

                for j, score in enumerate(scores):
                    pos_prob = score_to_prob(score, tokenizer)
                    ans_list.append(pos_prob)
                    pred_list.append(int(pos_prob > 0.5))
                if i % 100 == 0:
                    print(f"processing case {i}")
    total_time = time.time() - start_time
    print(f"Total time for prediction: {total_time:.2f}s")
    return pred_list, ans_list
