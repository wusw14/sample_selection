from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import time


model_name = "/ssddata/wushw/llama_models/llama-2-70b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name, device_map="auto")
model = LlamaForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.half, device_map="auto"
)
while True:
    print(time.time())
    time.sleep(3600)
