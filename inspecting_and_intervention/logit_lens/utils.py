'''
load models and tokenizers from locals; generation configs
'''

import os
import re
import torch
from transformers import (
    BertTokenizer,
    BertLMHeadModel,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM
)

# locate model from local path
model_dir = "/root/autodl-tmp/zhaoyi/huggingface_models" # change the path to yours

def model_name2path(model_name:str)->str:
    if model_name == "gpt2-xl":
        return model_dir + "/gpt2-xl"
    if model_name == "gpt2-medium":
        return model_dir + "/gpt2-medium"
    elif model_name == "gpt-j-6b":
        return model_dir + "/gpt-j-6b"
    elif model_name == "openalpaca-3b":
        return model_dir + "/openalpaca3b"
    elif model_name == "llama2-7b":
        return model_dir + "/llama2-7b-hf"
    elif model_name == "llama2-13b":
        return model_dir + "/llama2-13b-hf"
    elif model_name == "redpajama-7b":
        return model_dir + "/RedPajama-INCITE-7B-Instruct"
    elif model_name == "qwen-7b":
        return model_dir + "/qwen-7b"
    
def get_lm_type(model_name:str)->str:
    if "gpt" in model_name:
        # GPT2-small,base,large,xl; GPT-J-6B
        lm_type = "gpt"
    elif "llama" in model_name or "alpaca" in model_name:
        # OpenAlpaca-3B, LLaMA-2-7B
        lm_type = "llama"
    elif "qwen" in model_name:
        lm_type = "qwen"
    else: 
        raise Exception("model:{} is currently not covered in the model list.".format(model_name))
    return lm_type

def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model_type=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            assert model_type is not None

            if model_type == "llama":
                if "llama2" in model_name:
                    if "13b" in model_name:
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                else:
                    tokenizer = LlamaTokenizer.from_pretrained(model_name)
            elif model_type == "gpt":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            elif model_type == "qwen":
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            else:
                raise Exception("model type:{} is currently not covered in the model type list.".format(model_type))
        if model is None:
            assert model_name is not None
            assert model_type is not None
            if model_type == "llama":
                model = LlamaForCausalLM.from_pretrained(model_name,torch_dtype=torch_dtype)
            elif model_type == "gpt":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    low_cpu_mem_usage=low_cpu_mem_usage, 
                    torch_dtype=torch_dtype
                )
            elif model_type == "qwen":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    low_cpu_mem_usage=low_cpu_mem_usage, 
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
            else:
                raise Exception("model type:{} is currently not covered in the model type list.".format(model_type))

            # nethook.set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.model_type = model_type
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
        # for n, m in model.named_modules():
        #     print(n)
        # for n in self.layer_names:
        #     print(n)
        # raise Exception("debug")
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )

GEN_CONFIGS = dict()

GEN_CONFIGS["llama2-7b"]={
  "bos_token_id": 1,
  "do_sample": True,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "temperature": 0.6,
  "max_length": 50,
  "top_p": 0.9,
  "transformers_version": "4.31.0.dev0"
}

GEN_CONFIGS["llama2-13b"]={
  "bos_token_id": 1,
  "do_sample": True,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "temperature": 0.6,
  "max_length": 50,
  "top_p": 0.9,
  "transformers_version": "4.32.0.dev0"
}