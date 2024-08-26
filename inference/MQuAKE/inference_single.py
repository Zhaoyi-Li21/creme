import sys
sys.path.append("../..") 
from transformers import AutoTokenizer
import transformers
import torch
from utils import model_name2path, get_lm_type, ModelAndTokenizer, GEN_CONFIGS
import json
import random
random.seed(0)
# data_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/inference/2wmh.1000.json"
# write_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/inference/2wmh.1000.pred.example.json"
knowledge = "compositional" # single
mode = "question" # "cloze"
model = sys.argv[1].replace('_', '-') # "llama2-7b" or "openalpaca-3b" 
use_demon = True
dir_path = "/root/autodl-tmp/zhaoyi/creme" # change it to yours
data_path = dir_path + "/data/mquake/MQuAKE-CF-3k.2hop.json"
if use_demon:
    write_path = dir_path + "/inference/single-hop/"+model+'.json'

demonstration_path = dir_path + "/data/mquake/prompts/rel-prompts.dedup.json"
device = "cuda"
model_name = "llama2-7b"
generation_config = GEN_CONFIGS[model_name]
model_path = model_name2path(model_name)
model_type = get_lm_type(model_name)

fr = open(data_path, "r")
data = json.load(fr)
fr_demon = open(demonstration_path, "r")
demon = json.load(fr_demon) # dict?
assert type(demon) == dict
rel_codes = list(demon.keys())
def get_other_rel_codes(tgt_rel_code, num=8):
    others = list()
    for rel_code in rel_codes:
        if rel_code != tgt_rel_code:
            others.append(rel_code)
    return random.sample(others, num)

fw = open(write_path, "w")

mt = ModelAndTokenizer(
        model_name = model_path,
        #low_cpu_mem_usage=IS_COLAB,
        torch_dtype = (torch.float16 if (("7b" in model_name) or ("6b" in model_name)) else None),
        model_type = model_type,
    )


# mt.tokenizer.pad_token = mt.tokenizer.bos_token
# mt.model.config.pad_token_id = mt.model.config.bos_token_id

# reference: https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020/3
# debug: RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
# potentially debug: Batch Inference != Sequential Inference 
mt.tokenizer.pad_token = "[PAD]"
mt.tokenizer.padding_side = "left"

output_data = list()
count = 0
for datum in data:
    output_datum = dict()
    
    for idx in range(2):

        prompt = datum["single_hops"][idx][mode]
        if mode == "question" and use_demon:
            rel_code = datum["orig"]["triples"][idx][1]
            other_rel_codes = get_other_rel_codes(rel_code, 8)
            demonstrations = list()
            for rel_code in other_rel_codes:
                cands = demon[rel_code].split('\n')
                demonstrations.append(random.choice(cands))

            demonstrations = '\n'.join(demonstrations)
        
            # prompt = demonstrations + "\nQ: " + prompt + " A: "
            # prompt = demonstrations + "\nQ: " + prompt + " A:"
            prompt = demonstrations + "\nQ: " + prompt
        inputs = mt.tokenizer(prompt, return_tensors='pt', padding=True).to(device)
        input_length = inputs.input_ids.shape[1]
        input_num = inputs.input_ids.shape[0]
        with torch.no_grad():
            outputs = mt.model.generate(
                **inputs, 
                max_new_tokens=generation_config["max_length"], 
                do_sample=generation_config["do_sample"], 
                temperature=generation_config["temperature"], 
                top_p=generation_config["top_p"], 
            )
        # print(outputs)
        string = mt.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(strings)
        
        output_datum["prompt_"+str(idx)] = prompt
        output_datum["answer_"+str(idx)] = datum["single_hops"][idx]["answer"]
        output_datum["pred_"+str(idx)] = string

    output_data.append(output_datum)
    # if len(output_data) == 1:
    # break
    count += 1
    print(count)
    
json.dump(output_data, fw, indent=4)





