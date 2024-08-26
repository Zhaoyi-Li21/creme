'''
`rel-prompts.json` sometimes has the risk of data leaking;
This file is to find those examplars which has already appeared in the single-hop questions;
'''
def is_dup(question:str, data:list)->bool:
    for datum in data:
        for i in range(2):
            if question == datum["single_hops"][i]["question"]:
                return True
    return False

dir_path = "/root/autodl-tmp/zhaoyi" # your dictionary path
data_file = dir_path + "/creme/data/mquake/MQuAKE-CF-3k.2hop.json"
original_prompt_file = dir_path + "/creme/data/mquake/prompts/rel-prompts.json"
import json

fp_data = open(data_file, "r")
fp_prompt = open(original_prompt_file, "r")
data = json.load(fp_data)
prompts = json.load(fp_prompt)
new_prompts = dict()
for key in prompts.keys():
    prompt_set = prompts[key].split('\n')
    print(key)
    print(len(prompt_set))
    new_prompt_set = list()
    for prompt in prompt_set:
        temp = prompt.split(' A: ')
        question = temp[0]
        question = question.split('Q: ')
        question = question[1]
        if is_dup(question, data): continue
        new_prompt_set.append(prompt)
    print(len(new_prompt_set))
    new_prompts[key] = '\n'.join(new_prompt_set)

new_prompt_file = dir_path + "/creme/data/mquake/prompts/rel-prompts.dedup.json"
fw_prompt = open(new_prompt_file, "w")
json.dump(new_prompts, fw_prompt, indent=4)
