def answer_is_correct(pred:str, groundtruth:str, alias:list=None)->bool:
    '''
    e.g., (English, American English), (Obama, Barack Obama)
    '''
    # if "Joe Biden" in pred and "Trump" in groundtruth: return True 
    # if "Trump" in pred and "Joe Biden" in groundtruth: return True 
    if pred in groundtruth: return True
    if groundtruth in pred: return True
    for answer in alias:
        if pred in answer: return True
        if answer in pred: return True

    return False

import json
import sys
model = sys.argv[1].replace('_', '-') # "llama2-7b" or "openalpaca-3b" 
# fix = "suffix" # "suffix"
fix = sys.argv[2] # 'suffix' or 'prefix'
dir_path = "/root/autodl-tmp/zhaoyi/creme"
reference_path = dir_path + "/data/mquake/MQuAKE-CF-3k.2hop.json"
fr_refer = open(reference_path, "r")
refer_data = json.load(fr_refer)

comp_path = dir_path + "/inference/MQuAKE/compositional/llama2-7b.json"
fr_comp = open(comp_path, "r")
comp_data = json.load(fr_comp)
cnt_correct = list()
cnt_pred_1st_hop = list()
for i in range(len(comp_data)):
    refer_datum = refer_data[i]
    datum = comp_data[i]
    for suffix in ['_0', '_1', '_2']:
        prompt = datum['prompt'+suffix]
        pred = datum["pred"+suffix]
        answer = datum["answer"+suffix]
        if prompt+' A: ' not in pred:
            continue
        pred = pred.split(prompt+' A: ')
        pred = pred[1]
        pred = pred.split('\n')
        pred = pred[0]
        # print(pred)
        if answer_is_correct(pred, answer, refer_datum["answer_alias"]) == True:
            # print(pred, answer)
            cnt_correct.append(i)
            break
print(len(cnt_correct))



single_path = dir_path + "/inference/MQuAKE/single-hop/llama2-7b.json"
fr_single = open(single_path, "r")
single_data = json.load(fr_single)

cnt_single = dict()
for a in ["True", "False"]:
    cnt_single[a] = dict()
    for b in ["True", "False"]:
        cnt_single[a][b] = list()

for i in range(len(single_data)):
    refer_datum = refer_data[i]
    datum = single_data[i]
    flag_0 = "False"
    flag_1 = "False"
    for suffix in ['_0', '_1']:
        prompt = datum['prompt'+suffix]
        # if "Q: Which country was Elfquest created in?" in prompt:
        #     break
        pred = datum["pred"+suffix]
        answer = datum["answer"+suffix]
        if suffix == '_0':
            alias = refer_datum["single_hops"][0]["answer_alias"]
        elif suffix == '_1':
            alias = refer_datum["single_hops"][1]["answer_alias"]
        if prompt+' A: ' not in pred:
            continue
        pred = pred.split(prompt+' A: ')
        pred = pred[1]
        pred = pred.split('\n')
        pred = pred[0]
        # print(pred)
        if answer_is_correct(pred, answer, alias) == True:
            if suffix == '_0':
                flag_0 = "True"
            else:
                flag_1 = "True"
    cnt_single[flag_0][flag_1].append(i)

for a in ["True", "False"]:
    for b in ["True", "False"]:
        print(a, b, len(cnt_single[a][b]))


result = dict()
for a in ["True", "False"]:
    result[a] = dict()
    for b in ["True", "False"]:
        result[a][b] = dict()
        for c in ["True", "False"]:
            result[a][b][c] = list()

for k in range(1000):
    flag_comp = "False"
    flag_single_0 = "False"
    flag_single_1 = "False"
    if k in cnt_single["True"]["True"] or k in cnt_single["True"]["False"]:
        flag_single_0 = "True"
    if k in cnt_single["False"]["True"] or k in cnt_single["True"]["True"]:
        flag_single_1 = "True"
    if k in cnt_correct:
        flag_comp = "True"
    result[flag_single_0][flag_single_1][flag_comp].append(k)

cnt_pred_1st_hop = list()
for i in range(len(comp_data)):
    refer_datum = refer_data[i]
    datum = comp_data[i]
    for suffix in ['_0', '_1', '_2']:
        prompt = datum['prompt'+suffix]
        pred = datum["pred"+suffix]
        answer = single_data[i]["answer_0"]
        if prompt+' A: ' not in pred:
            continue
        pred = pred.split(prompt+' A: ')
        pred = pred[1]
        pred = pred.split('\n')
        pred = pred[0]
        # print(pred)
        if answer_is_correct(pred, answer, refer_datum["single_hops"][0]["answer_alias"]) == True:
            # print(pred, answer)
            cnt_pred_1st_hop.append(i)
            break
        
print(len(cnt_pred_1st_hop))

for a in ["True", "False"]:
    for b in ["True", "False"]:
        for c in ["True", "False"]:
            temp = 0
            for elem in result[a][b][c]:
                if elem in cnt_pred_1st_hop:
                    temp += 1
            print("pass_single_0:", a, "pass_single_1:", b, "pass_comp:", c, 'count', len(result[a][b][c]), 'overlap ratio with predict 1st hop', round(temp/len(result[a][b][c]),3))

file_pass_singles_fail_comp = dir_path + "/inference/MQuAKE/filter/"+model+".pass_singles_fail_comp."+fix+".json"
file_pass_all = dir_path + "/inference/MQuAKE/filter/"+model+".pass_all."+fix+".json"
fw_1 = open(file_pass_singles_fail_comp, "w")
fw_2 = open(file_pass_all, "w")
origin_data_path = dir_path + "/data/mquake/MQuAKE-CF-3k.2hop.json"
fr_origin = open(origin_data_path, "r")
origin_data = json.load(fr_origin)

comp_cloze_path = dir_path + "/data/mquake/comp_cloze_"+fix+".json"
fr_comp_cloze = open(comp_cloze_path, "r")
comp_cloze = json.load(fr_comp_cloze)

output_data_1 = list()
output_data_2 = list()
for id in result["True"]["True"]["False"]:
    origin_datum = origin_data[id]
    comp_datum = comp_data[id]
    datum = dict()
    datum["first-hop"] = origin_datum["single_hops"][0]["question"]
    datum["second-hop"] = origin_datum["single_hops"][1]["question"]
    datum["compositional"] = origin_datum["questions"]
    datum["answers"] = [origin_datum["single_hops"][0]["answer"],origin_datum["single_hops"][1]["answer"]]
    datum["preds"] = list()
    for suffix in ['_0', '_1', '_2']:
        pred = comp_datum["pred"+suffix]
    
        prompt = comp_datum['prompt'+suffix]
        if prompt+' A: ' not in pred:
            continue
        pred = pred.split(prompt+' A: ')
        pred = pred[1]
        pred = pred.split('\n')
        pred = pred[0]
        datum["preds"].append(pred)
    datum["comp cloze"] = comp_cloze[id]["cloze"]
    datum["single cloze"] = comp_cloze[id]["single cloze"]
    datum["subj_prefix_comp"] = comp_cloze[id]["subj_prefix_comp"]
    datum["subj_prefix_single"] = comp_cloze[id]["subj_prefix_single"]
    datum["subj_hop1"] = comp_cloze[id]["subj_hop1"]
    datum["subj"] = comp_cloze[id]["subj"]
    output_data_1.append(datum)

for id in result["True"]["True"]["True"]:
    origin_datum = origin_data[id]
    comp_datum = comp_data[id]
    datum = dict()
    datum["first-hop"] = origin_datum["single_hops"][0]["question"]
    datum["second-hop"] = origin_datum["single_hops"][1]["question"]
    datum["compositional"] = origin_datum["questions"]
    datum["answers"] = [origin_datum["single_hops"][0]["answer"],origin_datum["single_hops"][1]["answer"]]
    datum["preds"] = list()
    for suffix in ['_0', '_1', '_2']:
        pred = comp_datum["pred"+suffix]
    
        prompt = comp_datum['prompt'+suffix]
        if prompt+' A: ' not in pred:
            continue
        pred = pred.split(prompt+' A: ')
        pred = pred[1]
        pred = pred.split('\n')
        pred = pred[0]
        datum["preds"].append(pred)
    datum["comp cloze"] = comp_cloze[id]["cloze"]
    datum["single cloze"] = comp_cloze[id]["single cloze"]
    datum["subj_prefix_comp"] = comp_cloze[id]["subj_prefix_comp"]
    datum["subj_prefix_single"] = comp_cloze[id]["subj_prefix_single"]
    datum["subj_hop1"] = comp_cloze[id]["subj_hop1"]
    datum["subj"] = comp_cloze[id]["subj"]
    output_data_2.append(datum)

json.dump(output_data_1, fw_1, indent=4)
json.dump(output_data_2, fw_2, indent=4)


