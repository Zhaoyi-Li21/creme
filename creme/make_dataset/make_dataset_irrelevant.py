# [
#     {
#       "prompt_guide": "The country that {} belongs to is",
#       "prompt_comp":"The country that {} belongs to is",
#       "prompt_f_hop":"the creator of C. Auguste Dupin is",
#       "subject_guide": "Edgar Allan Poe",
#       "subject_comp": "the creator of C. Auguste Dupin",
#       "target": "United States of America",
#       "wrong answer":"France",
#       "paraphrase queries": [
#         "Q: Who is the president of the United States?\nA: Joe Biden\nQ: What is the citizenship of the creator of C. Auguste Dupin?\nA:",
#         "Q: Who is the president of the United States?\nA: Joe Biden\nQ: What is the nationality of the creator of C. Auguste Dupin?\nA:",
#         "Q: Who is the president of the United States?\nA: Joe Biden\nQ: Which country does the creator of C. Auguste Dupin belong to?\nA:",
#         "Q: Who is the president of the United States?\nA: Joe Biden\nQ: To which country is the creator of C. Auguste Dupin affiliated with?\nA:",
#         "The country that the creator of C. Auguste Dupin belongs to is",
#         "The nationality of the creator of C. Auguste Dupin is",
#         "The country where the creator of C. Auguste Dupin is a citizen is",
#         "The creator of C. Auguste Dupin is a citizen of"
#       ],
#       "generalization queries":[],
#       "generalization answers":[],
#       "layers":[17, 18, 19, 20, 21, 22, 23]
#     }
# ]

import json
import sys
import random

model_name = sys.argv[1].replace('_', '-')# "openalpaca-3b" or "llama2-7b"
if model_name == "llama2-7b":
    model_string = 'llama2'
elif model_name == "openalpaca-3b":
    model_string = 'openalpaca'

mode = 'suf' 
fix = 'suffix'
dir_path = "/root/autodl-tmp/zhaoyi/creme"
# compositional reasoning failures
read_path = dir_path + "/inference/MQuAKE/filter/"+model_name+".pass_singles_fail_comp."+fix+".json"
fr = open(read_path, "r")


reference_path = dir_path + "/data/mquake/MQuAKE-CF-3k.2hop.edit.json"
fr_reference = open(reference_path, "r")
refer_data = json.load(fr_reference)

write_path = dir_path + "/creme/make_dataset/"+model_string+'.'+mode+'.irre.json'
fw = open(write_path, "w")
data = json.load(fr)

pre_path = dir_path + "/data/mquake/comp_cloze_prefix.json"
suf_path = dir_path + "/data/mquake/comp_cloze_suffix.json"

fr_pre = open(pre_path, "r")
fr_suf = open(suf_path, "r")
pre_data = json.load(fr_pre)
suf_data = json.load(fr_suf)

pre_question_text = "Q: Who is the prime minister of Canada?\nA: Justin Trudeau\nQ: "
post_question_text = "\nA:"

output_data = list()

assert len(refer_data) == 1000
assert len(pre_data) == 1000
assert len(suf_data) == 1000

for i in range(len(data)):
    datum = data[i]
    output_datum = dict()
    
    output_datum["prompt_guide"] = datum["single cloze"].replace(datum["subj"], '{}')
    output_datum["prompt_comp"] = datum["single cloze"].replace(datum["subj"], '{}')
    output_datum["prompt_f_hop"] = datum["subj_hop1"] + ' is'
    output_datum["subject_guide"] = datum["subj"]
    output_datum["subject_comp"] = datum["subj_hop1"]
    output_datum["target"] = datum["answers"][1]
    output_datum["wrong answer"] = datum["preds"][0]
    output_datum["original query"] = datum["comp cloze"]
    
    if mode == 'suf': 
        mode_data = suf_data
        check_data = pre_data
    elif mode == 'pre':
        mode_data = pre_data
        check_data = suf_data
    ids = -1
    for j in range(len(mode_data)):
        if datum["comp cloze"] == mode_data[j]["cloze"]:
            ids = j
    # if ids == -1:
    #     print(1)
    #     print(datum["comp cloze"])

    assert ids != -1 
    output_datum["irrelevant queries"] = list()
    output_datum["irrelevant answers"] = list()
    for _ in range(5):
        ids = random.randint(0, len(refer_data)-1)
        if len(refer_data[ids]["irrelevant answers"])-1 == -1:
            continue
        sub_ids = random.randint(0, len(refer_data[ids]["irrelevant answers"])-1)
        if refer_data[ids]["irrelevant answers"][sub_ids] == output_datum["target"]:
            continue
        if refer_data[ids]["irrelevant answers"][sub_ids] == output_datum["subject_guide"]:
            continue    

        output_datum["irrelevant queries"].append(refer_data[ids]["irrelevant clozes"][sub_ids])
        output_datum["irrelevant queries"].append(pre_question_text + refer_data[ids]["irrelevant questions"][sub_ids] + post_question_text)

        output_datum["irrelevant answers"].append(refer_data[ids]["irrelevant answers"][sub_ids])
        output_datum["irrelevant answers"].append(refer_data[ids]["irrelevant answers"][sub_ids])
    output_data.append(output_datum)

json.dump(output_data, fw, indent=4)

