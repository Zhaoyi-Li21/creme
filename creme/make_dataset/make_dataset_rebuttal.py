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

model_name = "openalpaca"
if model_name == "llama-2":
    model_string = 'llama2'
elif model_name == "openalpaca":
    model_string = 'openalpaca'

mode = 'suf'
read_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/inference/MQuAKE/llama2-13b/filter/llama2-13b.pass_singles_fail_comp.suf.json"
fr = open(read_path, "r")
reference_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/datasets/MQuAKE/datasets/MQuAKE-CF-3k.2hop.edit.json"
fr_reference = open(reference_path, "r")
refer_data = json.load(fr_reference)

write_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/llama2_13b_suf.json"
fw = open(write_path, "w")
data = json.load(fr)

pre_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/datasets/MQuAKE/datasets/comp_cloze_prefix.json"
suf_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/datasets/MQuAKE/datasets/comp_cloze_suffix.json"
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

    if check_data[ids]["cloze"] != datum["comp cloze"]:
        # append `` cloze paraphrase "
        output_datum["paraphrase queries"] = [check_data[ids]["cloze"]] + [pre_question_text + q + post_question_text for q in datum["compositional"]]
    else:
        output_datum["paraphrase queries"] = [pre_question_text + q + post_question_text for q in datum["compositional"]]


    # paraphrase: 3 + 1
    # generalization: share first-hop, diverse second-hops;
    # neighbor: irrelevant prompts; (target != output_datum['target'])
        
    output_datum["generalization queries"] = refer_data[ids]["generalization clozes"] + [pre_question_text + q + post_question_text for q in refer_data[ids]["generalization questions"]]
    output_datum["generalization answers"] = refer_data[ids]["generalization answers"] + refer_data[ids]["generalization answers"]
    output_datum["irrelevant queries"] = refer_data[ids]["irrelevant clozes"] + [pre_question_text + q + post_question_text for q in refer_data[ids]["irrelevant questions"]]
    output_datum["irrelevant answers"] = refer_data[ids]["irrelevant answers"] + refer_data[ids]["irrelevant answers"]

    output_data.append(output_datum)

json.dump(output_data, fw, indent=4)

