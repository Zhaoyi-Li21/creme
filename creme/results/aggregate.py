import json
check_key = "delta"
read_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/results/rebuttal/llama2_13b_suf.json"
refer_path = read_path.replace('results/v2', 'make_dataset')
write_path = read_path.replace('suf','suf.avg.'+check_key)
fr = open(read_path, "r")
fr_refer = open(refer_path, "r")
fw = open(write_path, "w")
data = json.load(fr)
refer_data = json.load(fr_refer)
assert len(data) == len(refer_data)
out = dict()
# for key1 in ["correctness", "paraphrase", "generalize", "irrelevant"]:
# key_list = ["irrelevant"]
key_list = ["correctness", "paraphrase", "generalize", "irrelevant"]
for key1 in key_list :
    out[key1] = dict()
    for key2 in ["target", "wrong"]:
        if key2 == "wrong" and key1 == "generalize":
            continue
        out[key1][key2] = list()

for datum in data:
    for key1 in key_list :
        for key2 in datum[key1][check_key]:
            if len(datum[key1][check_key][key2]) == 0:
                continue
            out[key1][key2].append(sum(datum[key1][check_key][key2])/len(datum[key1][check_key][key2]))
            if key2 == "wrong":
                print(sum(datum[key1][check_key][key2])/len(datum[key1][check_key][key2]))

for key1 in key_list :
    for key2 in ["target", "wrong"]:
        if key2 == "wrong" and key1 == "generalize":
            continue
        out[key1][key2] = sum(out[key1][key2]) / len(out[key1][key2])

json.dump(out, fw, indent=4)