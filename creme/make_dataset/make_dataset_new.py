import json
model = "openalpaca"
file_1 = "/root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/"+model+".suf.json"
file_2 = "/root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/"+model+".suf.v2.json"
fr1 = open(file_1, "r")
fr2 = open(file_2, "r")
file_3 = file_2.replace('v2', 'new')
fw = open(file_3, "w")
data1 = json.load(fr1)
data2 = json.load(fr2)
assert len(data1) == len(data2)
output_data = list()
for i in range(len(data1)):
    output_datum = dict()
    for key in data1[i].keys():
        if "irrelevant" not in key:
            if key == "prompt_comp":
                output_datum["prompt"] = data1[i][key]
            elif key == "subject_comp":
                output_datum["subject"] = data1[i][key]
            elif key in ["prompt_guide", "prompt_f_hop", "subject_guide"]:
                continue
            else:
                output_datum[key] = data1[i][key]
        else:
            output_datum[key] = data2[i][key]
    output_data.append(output_datum)

json.dump(output_data, fw, indent=4)
