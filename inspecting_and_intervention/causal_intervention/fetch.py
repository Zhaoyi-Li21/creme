import json
import sys
mode = sys.argv[1] # "prefix" or "suffix"
model_name = sys.argv[2].replace('_', '-') # "llama2-7b"
dir_path = "/root/autodl-tmp/zhaoyi/creme"

fail_read_file = dir_path + "/inference/MQuAKE/filter/" + model_name + ".pass_singles_fail_comp."+mode+".json"
pass_read_file = dir_path + "/inference/MQuAKE/filter/" + model_name + ".pass_all."+mode+".json"

fr_fail = open(fail_read_file)
fr_pass = open(pass_read_file)
fail_data = json.load(fr_fail)
pass_data = json.load(fr_pass)

write_file = dir_path + "/inspecting_and_intervention/causal_intervention/" + model_name + "." + mode + ".json"
fw = open(write_file, "w")
output_data = list()
for datum in pass_data:
    output_datum = dict()
    output_datum["comp question"] = datum["compositional"][0]
    output_datum["comp cloze"] = datum["comp cloze"]
    output_datum["single cloze"] = datum["single cloze"]
    output_datum["implicit result"] = datum["answers"][0]
    output_datum["explicit result"] = datum["answers"][1]
    output_datum["type"] = "Pass"
    output_data.append(output_datum)

for datum in fail_data:
    output_datum = dict()
    output_datum["comp question"] = datum["compositional"][0]
    output_datum["comp cloze"] = datum["comp cloze"]
    output_datum["single cloze"] = datum["single cloze"]
    output_datum["implicit result"] = datum["answers"][0]
    output_datum["explicit result"] = datum["answers"][1]
    output_datum["type"] = "Fail"
    output_data.append(output_datum)

json.dump(output_data, fw, indent=4)
