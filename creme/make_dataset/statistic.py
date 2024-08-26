read_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/openalpaca.suf.new.json"
fr = open(read_path, "r")
import json
data = json.load(fr)
para_cnt = 0
gen_cnt = 0
irre_cnt = 0
for datum in data:
    para_cnt += len(datum["paraphrase queries"])
    gen_cnt += len(datum["generalization queries"])
    irre_cnt += len(datum["irrelevant queries"])

print(para_cnt/len(data), gen_cnt/len(data), irre_cnt/len(data))