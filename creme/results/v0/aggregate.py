import json
check_key = "post_probs"
read_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/results/rome/llama2.suf.new.json"
refer_path = read_path.replace('results/v2', 'make_dataset')

fr = open(read_path, "r")
fr_refer = open(refer_path, "r")

data = json.load(fr)
refer_data = json.load(fr_refer)
assert len(data) == len(refer_data)
out = dict()
# for key1 in ["correctness", "paraphrase", "generalize", "irrelevant"]:
# key_list = ["irrelevant"]
key_list = ["correctness", "paraphrase"]
correct_sum = 0
pre_correct_cnt = 0
post_correct_cnt = 0
para_sum = 0
pre_para_cnt = 0
post_para_cnt = 0





for datum in data:
    
    correct_sum += len(datum["correctness"]["pre_probs"]["target"])
    para_sum += len(datum["paraphrase"]["pre_probs"]["target"])
    for i in range(len(datum["correctness"]["pre_probs"]["target"])):
        if datum["correctness"]["pre_probs"]["target"][i] > datum["correctness"]["pre_probs"]["wrong"][i]:
            pre_correct_cnt += 1
        if datum["correctness"]["post_probs"]["target"][i] > datum["correctness"]["post_probs"]["wrong"][i]:
            post_correct_cnt += 1

    for i in range(len(datum["paraphrase"]["pre_probs"]["target"])):
        if datum["paraphrase"]["pre_probs"]["target"][i] > datum["paraphrase"]["pre_probs"]["wrong"][i]:
            pre_para_cnt += 1
        if datum["paraphrase"]["post_probs"]["target"][i] > datum["paraphrase"]["post_probs"]["wrong"][i]:
            post_para_cnt += 1

print('pre-edit', pre_correct_cnt/correct_sum, pre_para_cnt/para_sum)
print('post-edit', post_correct_cnt/correct_sum, post_para_cnt/para_sum)

