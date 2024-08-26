import json
check_key = "post_probs"
read_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/results/rome/openalpaca.suf.new.json"
refer_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/inference/MQuAKE/filter_2/openalpaca.pass_singles_fail_comp.wpred.dedup.switch.sufcloze.json"

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


short_cut = [18, 21, 30, 31, 50, 68, 82, 84, 91, 96, 115, 117, 122, 143, 145, 152, 178, 188]
incomplete = [1, 2, 5, 8, 9, 10, 11, 12, 13, 15, 16, 17, 22, 29, 38, 41, 42, 43, 44, 54, 72, 78, 80, 82]






for id in range(len(data)):
    datum = data[id]
    refer_datum = refer_data[id]
    # if refer_datum["answers"][0] != "association football":
    #     continue

    if refer_datum["answers"][0] != refer_datum["preds"][len(refer_datum["preds"])-1]:
        continue
    if 1 >0:
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

print('short-cut-pre-edit', pre_correct_cnt/correct_sum, pre_para_cnt/para_sum)
print('short-cut-post-edit', post_correct_cnt/correct_sum, post_para_cnt/para_sum)


# correct_sum = 0
# pre_correct_cnt = 0
# post_correct_cnt = 0
# para_sum = 0
# pre_para_cnt = 0
# post_para_cnt = 0

# for id in incomplete:
#     datum = data[id-1]
#     #for datum in data:
#     if 1 > 0:
#         correct_sum += len(datum["correctness"]["pre_probs"]["target"])
#         para_sum += len(datum["paraphrase"]["pre_probs"]["target"])
#         for i in range(len(datum["correctness"]["pre_probs"]["target"])):
#             if datum["correctness"]["pre_probs"]["target"][i] > datum["correctness"]["pre_probs"]["wrong"][i]:
#                 pre_correct_cnt += 1
#             if datum["correctness"]["post_probs"]["target"][i] > datum["correctness"]["post_probs"]["wrong"][i]:
#                 post_correct_cnt += 1

#         for i in range(len(datum["paraphrase"]["pre_probs"]["target"])):
#             if datum["paraphrase"]["pre_probs"]["target"][i] > datum["paraphrase"]["pre_probs"]["wrong"][i]:
#                 pre_para_cnt += 1
#             if datum["paraphrase"]["post_probs"]["target"][i] > datum["paraphrase"]["post_probs"]["wrong"][i]:
#                 post_para_cnt += 1

# print('incomplete-pre-edit', pre_correct_cnt/correct_sum, pre_para_cnt/para_sum)
# print('incomplete-post-edit', post_correct_cnt/correct_sum, post_para_cnt/para_sum)