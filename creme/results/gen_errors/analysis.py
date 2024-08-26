read_file = "/root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/results/gen_errors_2/llama2.suf.json"
import json

def answer_is_correct(pred:str, groundtruth:str)->bool:
    '''
    e.g., (English, American English), (Obama, Barack Obama)
    '''
    if pred in groundtruth: return True
    if groundtruth in pred: return True
    return False

fr = open(read_file, "r")
data = json.load(fr)
para_cnt = 0
para_cnt_acc = 0
para_pre_total = 0
para_post_total = 0
para_ppl = 0
gen_cnt = 0
gen_cnt_acc = 0
gen_pre_total = 0
gen_post_total = 0
gen_ppl = 0

para_consist_cnt = 0
gen_consist_cnt = 0
para_consist_cnt_total = 0
gen_consist_cnt_total = 0
for i in range(len(data)):

    for datum in data[i]["paraphrase"]:
        target = datum["target"]
        pre_predict = datum["pre_predict"]
        post_predict = datum["post_predict"]
        pre_ppl = datum["pre_ppl"]
        post_ppl = datum["post_ppl"]
        para_ppl += (pre_ppl-post_ppl)/pre_ppl
        para_cnt += 1
        if pre_predict == "":
            continue
        if answer_is_correct(pre_predict, target): para_pre_total += 1
        if answer_is_correct(post_predict, target): para_post_total += 1
        para_cnt_acc += 1

        if answer_is_correct(pre_predict, target): continue
        if answer_is_correct(pre_predict, post_predict): para_consist_cnt += 1
        para_consist_cnt_total += 1

    for datum in data[i]["generalization"]:
        target = datum["target"]
        pre_predict = datum["pre_predict"]
        post_predict = datum["post_predict"]
        pre_ppl = datum["pre_ppl"]
        post_ppl = datum["post_ppl"]
        gen_ppl += (pre_ppl-post_ppl)/pre_ppl
        gen_cnt += 1
        if pre_predict == "":
            continue
        if answer_is_correct(pre_predict, target): gen_pre_total += 1
        if answer_is_correct(post_predict, target): gen_post_total += 1
        gen_cnt_acc += 1

        if answer_is_correct(pre_predict, target): continue
        if answer_is_correct(pre_predict, post_predict): gen_consist_cnt += 1
        gen_consist_cnt_total += 1

print('paraphrase:')
print('ppl delta:', para_ppl/para_cnt, 'pre acc:', para_pre_total/para_cnt_acc, 'post acc:', para_post_total/para_cnt_acc, 'consist ratio:', para_consist_cnt/para_consist_cnt_total)
print('generalization:')
print('ppl delta:', gen_ppl/gen_cnt, 'pre acc:', gen_pre_total/gen_cnt_acc, 'post acc:', gen_post_total/gen_cnt_acc, 'consist ratio:', gen_consist_cnt/gen_consist_cnt_total)