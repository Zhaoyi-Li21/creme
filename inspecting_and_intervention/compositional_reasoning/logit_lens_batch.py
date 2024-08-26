import os, re, json, sys
# os.chdir("/root/autodl-tmp/zhaoyi/knowledge_locate/rome")
sys.path.append("..") 
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset
from utils import model_name2path, get_lm_type
import random
import matplotlib.pyplot as plt

random.seed(0)
torch.set_grad_enabled(False)

# model_name = "gpt2-xl"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
# model_name = "gpt-j-6b"
# model_name = "openalpaca-3b"
model_name = "llama2-7b"
model_string = model_name.split('-')[0]

model_path = model_name2path(model_name)
model_type = get_lm_type(model_name)
print("Model Name:{}, Model Path:{}, Model Type:{}".format(model_name, model_path, model_type))
mt = ModelAndTokenizer(
    model_name = model_path,
    #low_cpu_mem_usage=IS_COLAB,
    torch_dtype = (torch.float16 if (("7b" in model_name) or ("6b" in model_name)) else None),
    model_type = model_type,
)
vocab_size = int(mt.model.state_dict()['lm_head.weight'].shape[0])

average = True

save_dir = "/root/autodl-tmp/zhaoyi/knowledge_locate/logit_lens/batch/results"
exp_name = model_string + ".wpred.dedup.switch.suffix"

exp_name = exp_name +'.json'
save_path = save_dir + '/' +exp_name
fw = open(save_path,"w")
try_one = False


W = mt.model.state_dict()['lm_head.weight'] # W * h = v, shape = [32000, hid_dim]


def get_rank(logits, check_tok_enc):
                    logits_dict = dict()
                    for i in range(len(logits)):
                        logits_dict[i] = logits[i]
                    logits_dict = sorted(logits_dict.items(),key=lambda item:item[1], reverse=True)
                    
                    cnt = 0
                    
                    temp_rank = dict()
                    for enc in check_tok_enc:
                        temp_rank[enc] = 0

                    for elem in logits_dict:
                        cnt += 1
                        key, value = elem
                        if key in check_tok_enc:
                            temp_rank[key] += cnt
                            # check_rank.append(cnt)

                    temp_rank_sum = sum([v for v in temp_rank.values()])
                    return temp_rank_sum/len(check_tok_enc)


def calculate_hidden_flow(
    mt, prompt, check_tok_ids:int, explicit_toks:list
    ):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """

    inp = make_inputs(mt.tokenizer, [prompt] * 2)
    with torch.no_grad():
        out = mt.model(**inp,output_hidden_states=True)["hidden_states"]
        # print(len(out)) # 33, why not 32?
        # print(mt.num_layers) # 32
        # print(out.shape)
        for layer_idx in range(len(out)):
            projs = torch.matmul(W, out[layer_idx][0, check_tok_ids])
            logits = torch.softmax(projs, dim=0).tolist()
            origin_prob = 0
            for tok in explicit_toks:
                origin_prob += logits[tok]
            origin_prob /= len(explicit_toks)
            rank = get_rank(logits, explicit_toks)
            if layer_idx == 0:
                origin_lens = [origin_prob]
                origin_rank = [rank]
            else:
                origin_lens.append(origin_prob)
                origin_rank.append(rank)        
            
    return origin_lens, origin_rank




def get_tgt_tok_id(prefix):
    inner_prefix = make_inputs(mt.tokenizer, [prefix])
    inner_toks_prefix = [mt.tokenizer.decode(inner_prefix["input_ids"][0][i]) for i in range(inner_prefix["input_ids"].shape[1])]
    return len(inner_toks_prefix) - 1

def enc_tok(check_tok, avg=False):
        '''
        avg = True: return all of the encs of the answer string.
        '''
        print('check_tok_enc:', mt.tokenizer.encode(check_tok))
        # check_tok_enc = mt.tokenizer.encode(check_tok)[-1]
        if avg == False:
            check_tok_enc = mt.tokenizer.encode(check_tok)[1] # detach [SOS] token
        else:
            check_tok_enc = mt.tokenizer.encode(check_tok)[1:] # detach [SOS] token
        if isinstance(check_tok_enc, list):
            return check_tok_enc
        elif isinstance(check_tok_enc, int):
            return [check_tok_enc]
        else:
            print(check_tok_enc)
            raise Exception("format is not expected")

def exp(prompt, prompt_guide, prompt_subj, prompt_guide_subj, implicit_answer, explicit_answer):
    last_tok_id_1 = get_tgt_tok_id(prompt)
    last_tok_id_2 = get_tgt_tok_id(prompt_guide)
    subj_tok_id_1 = get_tgt_tok_id(prompt_subj)
    subj_tok_id_2 = get_tgt_tok_id(prompt_guide_subj)
    last_tok_id = max(last_tok_id_1, last_tok_id_2)
    subj_tok_id = max(subj_tok_id_1, subj_tok_id_2)
    check_tok_ids = [last_tok_id, subj_tok_id]
    print('check_tok_ids:', check_tok_ids)


    implicit_toks = enc_tok(implicit_answer, avg=average)
    explicit_toks = enc_tok(explicit_answer, avg=average)
    print(implicit_toks)
    print(explicit_toks)

    origin_prob, origin_rank, results, mlp_results, attn_results = calculate_hidden_flow(mt, prompt, check_tok_ids, implicit_toks, explicit_toks)

    last_tok_data = results[0]
    mlp_last_tok_data = mlp_results[0]
    attn_last_tok_data = attn_results[0]

    subj_tok_data = results[1]
    mlp_subj_tok_data = mlp_results[1]
    attn_subj_tok_data = attn_results[1]

    x = list(range(len(last_tok_data["debias_prob"])))
    baseline_prob = [origin_prob for i in range(len(x))]
    baseline_rank = [origin_rank for i in range(len(x))]
    
    return last_tok_data, mlp_last_tok_data, attn_last_tok_data, \
            subj_tok_data, mlp_subj_tok_data, attn_subj_tok_data, \
            baseline_prob, baseline_rank

# plot prob variance
def plot(x, ys, labels, x_label, y_label, title='test', save_path=None):
    colors = ['dimgrey','violet','green', 'red', 'magenta', "lime", "salmon"]
    plt.figure()
    for i in range(len(ys)):
        plt.plot(x, ys[i], linewidth = 0.7, marker='D', mec='black', label=labels[i], color=colors[i])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(prop = { "size": 9 })
    # 添加横向网格线
    plt.grid()
    # 显示图表
    plt.savefig(save_path)
    plt.clf()

output_data = list()


file_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/inference/MQuAKE/filter_2/"+model_string+".pass_singles_fail_comp.wpred.dedup.switch.sufcloze.json"
fr = open(file_path, "r")
data = json.load(fr)
for i in range(len(data)):
    datum = data[i]
    prompt_comp = datum["comp cloze"]
    prompt_single = datum["single cloze"]

    implicit_answer = datum["answers"][0]
    explicit_answer = datum["answers"][1]

    last_tok_id_comp = get_tgt_tok_id(prompt_comp)
    last_tok_id_single = get_tgt_tok_id(prompt_single)

    implicit_toks = enc_tok(implicit_answer, avg=average)
    explicit_toks = enc_tok(explicit_answer, avg=average)

    comp_implicit_probs, _ = calculate_hidden_flow(mt, prompt_comp, last_tok_id_comp, implicit_toks)
    comp_explicit_probs, _ = calculate_hidden_flow(mt, prompt_comp, last_tok_id_comp, explicit_toks)
    single_implicit_probs, _ = calculate_hidden_flow(mt, prompt_single, last_tok_id_single, implicit_toks)
    single_explicit_probs, _ = calculate_hidden_flow(mt, prompt_single, last_tok_id_single, explicit_toks)
    
    output_datum = dict()
    output_datum["flag"] = "Fail-"+str(i)
    output_datum["comp_implicit_probs"] = comp_implicit_probs
    output_datum["comp_explicit_probs"] = comp_explicit_probs
    output_datum["single_implicit_probs"] = single_implicit_probs
    output_datum["single_explicit_probs"] = single_explicit_probs

    output_data.append(output_datum)

    if try_one:
        break



file_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/inference/MQuAKE/filter_2/"+model_string+".pass_all.wpred.dedup.switch.sufcloze.json"
fr_2 = open(file_path, "r")
data = json.load(fr_2)
for i in range(len(data)):
    datum = data[i]
    prompt_comp = datum["comp cloze"]
    prompt_single = datum["single cloze"]

    implicit_answer = datum["answers"][0]
    explicit_answer = datum["answers"][1]

    last_tok_id_comp = get_tgt_tok_id(prompt_comp)
    last_tok_id_single = get_tgt_tok_id(prompt_single)

    implicit_toks = enc_tok(implicit_answer, avg=average)
    explicit_toks = enc_tok(explicit_answer, avg=average)

    comp_implicit_probs, _ = calculate_hidden_flow(mt, prompt_comp, last_tok_id_comp, implicit_toks)
    comp_explicit_probs, _ = calculate_hidden_flow(mt, prompt_comp, last_tok_id_comp, explicit_toks)
    single_implicit_probs, _ = calculate_hidden_flow(mt, prompt_single, last_tok_id_single, implicit_toks)
    single_explicit_probs, _ = calculate_hidden_flow(mt, prompt_single, last_tok_id_single, explicit_toks)
    

    output_datum = dict()
    output_datum["flag"] = "Pass-"+str(i)
    output_datum["comp_implicit_probs"] = comp_implicit_probs
    output_datum["comp_explicit_probs"] = comp_explicit_probs
    output_datum["single_implicit_probs"] = single_implicit_probs
    output_datum["single_explicit_probs"] = single_explicit_probs

    output_data.append(output_datum)

    if try_one:
        break




json.dump(output_data, fw, indent=4)