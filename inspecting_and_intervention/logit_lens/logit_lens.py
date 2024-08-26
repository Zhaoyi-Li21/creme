import os, re, json, sys
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


model_name = "llama2-7b"
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

trace_last_tok = True

dir_path = "/root/autodl-tmp/zhaoyi/creme"
save_dir = dir_path + "/inspecting_and_intervention/logit_lens/results"
# category = "hasty_answer"
# category = "incomplete_thinking"
# category = "distortion"
category = "short_cut"
# category = "guide"
# category = "correct"

save_dir = save_dir + '/'+category

exp_name = "test_1"

exp_name = exp_name + '_' +model_name

if os.path.exists(save_dir+'/'+exp_name) == False:
    # mkdir
    os.mkdir(save_dir+'/'+exp_name)

save_dir = save_dir + "/" + exp_name + "/" 
prob_path = save_dir + "prob.png"
rank_path = save_dir + "rank.png"


prompt = "The home country of the sport associated with Giorgio Chinaglia is"
prefix_inner_s = "The home country of the sport associated with Giorgio Chinaglia"
implicit_answer = "association football"
explicit_answer = "England"
# implicit_answer = "Giorgio Chinaglia"
# explicit_answer = "Italy"

if trace_last_tok == True:
    file_path = save_dir + 'last_' + implicit_answer+ '_' +explicit_answer+ '.png'
else:
    file_path = save_dir + 'inner_' + implicit_answer+ '_' + explicit_answer+ '.png'


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
    print('prefix:',inner_prefix["input_ids"][0])
    inner_toks_prefix = [mt.tokenizer.decode(inner_prefix["input_ids"][0][i]) for i in range(inner_prefix["input_ids"].shape[1])]
    return len(inner_toks_prefix) - 1

last_tok_id = get_tgt_tok_id(prompt)
last_inner_s_id = get_tgt_tok_id(prefix_inner_s)

if trace_last_tok == True:
    check_tok_ids = [last_tok_id]
else:
    check_tok_ids = [last_inner_s_id]
# check_tok_ids = [last_tok_id]

def enc_tok(check_tok, avg=False):
        '''
        avg = True: return all of the encs of the answer string.
        '''
        print('check_tok_enc:', mt.tokenizer.encode(check_tok))
        # check_tok_enc = mt.tokenizer.encode(check_tok)[-1]
        if model_type == 'gpt':
            start_id = 0
        elif model_type == 'llama':
            start_id = 1 # detach [SOS] token

        if avg == False:
            check_tok_enc = mt.tokenizer.encode(check_tok)[start_id] 
        else:
            check_tok_enc = mt.tokenizer.encode(check_tok)[start_id:] 
        if isinstance(check_tok_enc, list):
            return check_tok_enc
        elif isinstance(check_tok_enc, int):
            return [check_tok_enc]
        else:
            print(check_tok_enc)
            raise Exception("format is not expected")
        
implicit_toks = enc_tok(implicit_answer, avg=average)
explicit_toks = enc_tok(explicit_answer, avg=average)

origin_lens, origin_rank = calculate_hidden_flow(mt, prompt, check_tok_ids[0], explicit_toks)
origin_lens_1, origin_rank_1 = calculate_hidden_flow(mt, prompt, check_tok_ids[0], implicit_toks)


x = list(range(len(origin_lens)))

# plot prob variance
def plot_twin_im_ex(_y1_im, _y1_ex, _y2_im, _y2_ex, _ylabel1, _ylabel2, x_label, title, save_path):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(_ylabel1, color=color)
    ax1.plot(_y1_im, label=implicit_answer+'-prob', marker='D',mec='black', color=color)
    ax1.plot(_y1_ex, label=explicit_answer+'-prob', marker='o',mec='black', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

    color = 'tab:red'
    ax2.set_ylabel(_ylabel2, color=color)
    ax2.plot(_y2_im,label=implicit_answer+'-rank', marker='D',mec='black', color=color)
    ax2.plot(_y2_ex, label=explicit_answer+'-rank', marker='o',mec='black', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.legend(prop = { "size": 9 })
    fig.tight_layout()
    plt.title(title, fontsize=10)
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()

def plot_twin(_y1, _y2, _ylabel1, _ylabel2, x_label, title, save_path):
    fig, ax1 = plt.subplots(figsize=(6,4), dpi=500)
    color = '#4476D7'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(_ylabel1, color=color)
    ax1.plot(_y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

    color = 'tab:red'
    ax2.set_ylabel(_ylabel2, color=color)
    ax2.plot(_y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(title, fontsize=10)
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()

def plot(x, ys, labels, x_label, y_label, title='test', save_path=None):
    colors = ['steelblue','darkred','darkgreen', 'gold', 'orchid', "grey", "red", "lime", "fuchsia"]
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
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()



plot_twin(origin_lens, origin_lens_1, "explicit reasoning result", "implicit reasoning result", "layer", 'The home country of the sport associated with Giorgio Chinaglia is', file_path)
