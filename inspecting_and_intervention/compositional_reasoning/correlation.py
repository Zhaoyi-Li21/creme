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
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns

random.seed(0)
torch.set_grad_enabled(False)

# model_name = "gpt2-xl"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
# model_name = "gpt-j-6b"
# model_name = "openalpaca-3b"
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
use_t = False
recheck = True
rc_strength = 10
mode = "suppress" # or "enhance"
# mode = "enhance"
mode_str = "_" + mode[0]

save_dir = "/root/autodl-tmp/zhaoyi/knowledge_locate/debias_causal_intervent/new_cases_3"

# exp_name = "test_5_v2"
# exp_name = "test_6_transpose_compare"
# exp_name = "test_7"
# exp_name = "test_8_compare4_mask"
# exp_name = "test_12"
# exp_name = "test_10"
# average = False
# exp_name = "test_4_compare5"
# exp_name = "test_4_compare3_semantic"
exp_name = "test_1"
# exp_name = "test_12_compare_semantic"

if average:
    exp_name = exp_name + "_avg"
if use_t:
    exp_name = exp_name + "_t"
if recheck:
    exp_name = exp_name + "_rc"
    if rc_strength != 10:
        exp_name = exp_name + "_" +str(rc_strength)

exp_name = exp_name + mode_str

if os.path.exists(save_dir+'/'+exp_name) == False:
    # mkdir
    os.mkdir(save_dir+'/'+exp_name)

save_dir = save_dir + "/" + exp_name + "/" 
prob_path = save_dir + "prob.png"
rank_path = save_dir + "rank.png"
lens_path = save_dir + "lens.offset.1.png"


# prompt = "The capital city of China is"
# prefix_inner_s = "The capital city of China"
# implicit_answer = "America"
# explicit_answer = "Washington" # new_test_69

prompt = "The capital city of France is"
prefix_inner_s = "The capital city of France"
implicit_answer = "France"
explicit_answer = "Paris" # new_test_70
explicit_answer = "France"

implicit_answer = "German"
explicit_answer = "Berlin" # new_test_72
explicit_answer = "German"

# prompt = "The capital city of German is"
# prefix_inner_s = "The capital city of Berlin"
# implicit_answer = "German"
# explicit_answer = "Berlin" # new_test_71

trace_layers = [10, 15, 20 ,25]
prompt = "The capital city of the country that has the largest population is"
# prompt = "[MASK] of the country that has the largest population is"
# prompt = "[MASK]"
# prompt = "The captial city of China is"
implicit_answer_cands = ['America', 'China', 'India', 'Canada', 'Britain', 'France', 'German', 'Brazil', 'Russia', 'Japan']
explicit_answer_cands = ['Washington', 'Beijing', 'New Delhi', 'Ottawa', 'London', 'Paris', 'Berlin', 'Brasilia', 'Moscow', 'Tokyo']

# W = mt.model.state_dict()['lm_head.weight'] # W * h = v, shape = [32000, hid_dim]
# X = torch.matmul(torch.transpose(W, 1, 0), W) # X = W^T * W, shape = [hid_dim, hid_dim]
# print(X.shape)
# X = X.double()
# X_inv = torch.linalg.inv(X) # X_inv = (W^T * W)^(-1), shape = [hid_dim, hid_dim] 
# print(X_inv.shape)
# print(X_inv)
# A = torch.matmul(X_inv, torch.transpose(W.double(), 1, 0)).cpu() # A = (W^T * W)^(-1) * W^T, shape = [hid_dim, 32000]
# print(A.shape)
# torch.save(A, '/root/autodl-tmp/zhaoyi/knowledge_locate/rome/my_exps/cache/A.pt')
# raise Exception
W = mt.model.state_dict()['lm_head.weight'] # W * h = v, shape = [32000, hid_dim]
if use_t == False:
    A = torch.load('/root/autodl-tmp/zhaoyi/knowledge_locate/rome/my_exps/cache/A.pt').cuda() # A.shape = [hid_dim, 32000]
else:
    A = torch.transpose(W, 1, 0).double()

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

def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    implicit_toks, # toks to debias
    explicit_toks, # toks to check 
    ):
    tgt_toks = implicit_toks

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)


    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        '''
        tgt_toks = [tok_enc_1, tok_enc_2, ...]
        '''
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x) # h.ahspe = [bs, seq_len, hid_dim]
        for t in patch_spec[layer]:
            # t is traced tok
            v = torch.matmul(W, h[0, t]) # v.shape = [32000, 1] or [32000]?
            if len(v.shape) == 1:
                v = v.unsqueeze(1) # v.shape = [32000, 1]
            v_min = torch.min(v) # v_min.shape = []
            v_max = torch.max(v)
            delta_v = torch.zeros_like(v) # delta_v.shape = [32000, 1]
            delta_v_compare = torch.zeros_like(v)

            for k in tgt_toks:
                if mode == "suppress":
                    delta_v[k, 0] = v_min - v[k,0]
                elif mode == "enhance":
                    delta_v[k, 0] = max(100*v_max, 2*v[k,0]) - v[k,0]
                else:
                    raise Exception("Unexpected Mode:"+mode)

            comparsions = random.sample(range(0, vocab_size), len(tgt_toks))
            
            for k in comparsions:
                if mode == "suppress":
                    delta_v_compare[k, 0] = v_min - v[k,0]
                elif mode == "enhance":
                    delta_v_compare[k, 0] = max(100*v_max, 2*v[k,0]) - v[k,0]
                else:
                    raise Exception("Unexpected Mode:"+mode)

            delta_h = torch.matmul(A, delta_v.double()).squeeze(1) # delta_h.shape = [hid_dim]
            delta_h_compare = torch.matmul(A, delta_v_compare.double()).squeeze(1)

            h[0, t] += delta_h.half() # shape = [hid_dim]
            h[1, t] += delta_h_compare.half()

            # double check projection
            if recheck:
                v_update = torch.matmul(W, h[0, t]) # v_update.shape = [32000, 1] or [32000]
                if len(v_update.shape) == 1:
                    v_update = v_update.unsqueeze(1)
            
            
                # v_delta_check = v_update - v # shape = [32000, 1]
                # v_delta_check = v_delta_check.squeeze(1) # shape = [32000]

                # print('--- implicit tokens and their changed probabilities ---')
                # for k in tgt_toks:
                #     print(mt.tokenizer.decode(k), v_delta_check[k])

                # v_delta_sort, idx = torch.sort(v_delta_check, descending=False)

                # print('--- top 5 changed probabilities ---')
                # for i in range(20):
                #     print(mt.tokenizer.decode(idx[i]), v_delta_sort[i])
                
                # print('--- explicit tokens and their changed probabilities ---')
                # for k in explicit_toks:
                #     for i in range(len(idx)):
                #         if idx[i] == k:
                #             print(mt.tokenizer.decode(k), v_delta_check[k], 'rank=', i)

                v_delta_recheck = v - v_update # shape = [32000, 1]
                v_delta_2 = torch.zeros_like(v_delta_recheck)
                for i in explicit_toks:
                    v_delta_2[i] = rc_strength * v_delta_recheck[i]
                    print(mt.tokenizer.decode(i), v_delta_recheck[i]) # influence on explicit toks brought by interventing on implicit toks
                delta_h_2 = torch.matmul(A, v_delta_2.double()).squeeze(1)
                h[0, t] += delta_h_2.half()


                v_update = torch.matmul(W, h[0, t]) # v_update.shape = [32000, 1] or [32000]
                if len(v_update.shape) == 1:
                    v_update = v_update.unsqueeze(1)
                v_delta_check = torch.softmax(v_update, dim=0) - torch.softmax(v, dim=0) # shape = [32000, 1]
                v_delta_check = v_delta_check.squeeze(1) # shape = [32000]

                print('--- implicit tokens and their changed probabilities ---')
                for k in tgt_toks:
                    print(mt.tokenizer.decode(k), v_delta_check[k])

                v_delta_sort, idx = torch.sort(v_delta_check, descending=False)

                print('--- top 5 changed probabilities ---')
                for i in range(20):
                    print(mt.tokenizer.decode(idx[i]), v_delta_sort[i])
                
                print('--- explicit tokens and their changed probabilities ---')
                for k in explicit_toks:
                    for i in range(len(idx)):
                        if idx[i] == k:
                            print(mt.tokenizer.decode(k), v_delta_check[k], 'rank=', i)

                # raise Exception('debug')
            

        return x

    # With the patching rules defined, run the patched model in inference.

    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp,output_hidden_states=True)["hidden_states"] # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
    assert outputs_exp[0].shape[0] == 2 # exp, compare
    # We report softmax probabilities for the answers_t token predictions of interest.
    for layer_idx in range(len(outputs_exp)):
        debias_proj = torch.matmul(W, outputs_exp[layer_idx][0, -1, :])
        compare_proj = torch.matmul(W, outputs_exp[layer_idx][1, -1, :])
        debias_logits = torch.softmax(debias_proj, dim=0).tolist()
        compare_logits = torch.softmax(compare_proj, dim=0).tolist()
        debias_prob = 0
        compare_prob = 0

        for tok in explicit_toks:
            # print(tok)
            # print(debias_logits[tok], compare_logits[tok])
            debias_prob += debias_logits[tok]
            compare_prob += compare_logits[tok]

        debias_prob /= len(explicit_toks)
        compare_prob /= len(explicit_toks)

        if layer_idx == 0:
            debias_lens = [debias_prob]
            compare_lens = [compare_prob]
        else:
            debias_lens.append(debias_prob)
            compare_lens.append(compare_prob)

        
    return debias_lens, compare_lens

def calculate_hidden_flow(
    mt, prompt, check_tok_ids, implicit_toks, explicit_toks, trace_layers
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
            projs = torch.matmul(W, out[layer_idx][0, -1])
            logits = torch.softmax(projs, dim=0).tolist()
            origin_prob = 0
            for tok in explicit_toks:
                origin_prob += logits[tok]
            origin_prob /= len(explicit_toks)
            if layer_idx == 0:
                origin_lens = [origin_prob]
            else:
                origin_lens.append(origin_prob)


    results = debias_states(mt.model, inp, check_tok_ids, implicit_toks, explicit_toks, trace_layers)
    
    return origin_lens, results


def debias_states(model, inp, check_tok_ids, implicit_toks, explicit_toks, trace_layers):
    table = list()
    for tnum in check_tok_ids:

        line = dict()
        line["debias_lens"] = list()
        line["compare_lens"] = list()
        # for layer in range(0, num_layers):
        for layer in trace_layers:
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                implicit_toks,
                explicit_toks
            )
            debias_lens, compare_lens = r
            line["debias_lens"].append(debias_lens)
            line["compare_lens"].append(compare_lens)

        table.append(line)
    return table


def get_correlation(
    mt, prompt, implicit_toks_dict, explicit_toks_dict, trace_layers
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
        implicit_lens = dict()
        explicit_lens = dict()
        for layer_idx in range(len(out)):
            projs = torch.matmul(W, out[layer_idx][0, -1])
            logits = torch.softmax(projs, dim=0).tolist()
            origin_prob = 0
            for implicit_answer in implicit_toks_dict.keys():
                origin_prob = 0
                implicit_toks = implicit_toks_dict[implicit_answer]

                for tok in implicit_toks:
                    origin_prob += logits[tok]
                origin_prob /= len(implicit_toks)

                if layer_idx not in implicit_lens:
                    implicit_lens[layer_idx] = [origin_prob]
                else:
                    implicit_lens[layer_idx].append(origin_prob)


            origin_prob = 0
            for explicit_answer in explicit_toks_dict.keys():
                origin_prob = 0
                explicit_toks = explicit_toks_dict[explicit_answer]

                for tok in explicit_toks:
                    origin_prob += logits[tok]
                origin_prob /= len(explicit_toks)

                if layer_idx not in explicit_lens:
                    explicit_lens[layer_idx] = [origin_prob]
                else:
                    explicit_lens[layer_idx].append(origin_prob)

                    
    return implicit_lens, explicit_lens


def get_tgt_tok_id(prefix):
    inner_prefix = make_inputs(mt.tokenizer, [prefix])
    inner_toks_prefix = [mt.tokenizer.decode(inner_prefix["input_ids"][0][i]) for i in range(inner_prefix["input_ids"].shape[1])]
    return len(inner_toks_prefix) - 1

last_tok_id = get_tgt_tok_id(prompt)
last_inner_s_id = get_tgt_tok_id(prefix_inner_s)
check_tok_ids = [last_tok_id]

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
        

implicit_toks_dict = dict()
explicit_toks_dict = dict()
for implicit_answer in implicit_answer_cands:
    implicit_toks_dict[implicit_answer] = enc_tok(implicit_answer, avg=average)
for explicit_answer in explicit_answer_cands:
    explicit_toks_dict[explicit_answer] = enc_tok(explicit_answer, avg=average)


implicit_lens, explicit_lens = get_correlation(mt, prompt, implicit_toks_dict, explicit_toks_dict, trace_layers)


correlation = list()
for implicit_layer in implicit_lens.keys():
    implicit_corr = list()
    for explicit_layer in explicit_lens.keys():
        corr_value = stats.spearmanr(implicit_lens[implicit_layer], explicit_lens[explicit_layer])
        print(corr_value)
        implicit_corr.append(corr_value[0])
    correlation.append(implicit_corr)

correlation = np.array(correlation)

# plot prob variance



plt.figure()
plot_data = pd.DataFrame(correlation[20:,20:],  [i for i in range(20, len(list(implicit_lens.keys())))], [i for i in range(20, len(list(implicit_lens.keys())))])
plot = sns.heatmap(plot_data)
plt.xlabel("explicit layer")
plt.ylabel("implicit layer")
plt.title("Correlation between explicit answers and implicit answers")
# plt.savefig('test.mask.2.png')
plt.savefig('cor.'+prompt+'.png')

