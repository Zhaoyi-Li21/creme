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
save_dir = "/root/autodl-tmp/zhaoyi/knowledge_locate/rewrite_guide/the_whole_state"
# exp_name = "test_11_only_subj_guide"
exp_name = "test_2"
window_size = 6

if average:
    exp_name = exp_name + "_avg"
if window_size > 0:
    exp_name = exp_name + "_wd" + str(window_size)

exp_name = exp_name +'_'+ model_name

if os.path.exists(save_dir+'/'+exp_name) == False:
    # mkdir
    os.mkdir(save_dir+'/'+exp_name)

save_dir = save_dir + "/" + exp_name + "/" 
prob_path = save_dir + "prob.png"
rank_path = save_dir + "rank.png"

prompt = "The home country of the sport associated with Giorgio Chinaglia is"
prompt_subj = "The home country of the sport associated with Giorgio Chinaglia"
prompt_guide = "The home country of association football is"
prompt_guide_subj = "The home country of association football"
implicit_answer = "association football"
explicit_answer = "England"

# implicit_answer = "Giorgio Chinaglia"
# explicit_answer = "Italy"

prompt = "[MASK] is"
prompt_subj = "[MASK]"
prompt_guide = "The home country of association football is"
prompt_guide_subj = "The home country of association football"
implicit_answer = "association football"
explicit_answer = "England"

# prompt = "The home country of the sport [MASK] is"
# prompt_guide = "The home country of association football is"
# implicit_answer = "association football"
# explicit_answer = "England"

# prompt = "The [MASK] of the sport associated with Giorgio Chinaglia is"
# prompt_guide = "The home country of association football is"
# implicit_answer = "association football"
# explicit_answer = "England"


# prompt = "The home country of the sport associated with Giorgio Chinaglia is"
# prompt_subj = "The home country of the sport associated with Giorgio Chinaglia"
# prompt_guide = "association football is"
# prompt_guide_subj = "association football"
# implicit_answer = "association football"
# explicit_answer = "England"


# prompt = "The head of state of the country where ORLAN holds citizenship is"
# prompt_subj = "The head of state of the country where ORLAN holds citizenship"
# prompt_guide = "The head of state of France is"
# prompt_guide_subj = "The head of state of France"
# implicit_answer = "France"
# explicit_answer = "Emmanuel Macron"

# prompt = "The head of state of the country where ORLAN holds citizenship is"
# prompt_subj = "The head of state of the country where ORLAN holds citizenship"
# prompt_guide = "France is"
# prompt_guide_subj = "France"
# implicit_answer = "France"
# explicit_answer = "Emmanuel Macron"

# prompt = "[MASK] is"
# prompt_guide = "The head of state of France is"
# prompt_subj = "[MASK]"
# prompt_guide_subj = "The head of state of France"
# implicit_answer = "France"
# explicit_answer = "Emmanuel Macron"

# prompt = "The head of state of the country [MASK] is"
# prompt_guide = "The head of state of France is"
# prompt_subj = "The head of state of the country [MASK]"
# prompt_guide_subj = "The head of state of France"
# implicit_answer = "France"
# explicit_answer = "Emmanuel Macron"

# prompt = "The [MASK] of the country where ORLAN holds citizenship is"
# prompt_guide = "The head of state of France is"
# implicit_answer = "France"
# explicit_answer = "Emmanuel Macron"

# prompt = "The name of the current head of state of the country which Josh Barnett is a citizen of is" 
# prompt_subj = "The name of the current head of state of the country which Josh Barnett is a citizen of"
# prompt_guide = "The name of the current head of state of United States of America is"
# prompt_guide_subj = "The name of the current head of state of United States of America"
# implicit_answer = "United States of America"
# explicit_answer = "Donald Trump"

# prompt = "[MASK] is" 
# prompt_guide = "The name of the current head of state of United States of America is"
# implicit_answer = "United States of America"
# explicit_answer = "Donald Trump"

# prompt = "[MASK] is" 
# prompt_subj = "[MASK]"
# prompt_guide = "The name of the current head of state of United States of America is"
# prompt_guide_subj = "The name of the current head of state of United States of America"
# implicit_answer = "United States of America"
# explicit_answer = "Donald Trump"


# prompt = "The name of the current head of state of the country [MASK] is" 
# prompt_subj = "The name of the current head of state of the country [MASK]"
# prompt_guide = "The name of the current head of state of United States of America is"
# prompt_guide_subj = "The name of the current head of state of United States of America"
# implicit_answer = "United States of America"
# explicit_answer = "Donald Trump"

# prompt = "The name of [MASK] of the country which Josh Barnett is a citizen of is" 
# prompt_guide = "The name of the current head of state of United States of America is"
# implicit_answer = "United States of America"
# explicit_answer = "Donald Trump"

# prompt = "The country that the creator of C. Auguste Dupin belongs to is"
# prompt_subj = "The country that the creator of C. Auguste Dupin"
# prompt_guide = "The country that Edgar Allan Poe belongs to is"
# prompt_guide_subj = "The country that Edgar Allan Poe"
# implicit_answer = "Edgar Allan Poe"
# explicit_answer = "United States of America"

# prompt = "The country that the creator of C. Auguste Dupin belongs to is"
# prompt_subj = "The country that the creator of C. Auguste Dupin"
# prompt_guide = "Edgar Allan Poe belongs to is"
# prompt_guide_subj = "Edgar Allan Poe"
# implicit_answer = "Edgar Allan Poe"
# explicit_answer = "United States of America"

# prompt = "The first African-American president was succeeded by"
# prompt_subj = "The first African-American president"
# prompt_guide = "Barack Obama was succeeded by"
# prompt_guide_subj = "Barack Obama"
# implicit_answer = "Barack Obama"
# explicit_answer = "Donald Trump"

# prompt = "The sport that the position that Ty Detmer plays of is associated with is"
# prompt_subj = "The sport that the position that Ty Detmer plays of"
# prompt_guide = "The sport that quarterback is associated with is"
# prompt_guide_subj = "The sport that quarterback"
# implicit_answer = "quarterback"
# explicit_answer = "American football"

W = mt.model.state_dict()['lm_head.weight'] # W * h = v, shape = [32000, hid_dim]


def get_rank(logits, check_tok_enc, first=True):
                    # first = True: only trace the rank for the first token
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
            h[0, t] = h[1, t]

        return x

    # With the patching rules defined, run the patched model in inference.

    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
    assert outputs_exp.logits.shape[0] == 2 # exp, compare
    # We report softmax probabilities for the answers_t token predictions of interest.
    debias_logits = torch.softmax(outputs_exp.logits[0, -1, :], dim=0).tolist()

    debias_prob = 0


    for tok in explicit_toks:
        # print(tok)
        # print(debias_logits[tok], compare_logits[tok])
        debias_prob += debias_logits[tok]

    debias_prob /= len(explicit_toks)    
    debias_rank = get_rank(debias_logits, explicit_toks)   

    return debias_prob, debias_rank

def calculate_hidden_flow(
    mt, prompt, check_tok_ids, implicit_toks, explicit_toks,
    ):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt, prompt_guide])
    with torch.no_grad():
        out = mt.model(**inp)["logits"] # out.shape = (2, seq_len, vocab_size)
        logits = torch.softmax(out[0, -1], dim=0).tolist() # len = vocab_size
        origin_prob = 0
        for tok in explicit_toks:
            origin_prob += logits[tok]
        origin_prob /= len(explicit_toks)
        origin_rank = get_rank(logits, explicit_toks)

    hs_results = trace_important_states(mt.model, mt.num_layers, inp, check_tok_ids, implicit_toks, explicit_toks, kind=None)
    if window_size == 0:
        mlp_results = trace_important_states(mt.model, mt.num_layers, inp, check_tok_ids, implicit_toks, explicit_toks, kind="mlp")
        attn_results = trace_important_states(mt.model, mt.num_layers, inp, check_tok_ids, implicit_toks, explicit_toks, kind="attn")
    else:
        mlp_results = trace_important_window(mt.model, mt.num_layers, inp, check_tok_ids, implicit_toks, explicit_toks, kind="mlp", window_size=window_size)
        attn_results = trace_important_window(mt.model, mt.num_layers, inp, check_tok_ids, implicit_toks, explicit_toks, kind="attn", window_size=window_size)

    return origin_prob, origin_rank, hs_results, mlp_results, attn_results


def trace_important_states(model, num_layers, inp, check_tok_ids, implicit_toks, explicit_toks, kind=None):
    table = list()
    for tnum in check_tok_ids:
        line = dict()
        line["debias_prob"] = list()
        line["debias_rank"] = list()
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer, kind))],
                implicit_toks,
                explicit_toks
            )
            debias_prob, debias_rank = r
            line["debias_prob"].append(debias_prob)
            line["debias_rank"].append(debias_rank)
        table.append(line)
    return table

def trace_important_window(model, num_layers, inp, check_tok_ids, implicit_toks, explicit_toks, kind=None, window_size=6):
    table = list()
    for tnum in check_tok_ids:
        line = dict()
        line["debias_prob"] = list()
        line["debias_rank"] = list()
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window_size // 2), min(num_layers, layer - (-window_size // 2))
                )
            ]
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                implicit_toks,
                explicit_toks
            )
            debias_prob, debias_rank = r
            line["debias_prob"].append(debias_prob)
            line["debias_rank"].append(debias_rank)
        table.append(line)
    return table



def get_tgt_tok_id(prefix):
    inner_prefix = make_inputs(mt.tokenizer, [prefix])
    inner_toks_prefix = [mt.tokenizer.decode(inner_prefix["input_ids"][0][i]) for i in range(inner_prefix["input_ids"].shape[1])]
    return len(inner_toks_prefix) - 1

last_tok_id_1 = get_tgt_tok_id(prompt)
last_tok_id_2 = get_tgt_tok_id(prompt_guide)
subj_tok_id_1 = get_tgt_tok_id(prompt_subj)
subj_tok_id_2 = get_tgt_tok_id(prompt_guide_subj)
last_tok_id = max(last_tok_id_1, last_tok_id_2)
subj_tok_id = max(subj_tok_id_1, subj_tok_id_2)
check_tok_ids = [last_tok_id, subj_tok_id]
print('check_tok_ids:', check_tok_ids)

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
    plt.savefig(save_path, bbox_inches ='tight')
    plt.clf()

probs = [baseline_prob, last_tok_data["debias_prob"], mlp_last_tok_data["debias_prob"], attn_last_tok_data["debias_prob"],
         subj_tok_data["debias_prob"], mlp_subj_tok_data["debias_prob"], attn_subj_tok_data["debias_prob"]]

ranks = [baseline_rank, last_tok_data["debias_rank"], mlp_last_tok_data["debias_rank"], attn_last_tok_data["debias_rank"],
         subj_tok_data["debias_rank"], mlp_subj_tok_data["debias_rank"], attn_subj_tok_data["debias_rank"]]

labels = ["baseline", "last_hidden_state", "last_mlp", "last_attention", "subj_hidden_state", "subj_mlp", "subj_attention"]

plot(x, probs, labels, "layer", "probability", prompt+'\n'+prompt_guide+'\n'+"Prob Variance: "+explicit_answer, prob_path)
plot(x, ranks, labels, "layer", "rank", prompt+'\n'+prompt_guide+'\n'+"Rank Variance: "+explicit_answer, rank_path)