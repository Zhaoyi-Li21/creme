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

# only for llama-2-7b
NUM_HEADS = 32
HEAD_DIM = 128

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

save_dir = "/root/autodl-tmp/zhaoyi/knowledge_locate/rewrite_guide/head_batch"
exp_name = "wpred.dedup.switch.Biden.prefix"
window_size = 2
if window_size > 0:
    exp_name = exp_name + "_wd" + str(window_size)

try_one = False

# prompt = "The home country of the sport associated with Giorgio Chinaglia is"
# prompt_subj = "The home country of the sport associated with Giorgio Chinaglia"
# prompt_guide = "The home country of association football is"
# prompt_guide_subj = "The home country of association football"
# implicit_answer = "association football"
# explicit_answer = "England"


# prompt = "The head of state of the country where ORLAN holds citizenship is"
# prompt_subj = "The head of state of the country where ORLAN holds citizenship"
# prompt_guide = "France is"
# prompt_guide_subj = "France"
# implicit_answer = "France"
# explicit_answer = "Emmanuel Macron"


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

def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    head,
    implicit_toks, # toks to debias
    explicit_toks, # toks to check 
    ):

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
        h = untuple(x) # h.shape = [bs, seq_len, hid_dim]
        # print(h)
        # print(type(h))
        assert type(h) == torch.Tensor
        assert h.shape[2] == NUM_HEADS * HEAD_DIM
        for t in patch_spec[layer]:
            h[0, t, head * HEAD_DIM : (head+1) * HEAD_DIM] = h[1, t, head * HEAD_DIM : (head+1) * HEAD_DIM]

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
    mt, prompt, check_tok_ids, implicit_toks, explicit_toks, head=0,
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

    if window_size == 0:
        attn_head_results = trace_important_states(mt.model, mt.num_layers, inp, check_tok_ids, implicit_toks, explicit_toks, kind="attn_head")
    else:
        attn_head_results = trace_important_window(mt.model, mt.num_layers, inp, check_tok_ids, implicit_toks, explicit_toks, window_size=window_size, head=head)

    return origin_prob, origin_rank, attn_head_results

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

def trace_important_window(model, num_layers, inp, check_tok_ids, implicit_toks, explicit_toks, window_size=6, head=0):
        table = list()
        for tnum in check_tok_ids:
            line = dict()
            line["debias_prob"] = list()
            line["debias_rank"] = list()
            for layer in range(14, 25):

                layerlist = list()
                for kind in ["attn_head_q", "attn_head_k", "attn_head_v"]:
                    for L in range(max(0, layer - window_size // 2), min(num_layers, layer - (-window_size // 2))):
                        layerlist.append((tnum, layername(model, L, kind)))

                r = trace_with_patch(
                    model,
                    inp,
                    layerlist,
                    head,
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

def exp(prompt, prompt_guide, prompt_subj, prompt_guide_subj, implicit_answer, explicit_answer, head):
    last_tok_id_1 = get_tgt_tok_id(prompt)
    last_tok_id_2 = get_tgt_tok_id(prompt_guide)
    subj_tok_id_1 = get_tgt_tok_id(prompt_subj)
    subj_tok_id_2 = get_tgt_tok_id(prompt_guide_subj)
    last_tok_id = max(last_tok_id_1, last_tok_id_2)
    subj_tok_id = max(subj_tok_id_1, subj_tok_id_2)
    # check_tok_ids = [last_tok_id, subj_tok_id]
    check_tok_ids = [last_tok_id]
    print('check_tok_ids:', check_tok_ids)


            
    implicit_toks = enc_tok(implicit_answer, avg=average)
    explicit_toks = enc_tok(explicit_answer, avg=average)
    print(implicit_toks)
    print(explicit_toks)

    origin_prob, origin_rank, attn_head_results = calculate_hidden_flow(mt, prompt, check_tok_ids, implicit_toks, explicit_toks, head)
    
    last_tok_data = attn_head_results[0]
    # subj_tok_data = attn_head_results[1]

    x = list(range(len(last_tok_data["debias_prob"])))
    baseline_prob = [origin_prob for i in range(len(x))]
    baseline_rank = [origin_rank for i in range(len(x))]
    
    return last_tok_data, baseline_prob, baseline_rank

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


file_path_fail = "/root/autodl-tmp/zhaoyi/knowledge_locate/inference/MQuAKE/filter_2/pass_singles_fail_comp.wpred.dedup.switch.Biden.precloze.json"
fr_fail = open(file_path_fail, "r")
data_fail = json.load(fr_fail)

file_path_pass = "/root/autodl-tmp/zhaoyi/knowledge_locate/inference/MQuAKE/filter_2/pass_all.wpred.dedup.switch.Biden.precloze.json"
fr_pass = open(file_path_pass, "r")
data_pass = json.load(fr_pass)

for head in range(NUM_HEADS):
    output_data = list()
    file_name = exp_name +'_'+ model_name + '.head.'+ str(head) + '.json'
    save_path = save_dir + '/' +file_name
    fw = open(save_path,"w")
    for i in range(len(data_fail)):
        datum = data_fail[i]
        prompt = datum["comp cloze"]
        prompt_guide = datum["single cloze"]
        prompt_subj = datum["subj_prefix_comp"]
        prompt_guide_subj = datum["subj_prefix_single"]

        implicit_answer = datum["answers"][0]
        explicit_answer = datum["answers"][1]

        last_tok_data, baseline_prob, baseline_rank = exp(prompt, prompt_guide, prompt_subj, prompt_guide_subj, implicit_answer, explicit_answer, head)
        
        
        output_datum = dict()
        output_datum["flag"] = "Fail-normal-"+str(i)

        output_datum["last_hs_prob"] = ' '.join([str(e) for e in last_tok_data["debias_prob"]])
        output_datum["last_hs_rank"] = ' '.join([str(e) for e in last_tok_data["debias_rank"]])

        output_datum["baseline_prob"] = ' '.join([str(e) for e in baseline_prob])
        output_datum["baseline_rank"] = ' '.join([str(e) for e in baseline_rank])

        output_data.append(output_datum)

        if try_one:
            probs = [baseline_prob, last_tok_data["debias_prob"]]
            labels = ["baseline", "last_head_0"]
            x = list(range(len(last_tok_data["debias_prob"])))
            plot(x, probs, labels, "layer", "probability", prompt+'\n'+prompt_guide+'\n'+"Prob Variance: "+explicit_answer, 'test_1.head.png')
            break
        
        print('head:', head)
        print('first loop')
        print('processed data number:',i)


    for i in range(len(data_pass)):
        datum= data_pass[i]
        prompt = datum["comp cloze"]
        prompt_guide = datum["single cloze"]
        prompt_subj = datum["subj_prefix_comp"]
        prompt_guide_subj = datum["subj_prefix_single"]

        implicit_answer = datum["answers"][0]
        explicit_answer = datum["answers"][1]

        last_tok_data, baseline_prob, baseline_rank = exp(prompt, prompt_guide, prompt_subj, prompt_guide_subj, implicit_answer, explicit_answer, head)
        

        output_datum = dict()
        output_datum["flag"] = "Pass-normal-"+str(i)
        output_datum["last_hs_prob"] = ' '.join([str(e) for e in last_tok_data["debias_prob"]])
        output_datum["last_hs_rank"] = ' '.join([str(e) for e in last_tok_data["debias_rank"]])

        output_datum["baseline_prob"] = ' '.join([str(e) for e in baseline_prob])
        output_datum["baseline_rank"] = ' '.join([str(e) for e in baseline_rank])

        output_data.append(output_datum)

        if try_one:
            probs = [baseline_prob,last_tok_data["debias_prob"]]

            labels = ["baseline", "last_head_0", "subj_head_0"]
            x = list(range(len(last_tok_data["debias_prob"])))
            plot(x, probs, labels, "layer", "probability", prompt+'\n'+prompt_guide+'\n'+"Prob Variance: "+explicit_answer, 'test_3.head.png')
            break

        print('head:', head)
        print('second loop')
        print('processed data number:',i)



    json.dump(output_data, fw, indent=4)
    if try_one:
        break