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
use_t = False
recheck = True
rc_strength = 10
mode = "suppress" # or "enhance"
# mode = "enhance"
mode_str = "_" + mode[0]

exp_name = 'new_overall'
write_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/debias_causal_intervent/datasets/"+exp_name+'.suf.taketime.3.json'
fw = open(write_path, "w")

data_path = "/root/autodl-tmp/zhaoyi/knowledge_locate/datasets/MQuAKE/datasets/new_debias/suf.groundtruth.json"
fr = open(data_path, "r")
data = json.load(fr)


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
                    delta_v[k, 0] = max(v_max, 2*v[k,0]) - v[k,0]
                else:
                    raise Exception("Unexpected Mode:"+mode)

            comparsions = random.sample(range(0, vocab_size), len(tgt_toks))
            
            for k in comparsions:
                if mode == "suppress":
                    delta_v_compare[k, 0] = v_min - v[k,0]
                elif mode == "enhance":
                    delta_v_compare[k, 0] = max(v_max, 2*v[k,0]) - v[k,0]
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
        outputs_exp = model(**inp) # outputs_exp.logits.shape = [bs(=2), seq_len, vocab_size]
    assert outputs_exp.logits.shape[0] == 2 # exp, compare
    # We report softmax probabilities for the answers_t token predictions of interest.
    debias_logits = torch.softmax(outputs_exp.logits[0, -1, :], dim=0).tolist()
    compare_logits = torch.softmax(outputs_exp.logits[1, -1, :], dim=0).tolist()
    debias_prob = 0
    compare_prob = 0

    for tok in explicit_toks:
        # print(tok)
        # print(debias_logits[tok], compare_logits[tok])
        debias_prob += debias_logits[tok]
        compare_prob += compare_logits[tok]

    debias_prob /= len(explicit_toks)
    compare_prob /= len(explicit_toks)

    
    debias_rank = get_rank(debias_logits, explicit_toks)
    compare_rank = get_rank(compare_logits, explicit_toks)

        
    return debias_prob, debias_rank, compare_prob, compare_rank

def calculate_hidden_flow(
    mt, prompt, check_tok_ids, implicit_toks, explicit_toks,
    ):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * 2)
    with torch.no_grad():
        out = mt.model(**inp)["logits"] # out.shape = (2, seq_len, vocab_size)
        logits = torch.softmax(out[0, -1], dim=0).tolist() # len = vocab_size
        origin_prob = 0
        for tok in explicit_toks:
            origin_prob += logits[tok]
        origin_prob /= len(explicit_toks)
        origin_rank = get_rank(logits, explicit_toks)

    results = debias_states(mt.model, mt.num_layers, inp, check_tok_ids, implicit_toks, explicit_toks)
    
    return origin_prob, origin_rank, results


def debias_states(model, num_layers, inp, check_tok_ids, implicit_toks, explicit_toks):
    table = list()
    for tnum in check_tok_ids:

        line = dict()
        line["debias_prob"] = list()
        line["debias_rank"] = list()
        line["compare_prob"] = list()
        line["compare_rank"] = list()
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                implicit_toks,
                explicit_toks
            )
            debias_prob, debias_rank, compare_prob, compare_rank = r
            line["debias_prob"].append(debias_prob)
            line["debias_rank"].append(debias_rank)
            line["compare_prob"].append(compare_prob)
            line["compare_rank"].append(compare_rank)

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

def plot(x, ys, labels, x_label, y_label, title='test', save_path=None):
    colors = ['steelblue','darkred','darkgreen', 'gold', 'orchid', "grey", "red", "lime"]
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

for datum in data:
    output_datum = dict()
    for key in datum:
        output_datum[key] = datum[key]

    for inp_mode in ['comp', 'single']:
    # for inp_mode in ['comp']:
        prompt = datum[inp_mode+' cloze']
        implicit_answer = datum['implicit result']
        explicit_answer = datum['explicit result']

        last_tok_id = get_tgt_tok_id(prompt)
        check_tok_ids = [last_tok_id]
        implicit_toks = enc_tok(implicit_answer, avg=average)
        explicit_toks = enc_tok(explicit_answer, avg=average)
        print(implicit_toks)
        print(explicit_toks)
        origin_prob, origin_rank, results = calculate_hidden_flow(mt, prompt, check_tok_ids, implicit_toks, explicit_toks)
        last_tok_data = results[0]
        x = list(range(len(last_tok_data["debias_prob"])))
        baseline_prob = [origin_prob for i in range(len(x))]
        baseline_rank = [origin_rank for i in range(len(x))]
        # plot prob variance

        baseline_prob = [str(e) for e in baseline_prob]
        debias_prob = [str(e) for e in last_tok_data["debias_prob"]]
        compare_prob = [str(e) for e in last_tok_data["compare_prob"]]

        output_datum[inp_mode+' baseline_prob'] = '@'.join(baseline_prob)

        output_datum[inp_mode+' debias_prob'] = '@'.join(debias_prob)

        output_datum[inp_mode+' compare_prob'] = '@'.join(compare_prob)

    output_data.append(output_datum)
    
    

json.dump(output_data, fw, indent=4)
# plot(x, probs, labels, "layer", "probability", "Prob Variance: "+implicit_answer+'@'+explicit_answer, prob_path)
# plot(x, ranks, labels, "layer", "rank", "Rank Variance: "+implicit_answer+'@'+explicit_answer, rank_path)

