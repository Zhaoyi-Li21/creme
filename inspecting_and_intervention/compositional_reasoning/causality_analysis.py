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

save_dir = "/root/autodl-tmp/zhaoyi/knowledge_locate/debias_causal_intervent/new_cases_2"

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

# test(avg)
# prompt = "The country of citizenship of the director of Lilli's Marriage is" # test
# prefix_inner_s = "The country of citizenship of the director of Lilli's Marriage"
# implicit_answer = "Jaap Speyer"
# explicit_answer = "Dutch"


prompt = "The country of citizenship of Jaap Speyer is" # new_test_4
prefix_inner_s = "The country of citizenship of Jaap Speyer"
implicit_answer = "Jaap Speyer"
explicit_answer = "Dutch"

prompt = "The head of state of the country that Ellie Kemper is a citizen of" # new_test_5
prefix_inner_s = "The head of state of the country"
prompt = "[MASK]" # new_test_6
prefix_inner_s = "[MASK]"
implicit_answer = "America"
explicit_answer = "Donald Trump"

prompt = "The nationality of the author of 'Misery' is" # new_test_7
prefix_inner_s = "The nationality of the author of 'Misery'"
implicit_answer = "Stephen King"
explicit_answer = "America"

prompt = "The capital city of the country that Michael Feinstein is a citizen of is" # new_test_8
prefix_inner_s = "The capital city of the country that Michael Feinstein is a citizen of"
implicit_answer = "USA" # new_test_9 for USA, new_test_8 for America
explicit_answer = "Washington D.C."
# implicit_answer = "China" # new_test_10
# explicit_answer = "Beijing"
implicit_answer = "France" # new_test_12
explicit_answer = "Paris"
implicit_answer = "German" # new_test_13
explicit_answer = "Berlin"
implicit_answer = "India"
explicit_answer = "New Delhi" # new_test_14
implicit_answer = "Canada"
explicit_answer = "Toronto" # new_test_15, lamda = 12
implicit_answer = "America" # new_test_16
explicit_answer = "New York"
implicit_answer = "America"
explicit_answer = "Washington" # new_test_17
# implicit_answer = "Japan"
# explicit_answer = "Tokyo" # new_test_18

# prompt = "The capital city of the country that has the largest area is"
# prefix_inner_s = "The capital city of the country that has the largest area"
# implicit_answer = "America"
# explicit_answer = "Washington" # new_test_19

# prompt = "The capital city of the country that has the largest population is"
# prefix_inner_s = "The capital city of the country that has the largest population"
# implicit_answer = "America"
# explicit_answer = "Washington" # new_test_20

prompt = "The capital of the country to which Gato Barbieri belonged is"
prefix_inner_s = "The capital of the country to which Gato Barbieri belonged"
implicit_answer = "America"
explicit_answer = "Washington" # new_test_21
implicit_answer = "China"
explicit_answer = "Beijing" # new_test_22

prompt = "China"
prefix_inner_s = "China"
implicit_answer = "China"
explicit_answer = "China" # new_test_23, for rc v.s. non-rc


prompt = "the head of state of the country where Ellie Kemper holds a citizenship is"
prefix_inner_s = "the head of state of the country where Ellie Kemper holds a citizenship"
implicit_answer = "America"
explicit_answer = "Donald Trump" # new_test_24, rc = 5

prompt = "The nationality of Stephen King is" # new_test_25
prefix_inner_s = "The nationality of Stephen King"
implicit_answer = "Stephen King"
explicit_answer = "America" 

prompt = "The continent that Emma Bunton's country of citizenship belongs to is"
prefix_inner_s = "The continent that Emma Bunton's country of citizenship belongs to"
implicit_answer = "United kingdom" # new_test_26
implicit_answer = "England" # new_test_27
explicit_answer = "Europe"
implicit_answer = "China" # new_test_28
explicit_answer = "Asia"
implicit_answer = "China" # new_test_29
explicit_answer = "Europe"

prompt = "The capital of [MASK] is"
prefix_inner_s = "The capital"
implicit_answer = "China" # new_test_30
explicit_answer = "Beijing"
# implicit_answer = "America" # new_test_31
# explicit_answer = "Washington"

# prompt = "[MASK] of China is"
# prefix_inner_s = "[MASK] of China"
# implicit_answer = "China"
# explicit_answer = "Beijing" # new_test_32

# prompt = "[MASK]"
# prefix_inner_s = "[MASK]"
# implicit_answer = "China"
# explicit_answer = "Beijing" # new_test_33

# prompt = "The capital city of the country that has the largest population is"
# prefix_inner_s = "The capital city of the country that has the largest population"
# implicit_answer = "Beijing"
# explicit_answer = "Shanghai" # new_test_34


# prompt = "Michael Jordan played the sports of"
# prefix_inner_s = "Michael Jordan"
# implicit_answer = "Lionel Messi"
# explicit_answer = "soccer" # new_test_35

# prompt = "Michael Jordan played the sports of"
# prefix_inner_s = "Michael Jordan"
# implicit_answer = "Lionel Messi"
# explicit_answer = "basketball" # new_test_36

# prompt = "The capital city of Russia is"
# prefix_inner_s = "The capital city of Russia"
# implicit_answer = "China"
# explicit_answer = "Beijing" # new_test_37

# implicit_answer = "India"
# explicit_answer = "New Delhi" # new_test_38

# implicit_answer = "China"
# explicit_answer = "New Delhi" # new_test_39

# implicit_answer = "Russia"
# explicit_answer = "Moscow" # new_test_40

# prompt = "The capital city of the country that has the largest population is"
# prefix_inner_s = "The capital city of the country that has the largest population"
# implicit_answer = "China"
# explicit_answer = "Shanghai" # new_test_41

prompt = "[MASK]"
prefix_inner_s = "[MASK]"
implicit_answer = "Lionel Messi"
explicit_answer = "soccer" # new_test_42

implicit_answer = "China"
explicit_answer = "Asia" # new_test_43

implicit_answer = "China"
explicit_answer = "Russia" # new_test_44

implicit_answer = "America"
explicit_answer = "Trump" # new_test_45

prompt = "the official language of the country to which Herb Caen belongs is"
prefix_inner_s = "the official language of the country to which Herb Caen belongs"
implicit_answer = "America"
explicit_answer = "English" # new_test_46

prompt = "the country that the sport associated with KK Crvena Zvezda originate from is"
prefix_inner_s = "the country that the sport associated with KK Crvena Zvezda originate from"
implicit_answer = "basketball"
explicit_answer = "America" # new_test_47


prompt = "the continent that Thomas Graham Jackson born in is"
prefix_inner_s = "the continent that Thomas Graham Jackson born in is"
implicit_answer = "London"
explicit_answer = "Europe" # new_test_48

prompt = "The capital of the country to which Gato Barbieri belonged is"
prefix_inner_s = "The capital of the country to which Gato Barbieri belonged"
implicit_answer = "United States"
explicit_answer = "Washington" # new_test_49

implicit_answer = "United States of America"
explicit_answer = "Washington" # new_test_50

prompt = "The capital city of the country where Miranda Hart is a citizen is"
prefix_inner_s = "The capital city of the country where Miranda Hart is a citizen"
implicit_answer = "England"
explicit_answer = "London" # new_test_52

implicit_answer = "France"
explicit_answer = "Paris" # new_test_53

prompt = "The city that is the capital of the country where Lou Pearlman had citizenship is"
prefix_inner_s = "The city that is the capital of the country where Lou Pearlman had citizenship"

implicit_answer = "America"
explicit_answer = "Washington" # new_test_54


prompt = "The capital of the country where Lou Pearlman had citizenship is"
prefix_inner_s = "The capital of the country where Lou Pearlman had citizenship"

implicit_answer = "America"
explicit_answer = "Washington" # new_test_55

prompt = "The city that is the capital of the country where Lou Pearlman had citizenship is"
prefix_inner_s = "The city that is the capital of the country where Lou Pearlman had citizenship"

implicit_answer = "France"
explicit_answer = "Paris" # new_test_56

implicit_answer = "China"
explicit_answer = "Beijing" # new_test_57

prompt = "The city that is the capital of the country of citizenship of Henri Lefebvre is"
prefix_inner_s = "The city that is the capital of the country of citizenship of Henri Lefebvre"
implicit_answer = "China"
explicit_answer = "Beijing" # new_test_58


prompt = "The capital city of the country of citizenship of Henri Lefebvre is"
prefix_inner_s = "The capital city of the country of citizenship of Henri Lefebvre"
implicit_answer = "France"
explicit_answer = "Paris" # new_test_59
implicit_answer = "China"
explicit_answer = "Beijing" # new_test_60
implicit_answer = "France"
explicit_answer = "Paris" # new_test_61
implicit_answer = "Brazil"
explicit_answer = "Brasilia" # new_test_62

prompt = "The capital city of America is"
prefix_inner_s = "The capital city of America"
implicit_answer = "America"
explicit_answer = "Washington" # new_test_63

prompt = "The capital city of United States is"
prefix_inner_s = "The capital city of United States"
implicit_answer = "United States"
explicit_answer = "Washington" # new_test_64

prompt = "The capital city of China is"
prefix_inner_s = "The capital city of China"
implicit_answer = "China"
explicit_answer = "Beijing" # new_test_65

prompt = "The capital city of America is"
prefix_inner_s = "The capital city of America"
implicit_answer = "China"
explicit_answer = "Beijing" # new_test_66




# prompt = "President Barack Obama was succeeded by President"
# prefix_inner_s = "President Obama"
# implicit_answer = "Barack Obama"
# explicit_answer = "Donald Trump" # new_test_67

# prompt = "[MASK] of America is"
# prefix_inner_s = "[MASK]"
# implicit_answer = "America"
# explicit_answer = "Washington" # new_test_68

prompt = "The capital city of China is"
prefix_inner_s = "The capital city of China"
implicit_answer = "America"
explicit_answer = "Washington" # new_test_69


prompt = "The capital city of France is"
prefix_inner_s = "The capital city of France"
implicit_answer = "France"
explicit_answer = "Paris" # new_test_70

implicit_answer = "German"
explicit_answer = "Berlin" # new_test_72

prompt = "The capital city of German is"
prefix_inner_s = "The capital city of Berlin"
implicit_answer = "German"
explicit_answer = "Berlin" # new_test_71

prompt = "The capital city of America is"
prefix_inner_s = "The capital city of America"
implicit_answer = "America"
explicit_answer = "Washington" # new_test_63

prompt = "The capital of the country where victorious originates is"
prefix_inner_s = "The capital of the country where victorious originates"
implicit_answer = "America"
explicit_answer = "Washington" # new_test_73

prompt = "The first African-American president was succeeded by"
prefix_inner_s = "The first African-American president"
implicit_answer = "Barack Obama"
explicit_answer = "Donald Trump" # new_test_2_1


# test_2_avg
# prompt = "The first African-American president was succeeded by"
# prefix_inner_s = "The first African-American president"
# implicit_answer = "Barack Obama"
# explicit_answer = "Donald Trump"

# prompt = "[MASK]" # new_test_1
# prefix_inner_s = "[MASK]"
# prompt = "The president of the United States is" # new_test_2
# prefix_inner_s = "The president of the United States"
# implicit_answer = "United States"
# explicit_answer = "Donald Trump"
# explicit_answer = "Joe Biden" # new_test_3

# test_3
# prompt = "The sports that Michael Jordan played of was invented by"
# prefix_inner_s = "The sports that Michael Jordan played of"
# implicit_answer = "Basketball"
# explicit_answer = "James Naismith"

# test_4
# prompt = "The capital city of the country that has the largest population is"
# # prefix_inner_s = "The capital city of the country"
# prefix_inner_s = "The capital city of the country that has the largest population" # v2
# implicit_answer = "China"
# implicit_answer = "Beijing"
# implicit_answer = "Russia"
# implicit_answer = "India" # test_5
# implicit_answer = "Russia" # test_6
# explicit_answer = "Moscow" # for comparison

# explicit_answer = "New Delhi" # for comparison 2
# explicit_answer = "Beijing"
# explicit_answer = "Shanghai"
# explicit_answer = "Saint Petersburg" 
# explicit_answer = "China" 

# test_7
# prompt = "The director of Jo Jeeta Wohi Sikandar received the"
# prefix_inner_s = "The director of Jo Jeeta Wohi Sikandar"
# implicit_answer = "Mansoor Khan"
# explicit_answer = "Filmfare Award for Best Director"


# test_8
# prompt = "The capital city of Russia is"
# prefix_inner_s = "The capital city of the country"
# prefix_inner_s = "The capital city of Russia" # v2
# prompt = "[MASK] of Russia is"
# prefix_inner_s = "[MASK] of Russia" # mask
# implicit_answer = "Russia"
# implicit_answer = "China"
# implicit_answer = "India" # test_8_compare
# implicit_answer = "Russia" # test_8
# explicit_answer = "Moscow" # for comparison
# explicit_answer = "New Delhi" # for comparison 2
# explicit_answer = "Beijing"
# explicit_answer = "Moscow"


# test_9
# prompt = "Michael Jordan played the sports of"
# prefix_inner_s = "Michael Jordan"
# implicit_answer = "Michael Jordan"
# explicit_answer = "basketball"

# test_10
# prompt = "Michael Jordan played the sports of"
# prefix_inner_s = "Michael Jordan"
# implicit_answer = "Lionel Messi"
# explicit_answer = "soccer"

# test_11
# prompt = "Lionel Messi plays the sports of"
# prefix_inner_s = "Lionel Messi"
# implicit_answer = "Lionel Messi"
# explicit_answer = "soccer"

# test_12
# prompt = "[MASK]"
# prefix_inner_s = "[MASK]"
# implicit_answer = "China"
# explicit_answer = "Beijing"

# # test_13
# prompt = "The country of the publisher of Prague Papers on the History of International Relations is"
# prefix_inner_s = "The country of the publisher of Prague Papers on the History of International Relations"
# implicit_answer = "Charles University"
# explicit_answer = "Czech Republic"

# # test_14
# prompt = "The country of citizenship of the director of Close-Up is"
# prefix_inner_s = "The country of citizenship of the director of Close-Up"
# implicit_answer = "Abbas Kiarostami"
# explicit_answer = "Iran"

# # test_15
# prompt = "The father of Theodore Salisbury Woolsey was educated at"
# prefix_inner_s = "The father of Theodore Salisbury Woolsey"
# implicit_answer = "Theodore Dwight Woolsey"
# explicit_answer = "Yale College"

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

last_tok_id = get_tgt_tok_id(prompt)
last_inner_s_id = get_tgt_tok_id(prefix_inner_s)
check_tok_ids = [last_tok_id, last_inner_s_id]

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
origin_prob, origin_rank, results = calculate_hidden_flow(mt, prompt, check_tok_ids, implicit_toks, explicit_toks)

last_tok_data = results[0]
last_s_data = results[1]

x = list(range(len(last_tok_data["debias_prob"])))
baseline_prob = [origin_prob for i in range(len(x))]
baseline_rank = [origin_rank for i in range(len(x))]
# plot prob variance

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

probs = [baseline_prob, last_tok_data["debias_prob"], last_tok_data["compare_prob"], 
         last_s_data["debias_prob"], last_s_data["compare_prob"]]

ranks = [baseline_rank, last_tok_data["debias_rank"], last_tok_data["compare_rank"], 
         last_s_data["debias_rank"], last_s_data["compare_rank"]]

labels = ["baseline", "last-tok-debias", "last-tok-compare", "inner-s-debias", "inner-s-compare"]

plot(x, probs, labels, "layer", "probability", "Prob Variance: "+implicit_answer+'@'+explicit_answer, prob_path)
plot(x, ranks, labels, "layer", "rank", "Rank Variance: "+implicit_answer+'@'+explicit_answer, rank_path)