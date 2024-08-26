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
offset = False

trace_last_tok = True
mode_str = "_" + mode[0]

save_dir = "/root/autodl-tmp/zhaoyi/knowledge_locate/logit_lens/a_paper_cases"
# category = "hasty_answer"
# category = "incomplete_thinking"
# category = "distortion"
category = "short_cut"
# category = "guide"
# category = "correct"

save_dir = save_dir + '/'+category

exp_name = "test_6"

exp_name = exp_name + '_' +model_name

if os.path.exists(save_dir+'/'+exp_name) == False:
    # mkdir
    os.mkdir(save_dir+'/'+exp_name)

save_dir = save_dir + "/" + exp_name + "/" 
prob_path = save_dir + "prob.png"
rank_path = save_dir + "rank.png"


prompt = "The capital city of France is"
prefix_inner_s = "The capital city of France"
implicit_answer = "France" 
explicit_answer = "France"# new_test_1
explicit_answer = "Ottawa"
# explicit_answer = "German"
# explicit_answer = "Canada"

prompt = "The capital city of German is"
prefix_inner_s = "The capital city of German"
explicit_answer = "France"

prompt = "The capital city of China is"
prefix_inner_s = "The capital city of China"
explicit_answer = "China"
explicit_answer = "France"
explicit_answer = "Beijing"
explicit_answer = "Paris"

prompt = "China is"
prefix_inner_s = "China"
explicit_answer = "Beijing"
explicit_answer = "China"

prompt = "France is"
prefix_inner_s = "France"
explicit_answer = "France"
explicit_answer = "Paris"
explicit_answer = "China"
explicit_answer = "Beijing"

prompt = "The nationality of the author of the \"Harry Potter\" series is" # pass all
prefix_inner_s = "The nationality of the author of the \"Harry Potter\" series"
explicit_answer = "J. K. Rowling"
explicit_answer = "United Kingdom"

prompt = "The location of the headquarters of the institution where Bohumil Hrabal studied is" # pass all
prefix_inner_s = "The location of the headquarters of the institution where Bohumil Hrabal studied"
explicit_answer = "Charles University"
explicit_answer = "Prague"

prompt = "The capital city of the country from which HKT48 originated is" # pass all
prefix_inner_s = "The capital city of the country from which HKT48 originated"
implicit_answer = "Japan"
explicit_answer = "Tokyo"

prompt = "The capital city of the country that Michael Feinstein is a citizen of is" # pass single fail comp
prefix_inner_s = "The capital city of the country that Michael Feinstein is a citizen of"
implicit_answer = "United States of America"
explicit_answer = "Washington, D.C."

prompt = "The capital city of United States of America is"
prefix_inner_s = "The capital city of United States of America"
implicit_answer = "United States of America"
explicit_answer = "Washington, D.C."

prompt = "The capital city of Japan is"
prefix_inner_s = "The capital city of Japan"
implicit_answer = "Japan"
explicit_answer = "Tokyo"


prompt = "The capital city of the country where \"American Ninja Warrior\" originated is" # pass single fail comp
prefix_inner_s = "The capital city of the country where \"American Ninja Warrior\" originated"
implicit_answer = "United States of America"
explicit_answer = "Washington, D.C."
implicit_answer = "Japan"
explicit_answer = "Tokyo"
implicit_answer = "France"
explicit_answer = "Paris"

prompt = "The name of the current head of state in United States of America is" 
prefix_inner_s = "The name of the current head of state in United States of America"
implicit_answer = "Uinted States of America"
explicit_answer = "Donald Trump"

prompt = "The name of the current head of state of the country where Micky Ward holds a citizenship is" # pass single fail comp
prefix_inner_s = "The name of the current head of state of the country where Micky Ward holds a citizenship" 
implicit_answer = "Uinted States of America"
explicit_answer = "Donald Trump"

# prompt = "The head of government of the country which Josh Barnett is a citizen of is" # pass all
# prefix_inner_s = "The head of government of the country which Josh Barnett is a citizen of"
# implicit_answer = "United States of America"
# explicit_answer = "Donald Trump"

# prompt = "The name of the current head of the country which Josh Barnett is a citizen of is" 
# prefix_inner_s = "The name of the current head of the country which Josh Barnett is a citizen of"
# implicit_answer = "United States of America"
# explicit_answer = "Donald Trump"

# prompt = "The head of government of the country of origin of Chavez is"  # pass single fail comp
# prefix_inner_s = "The head of government of the country of origin of Chavez"
# implicit_answer = "United States of America"
# explicit_answer = "Donald Trump"

# implicit_answer = "Venezuela"
# explicit_answer = "Hugo Chavez"

# prompt = "The country of origin of Chavez is"
# prefix_inner_s = "The country of origin of Chavez"
# implicit_answer = "Venezuela"
# explicit_answer = "United States of America"

# prompt = "The country that the creator of C. Auguste Dupin belongs to is"
# prefix_inner_s = "The country that the creator of C. Auguste Dupin belongs to"
# implicit_answer = "Edgar Allan Poe"
# explicit_answer = "United States of America"
# explicit_answer = "France"

# prompt = "The country that Edgar Allan Poe belongs to"
# implicit_answer = "Edgar Allan Poe"
# explicit_answer = "United States of America"
# explicit_answer = "France"



# prompt = "The country of origin of basketball is"
# prefix_inner_s = "The country of origin of basketball"
# implicit_answer = "basketball"
# explicit_answer = "United States of America"
# explicit_answer = "Spain"

# prompt = "The country of origin of the sport associated with EuroBasket 1999 is"
# prefix_inner_s = "The country of origin of the sport associated with EuroBasket 1999"
# implicit_answer = "basketball"
# explicit_answer = "United States of America"
# explicit_answer = "Spain"

# prompt = "The continent that the country of citizenship of Maurice Strong belongs to is"
# prefix_inner_s = "The continent that the country of citizenship of Maurice Strong belongs to"
# implicit_answer = "Canada"
# explicit_answer = "Asia"
# explicit_answer = "North America"
# explicit_answer = "India"

# prompt = "The continent that Canada belongs to is"
# prefix_inner_s = "The continent that Canada belongs to"
# implicit_answer = "Canada"
# explicit_answer = "Asia"
# explicit_answer = "North America"

# prompt = "The nationality of the performer of the song \"I Feel Love\" is"
# prefix_inner_s = "The nationality of the performer of the song \"I Feel Love\""
# implicit_answer = "Donna Summer"
# explicit_answer = "United States of America"


# prompt = "The nationality of Donna Summer is"
# prefix_inner_s = "The nationality of Donna Summer"
# implicit_answer = "Donna Summer"
# explicit_answer = "United States of America"

# prompt = "The performer of the song \"I Feel Love\" is"
# prefix_inner_s = "The performer of the song \"I Feel Love\""
# implicit_answer = "\"I Feel Love\""
# explicit_answer = "Donna Summer"


prompt = "The head of state of the country where John Cho is a citizen is"
prefix_inner_s = "The head of state of the country where John Cho is a citizen"
implicit_answer = "Uinted States of America"
explicit_answer = "Donald Trump"

prompt = "The home country of the sport associated with Giorgio Chinaglia is"
prefix_inner_s = "The home country of the sport associated with Giorgio Chinaglia"
implicit_answer = "association football"
explicit_answer = "England"
implicit_answer = "Giorgio Chinaglia"
explicit_answer = "Italy"

prompt = "The sport associated with Giorgio Chinaglia is"
prefix_inner_s = "The sport associated with Giorgio Chinaglia"
implicit_answer = "Giorgio Chinaglia"
implicit_answer = "sport"
explicit_answer = "association football"

prompt = "The home country of association football is"
prefix_inner_s = "The home country of association football"
implicit_answer = "association football"
explicit_answer = "England"

prompt = "The head of state of the country where ORLAN holds citizenship is"
prefix_inner_s = "The head of state of the country where ORLAN holds citizenship"
implicit_answer = "France"
explicit_answer = "Emmanuel Macron"

prompt = "The head of state of France is"
prefix_inner_s = "The head of state of France"
implicit_answer = "France"
explicit_answer = "Emmanuel Macron"

prompt = "The head of state of the country where Micky Ward holds a citizenship is" # pass single fail comp
prefix_inner_s = "The name of the current head of state of the country where Micky Ward holds a citizenship" 
implicit_answer = "Uinted States of America"

explicit_answer = "Donald Trump"


prompt = "China"
prefix_inner_s = "China"
implicit_answer = "France"
explicit_answer = "China"

prompt = "The name of the current head of the country which Josh Barnett is a citizen of is" 
prefix_inner_s = "The name of the current head of the country which Josh Barnett is a citizen of"
implicit_answer = "United States of America"
explicit_answer = "Donald Trump"
explicit_answer = "Joe Biden"

prompt = "The capital city of the country where \"Work from Home\" originated is"
prefix_inner_s = "The capital city of the country where \"Work from Home\" originated"
implicit_answer = "United States of America"
explicit_answer = "Washington, D.C."

prompt = "The capital city of the United States of America is"
prefix_inner_s = "The capital city of the United States of America"
implicit_answer = "United States of America"
explicit_answer = "Washington, D.C."

prompt = "The head of state of the country where ORLAN holds citizenship is"
prefix_inner_s = "The head of state of the country where ORLAN holds citizenship"
implicit_answer = "France"
explicit_answer = "Emmanuel Macron"

prompt = "The head of state of France is"
prefix_inner_s = "The head of state of France"
implicit_answer = "France"
explicit_answer = "Emmanuel Macron"

prompt = "The nationality of the performer of the song \"I Feel Love\" is"
prefix_inner_s = "The nationality of the performer of the song \"I Feel Love\""
implicit_answer = "Donna Summer"
explicit_answer = "United States of America"

prompt = "The nationality of Donna Summer is"
prefix_inner_s = "The nationality of Donna Summer"
implicit_answer = "Donna Summer"
explicit_answer = "United States of America"

prompt = "The home country of the sport associated with Giorgio Chinaglia is"
prefix_inner_s = "The home country of the sport associated with Giorgio Chinaglia"
# implicit_answer = "association football"
# explicit_answer = "England"
implicit_answer = "Giorgio Chinaglia"
explicit_answer = "Italy"

# prompt = "The home country of association football is"
# prefix_inner_s = "The home country of association football"
# implicit_answer = "association football"
# explicit_answer = "England"

# prompt = "The home country of Giorgio Chinaglia is"
# prefix_inner_s = "The home country of Giorgio Chinaglia"
# implicit_answer = "Giorgio Chinaglia"
# explicit_answer = "Italy"

prompt = "The head of state of the country where Micky Ward holds a citizenship is" # pass single fail comp
prefix_inner_s = "The name of the current head of state of the country where Micky Ward holds a citizenship" 
implicit_answer = "Uinted States of America"

explicit_answer = "Donald Trump"

prompt = "The name of the current head of state of the Uinted States of America is" # pass single fail comp
prefix_inner_s = "The name of the current head of state of the Uinted States of America" 
implicit_answer = "Uinted States of America"

explicit_answer = "Donald Trump"

prompt = "The name of the current head of state of the country where Micky Ward holds a citizenship is" # pass single fail comp
prefix_inner_s = "The name of the current head of state of the country where Micky Ward holds a citizenship" 
implicit_answer = "Uinted States of America"

explicit_answer = "Donald Trump"


# prompt = "The name of the current head of state of the country where John Cho is a citizen is" # pass single fail comp
# prefix_inner_s = "The name of the current head of state of the country where John Cho is a citizen" 
# implicit_answer = "Uinted States of America"

# explicit_answer = "Donald Trump"


prompt = "The country that the creator of C. Auguste Dupin belongs to is"
prefix_inner_s = "The country that the creator of C. Auguste Dupin belongs to"
implicit_answer = "Edgar Allan Poe"
explicit_answer = "United States of America"

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
# plot_twin(origin_lens, origin_rank, "probability w. lens", "rank w. lens", "layer", "Logit Lens: "+explicit_answer+'@'+prompt, file_path)
# plot_twin_im_ex(origin_lens, origin_lens_1, origin_rank, origin_rank_1, "probability w. lens", "rank w. lens", "layer", implicit_answer+'@'+explicit_answer+'@'+prompt, file_path)