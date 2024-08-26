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
from utils import model_name2path, get_lm_type, get_distance
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

save_dir = "/root/autodl-tmp/zhaoyi/knowledge_locate/distance_bet_reps"


exp_name = "test_1"

exp_name = exp_name + '_' +model_name


if os.path.exists(save_dir+'/'+exp_name) == False:
    # mkdir
    os.mkdir(save_dir+'/'+exp_name)

save_dir = save_dir + "/" + exp_name + "/" 


comp_prompt = "The nationality of the performer of the song \"I Feel Love\" is"
comp_prefix_inner_s = "The nationality of the performer of the song \"I Feel Love\""
single_prompt = "The nationality of Donna Summer is"
single_prefix_inner_s = "The nationality of Donna Summer"

comp_subj = "the performer of the song \"I Feel Love\""
single_subj = "Donna Summer" # test_1


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

def get_hidden_states(
    mt, prompt
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
        hidden_states = list()
        for layer_idx in range(len(out)):
            hidden_state = out[layer_idx][0, -1] # shape = [hidden_dim]
            hidden_states.append(hidden_state)
    return hidden_states
            
hs_dict = dict()
hs_dict['comp_last_hs'] = get_hidden_states(mt, comp_prompt)
tot_number = len(hs_dict['comp_last_hs'])
hs_dict['comp_last_inner_s_hs'] = get_hidden_states(mt, comp_prefix_inner_s)
hs_dict['single_last_hs'] = get_hidden_states(mt, single_prompt)
hs_dict['single_last_inner_s_hs'] = get_hidden_states(mt, single_prefix_inner_s)
hs_dict['comp_subj_hs'] = get_hidden_states(mt, comp_subj)
hs_dict['single_subj_hs'] = get_hidden_states(mt, single_subj)



def get_dist_matrix(comp_hs, single_hs, distance='L2'):
    dist_matrix = list()
    for i in range(len(comp_hs)):
        dist_by_layer = list()
        for j in range(len(single_hs)):
            dist_by_layer.append(get_distance(comp_hs[i], single_hs[j], distance).cpu())
        dist_matrix.append(dist_by_layer)
    return np.array(dist_matrix)

for exp_type in ['last_hs', 'last_inner_s_hs', 'subj_hs']:
    for dist_type in ['L1', 'L2', 'Cosine']:
        dist_matrix = get_dist_matrix(hs_dict['comp_'+exp_type], hs_dict['single_'+exp_type], dist_type)
        plt.figure()
        plot_data = pd.DataFrame(dist_matrix,  [i for i in range(tot_number)], [i for i in range(tot_number)])
        plot = sns.heatmap(plot_data)
        plt.xlabel('single_'+exp_type)
        plt.ylabel('comp_'+exp_type)
        plt.title('distance matrix:'+exp_type+'@'+dist_type)
        plt.savefig(save_dir+exp_type+'@'+dist_type+'.png')
        plt.clf()



