import os
import fire
import json
from typing import Optional
import torch
from rome import ROMEHyperParams, apply_rome_to_model
from utils.prints import print_loud
from utils.template import Template
from utils.mtloader import load_model_and_tokenizer
from utils.generate import generate_fast, generate_interactive, get_key_probs, get_prob
import matplotlib.pyplot as plt


def test_rome(
    data: str, model: str, config: str, template: Optional[str] = "default",
    output: Optional[str] = None, checkpointing: Optional[bool] = False
) -> None:
    r"""
    Edits a pre-trained model using model-editing algorithms.

    Args:
        data (`str`):
            The path of the `json` file containing the samples for editing.
        model (`str`):
            The name or path of the pre-trained transformer model to be edited.
        config (`str`):
            The name of the hyper-parameters to use for editing the model.
        template (`str`, *optional*, defaults to `default`):
            The name of the template to use in generation.
        output (`str`, *optional*, defaults to `None`):
            The path to save the edited model.
        checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable gradient checkpointing or not.
    """

    assert os.path.exists(data), "data not found"

    with open(data, "r", encoding="utf-8") as f:
        requests = json.load(f)



    model_old, tokenizer, batch_first = load_model_and_tokenizer(model, checkpointing)
    # input()
    template = Template(name=template)

    print_loud("Retrieving hyperparameters")
    hparams = ROMEHyperParams.from_name(config)
    print(hparams)

    def enc_tok(check_tok, avg=False):
            '''
            avg = True: return all of the encs of the answer string.
            '''
            print('check_tok_enc:', tokenizer.encode(check_tok))
            # check_tok_enc = mt.tokenizer.encode(check_tok)[-1]
            if avg == False:
                check_tok_enc = tokenizer.encode(check_tok)[1] # detach [SOS] token
            else:
                check_tok_enc = tokenizer.encode(check_tok)[1:] # detach [SOS] token
            if isinstance(check_tok_enc, list):
                return check_tok_enc
            elif isinstance(check_tok_enc, int):
                return [check_tok_enc]
            else:
                print(check_tok_enc)
                raise Exception("format is not expected")

    request = requests[0]
    # queries = [query for request in requests for query in request["queries"]]
    paraphrase_queries = [query for query in request["paraphrase queries"]]
    generalize_queries = [query for query in request["generalization queries"]]
    generalize_answers = [answer for answer in request["generalization answers"]]

    cloze_comp = request["prompt_comp"].format(request["subject_comp"])
    cloze_guide = request["prompt_guide"].format(request["subject_guide"])
    cloze_f_hop = request["prompt_f_hop"]
    wrong_answer = request["wrong answer"]
    explicit_toks = enc_tok(request["target"], True)
    implicit_toks = enc_tok(request["subject_guide"], True)
    wrong_toks = enc_tok(wrong_answer, True)



    request_comp = dict()
    request_comp["prompt"] = request["prompt_comp"]
    request_comp["subject"] = request["subject_comp"]
    request_guide = dict()
    request_guide["prompt"] = request["prompt_guide"]
    request_guide["subject"] = request["subject_guide"]
    my_requests = [(request_comp, request_guide)]

    if len(paraphrase_queries) > 0:
        print_loud("Generating pre-update text for paraphrase cases")
        pre_update_text = generate_fast(model_old, tokenizer, paraphrase_queries, template, max_length=50)
        pre_para_probs = [get_prob(model_old, tokenizer, query, template, explicit_toks, wrong_toks) for query in paraphrase_queries]
        print("\n\n".join([paraphrase_queries[i] + " " + pre_update_text[i] + "\n" +str(request["target"])+" prob:" + str(pre_para_probs[i][0]) +',' + str(wrong_answer)+" prob:" + str(pre_para_probs[i][1]) for i in range(len(paraphrase_queries))]))

    if len(generalize_queries) > 0:
        print_loud("Generating pre-update text for generalization cases")
        pre_update_text = generate_fast(model_old, tokenizer, generalize_queries, template, max_length=50)
        pre_gen_probs = [get_prob(model_old, tokenizer, generalize_queries[i], template, enc_tok(generalize_answers[i], True) ) for i in range(len(generalize_queries))]
        print("\n\n".join([generalize_queries[i] + " " + pre_update_text[i] + "\n" +str(generalize_answers[i])+" prob:" + str(pre_gen_probs[i][0])  for i in range(len(generalize_queries))]))

    print_loud("Key possibilities for comp, guide and first-hop with the pre-update model")
    pre_prob_comp, pre_prob_wrong_comp, pre_prob_guide,  pre_prob_f_hop\
        = get_key_probs(model_old, tokenizer, cloze_comp, cloze_guide, cloze_f_hop, template, explicit_toks, implicit_toks, wrong_toks)
    
    print('comp:',cloze_comp, request["target"], pre_prob_comp)
    print('comp:',cloze_comp, wrong_answer, pre_prob_wrong_comp)
    print('guide:',cloze_guide, request["target"],pre_prob_guide)
    print('first-hop:',cloze_f_hop,request["subject_guide"],pre_prob_f_hop)

    print_loud(f"Applying rome to model")
    model_new, _ = apply_rome_to_model(
        model_old,
        tokenizer,
        my_requests,
        hparams,
        hparams.edit_mode,
        batch_first,
        return_diff_weights=False
    )

    if len(paraphrase_queries) > 0:
        print_loud("Generating post-update text for paraphrase cases")
        post_update_text = generate_fast(model_new, tokenizer, paraphrase_queries, template, max_length=50)
        post_para_probs = [get_prob(model_new, tokenizer, query, template, explicit_toks, wrong_toks) for query in paraphrase_queries]
        print("\n\n".join([paraphrase_queries[i] + " " + post_update_text[i] + "\n" +str(request["target"])+" prob(post-pre):" + str((post_para_probs[i][0]-pre_para_probs[i][0])/pre_para_probs[i][0] * 100)+'%' +',' + str(wrong_answer)+" prob(post-pre):" + str((post_para_probs[i][1]-pre_para_probs[i][1])/pre_para_probs[i][1] * 100) +'%' for i in range(len(paraphrase_queries))]))

    if len(generalize_queries) > 0:
        print_loud("Generating post-update text for generalization cases")
        post_update_text = generate_fast(model_new, tokenizer, generalize_queries, template, max_length=50)
        post_gen_probs = [get_prob(model_new, tokenizer, generalize_queries[i], template, enc_tok(generalize_answers[i], True) ) for i in range(len(generalize_queries))]
        print("\n\n".join([generalize_queries[i] + " " + post_update_text[i] + "\n" +str(generalize_answers[i])+" prob(post-pre):" + str((post_gen_probs[i][0]-pre_gen_probs[i][0])/pre_gen_probs[i][0] * 100)+'%'  for i in range(len(generalize_queries))]))
    print_loud("Key possibilities for comp, guide and first-hop with the pre-update model")
    after_prob_comp, after_prob_wrong_comp, after_prob_guide, after_prob_f_hop\
        = get_key_probs(model_new, tokenizer,cloze_comp, cloze_guide, cloze_f_hop, template, explicit_toks, implicit_toks, wrong_toks)

    print('comp:',cloze_comp, request["target"], str((after_prob_comp-pre_prob_comp)/pre_prob_comp * 100)+'%')
    print('comp:',cloze_comp, wrong_answer, str((after_prob_wrong_comp-pre_prob_wrong_comp)/pre_prob_wrong_comp * 100)+'%')
    print('guide:',cloze_guide, request["target"], str((after_prob_guide-pre_prob_guide)/pre_prob_guide * 100) + '%')
    print('first-hop:',cloze_f_hop,request["subject_guide"],str((after_prob_f_hop-pre_prob_f_hop)/pre_prob_f_hop) + '%')



    vocab_size = int(model_new.state_dict()['lm_head.weight'].shape[0])
    average = True

    trace_last_tok = True


    save_dir = "/root/autodl-tmp/zhaoyi/knowledge_locate/logit_lens/post_edit"
    # category = "hasty_answer"
    # category = "incomplete_reason"
    # category = "distortion"
    category = "short_cut"
    # category = "guide"
    # category = "correct"

    save_dir = save_dir + '/'+category

    exp_name = "test_0"

    exp_name = exp_name + '_llama'

    if os.path.exists(save_dir+'/'+exp_name) == False:
        # mkdir
        os.mkdir(save_dir+'/'+exp_name)

    save_dir = save_dir + "/" + exp_name + "/" 
    prob_path = save_dir + "prob.png"
    rank_path = save_dir + "rank.png"

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

    prompt = "The nationality of the performer of the song \"I Feel Love\" is"
    prefix_inner_s = "The nationality of the performer of the song \"I Feel Love\""
    implicit_answer = "Donna Summer"
    explicit_answer = "United States of America"

    # prompt = "The head of state of the country where ORLAN holds citizenship is"
    # prefix_inner_s = "The head of state of the country where ORLAN holds citizenship"
    # implicit_answer = "France"
    # explicit_answer = "Emmanuel Macron"

    # prompt = "The capital city of the country where \"Work from Home\" originated is"
    # prefix_inner_s = "The capital city of the country where \"Work from Home\" originated"
    # implicit_answer = "United States of America"
    # explicit_answer = "Washington, D.C."

    prompt = "The home country of the sport associated with Giorgio Chinaglia is"
    prefix_inner_s = "The home country of the sport associated with Giorgio Chinaglia"
    implicit_answer = "association football"
    explicit_answer = "England"

    if trace_last_tok == True:
        file_path = save_dir + 'last_' + implicit_answer+ '_' +explicit_answer+ '.png'
    else:
        file_path = save_dir + 'inner_' + implicit_answer+ '_' + explicit_answer+ '.png'

    W = model_new.state_dict()['lm_head.weight'] # W * h = v, shape = [32000, hid_dim]
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
        model, tokenizer, prompt, check_tok_ids:int, explicit_toks:list
        ):
        """
        Runs causal tracing over every token/layer combination in the network
        and returns a dictionary numerically summarizing the results.
        """

        inp = make_inputs(tokenizer, [prompt] * 2)
        with torch.no_grad():
            out = model(**inp,output_hidden_states=True)["hidden_states"]
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

    def make_inputs(tokenizer, prompts, device="cuda"):
        token_lists = [tokenizer.encode(p) for p in prompts]
        print('input_encs:', token_lists)
        maxlen = max(len(t) for t in token_lists)
        if "[PAD]" in tokenizer.all_special_tokens:
            pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
        else:
            pad_id = 0
        input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
        print('length of input_ids:', maxlen)
        # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
        attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
        return dict(
            input_ids=torch.tensor(input_ids).to(device),
            #    position_ids=torch.tensor(position_ids).to(device),
            attention_mask=torch.tensor(attention_mask).to(device),
        )

    def get_tgt_tok_id(prefix):
        inner_prefix = make_inputs(tokenizer, [prefix])
        print('prefix:',inner_prefix["input_ids"][0])
        inner_toks_prefix = [tokenizer.decode(inner_prefix["input_ids"][0][i]) for i in range(inner_prefix["input_ids"].shape[1])]
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
            print('check_tok_enc:', tokenizer.encode(check_tok))
            # check_tok_enc = mt.tokenizer.encode(check_tok)[-1]

            start_id = 1 # detach [SOS] token

            if avg == False:
                check_tok_enc = tokenizer.encode(check_tok)[start_id] 
            else:
                check_tok_enc = tokenizer.encode(check_tok)[start_id:] 
            if isinstance(check_tok_enc, list):
                return check_tok_enc
            elif isinstance(check_tok_enc, int):
                return [check_tok_enc]
            else:
                print(check_tok_enc)
                raise Exception("format is not expected")
            
    implicit_toks = enc_tok(implicit_answer, avg=average)
    explicit_toks = enc_tok(explicit_answer, avg=average)

    origin_lens, origin_rank = calculate_hidden_flow(model_new, tokenizer, prompt, check_tok_ids[0], explicit_toks)
    origin_lens_1, origin_rank_1 = calculate_hidden_flow(model_new, tokenizer, prompt, check_tok_ids[0], implicit_toks)


    x = list(range(len(origin_lens)))

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

    plot_twin(origin_lens, origin_lens_1, "explicit reasoning result", "implicit reasoning result", "layer", 'The home country of the sport associated with Giorgio Chinaglia is', file_path)



if __name__ == "__main__":
    fire.Fire(test_rome)
