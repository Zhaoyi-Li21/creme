import os
import fire
import json
from typing import Optional
import sys
from rome import ROMEHyperParams, apply_rome_to_model
from utils.prints import print_loud
from utils.template import Template
from utils.mtloader import load_model_and_tokenizer
from utils.generate import generate_fast, generate_interactive, get_key_probs, get_prob

cnt = 0


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

    output_datum = dict()

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

    global cnt
    # print(len(requests))
    if cnt >= len(requests): return
    request = requests[cnt] # get cnt-th datum
    cnt += 1 # update cnt

    # queries = [query for request in requests for query in request["queries"]]
    original_query = request["original query"]
    paraphrase_queries = [query for query in request["paraphrase queries"]]
    generalize_queries = [query for query in request["generalization queries"]]
    generalize_answers = [answer for answer in request["generalization answers"]]

    irrelevant_queries = [query for query in request["irrelevant queries"]]
    irrelevant_answers = [answer for answer in request["irrelevant answers"]]

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

    output_datum["correctness"] = dict()
    output_datum["correctness"]["pre_probs"] = dict()


    output_datum["paraphrase"] = dict()
    output_datum["paraphrase"]["pre_probs"] = dict()
    for key in ["target", "wrong"]:
        output_datum["paraphrase"]["pre_probs"][key] = list()

    if len(paraphrase_queries) > 0:
        # print_loud("Generating pre-update text for paraphrase cases")
        # pre_update_text = generate_fast(model_old, tokenizer, paraphrase_queries, template, max_length=100)
        pre_para_probs = [get_prob(model_old, tokenizer, query, template, explicit_toks, wrong_toks) for query in paraphrase_queries]
        # print("\n\n".join([paraphrase_queries[i] + " " + pre_update_text[i] + "\n" +str(request["target"])+" prob:" + str(pre_para_probs[i][0]) +',' + str(wrong_answer)+" prob:" + str(pre_para_probs[i][1]) for i in range(len(paraphrase_queries))]))
        output_datum["paraphrase"]["pre_probs"]["target"] = [pre_para_probs[i][0] for i in range(len(paraphrase_queries))]
        output_datum["paraphrase"]["pre_probs"]["wrong"] = [pre_para_probs[i][1] for i in range(len(paraphrase_queries))]

    output_datum["generalize"] = dict()
    output_datum["generalize"]["pre_probs"] = dict()
    for key in ["target"]:
        output_datum["generalize"]["pre_probs"][key] = list()

    if len(generalize_queries) > 0:
        # print_loud("Generating pre-update text for generalization cases")
        # pre_update_text = generate_fast(model_old, tokenizer, generalize_queries, template, max_length=100)
        pre_gen_probs = [get_prob(model_old, tokenizer, generalize_queries[i], template, enc_tok(generalize_answers[i], True) ) for i in range(len(generalize_queries))]
        # print("\n\n".join([generalize_queries[i] + " " + pre_update_text[i] + "\n" +str(generalize_answers[i])+" prob:" + str(pre_gen_probs[i][0])  for i in range(len(generalize_queries))]))
        output_datum["generalize"]["pre_probs"]["target"] = [pre_gen_probs[i][0] for i in range(len(generalize_queries))]

    output_datum["irrelevant"] = dict()
    output_datum["irrelevant"]["pre_probs"] = dict()
    for key in ["target", "wrong"]:
        output_datum["irrelevant"]["pre_probs"][key] = list()

    if len(irrelevant_queries) > 0:
        # print_loud("Generating pre-update text for generalization cases")
        # pre_update_text = generate_fast(model_old, tokenizer, irrelevant_queries, template, max_length=100)
        pre_irre_probs = [get_prob(model_old, tokenizer, irrelevant_queries[i], template, enc_tok(irrelevant_answers[i], True), explicit_toks ) for i in range(len(irrelevant_queries))]
        # print("\n\n".join([irrelevant_queries[i] + " " + pre_update_text[i] + "\n" +str(irrelevant_answers[i])+" prob:" + str(pre_irre_probs[i][0])  for i in range(len(irrelevant_queries))]))
        output_datum["irrelevant"]["pre_probs"]["target"] = [pre_irre_probs[i][0] for i in range(len(irrelevant_queries))]
        output_datum["irrelevant"]["pre_probs"]["wrong"] = [pre_irre_probs[i][1] for i in range(len(irrelevant_queries))]

    # print_loud("Key possibilities for comp, guide and first-hop with the pre-update model")
    pre_prob_comp, pre_prob_wrong_comp, pre_prob_guide,  pre_prob_f_hop\
        = get_key_probs(model_old, tokenizer, cloze_comp, cloze_guide, cloze_f_hop, template, explicit_toks, implicit_toks, wrong_toks)
    
    output_datum["correctness"]["pre_probs"]["target"]=[pre_prob_comp]
    output_datum["correctness"]["pre_probs"]["wrong"]=[pre_prob_wrong_comp]
    output_datum["first-hop"] = dict()
    output_datum["second-hop"] = dict()
    output_datum["second-hop"]["pre_prob"] = pre_prob_guide
    output_datum["first-hop"]["pre_prob"] = pre_prob_f_hop

    # print('comp:',cloze_comp, request["target"], pre_prob_comp)
    # print('comp:',cloze_comp, wrong_answer, pre_prob_wrong_comp)
    # print('guide:',cloze_guide, request["target"],pre_prob_guide)
    # print('first-hop:',cloze_f_hop,request["subject_guide"],pre_prob_f_hop)

    # print_loud(f"Applying rome to model")
    model_new, _ = apply_rome_to_model(
        model_old,
        tokenizer,
        my_requests,
        hparams,
        hparams.edit_mode,
        batch_first,
        return_diff_weights=False
    )


    output_datum["correctness"]["post_probs"] = dict()

    output_datum["paraphrase"]["post_probs"] = dict()
    for key in ["target", "wrong"]:
        output_datum["paraphrase"]["post_probs"][key] = list()

    if len(paraphrase_queries) > 0:
        # print_loud("Generating post-update text for paraphrase cases")
        # post_update_text = generate_fast(model_new, tokenizer, paraphrase_queries, template, max_length=50)
        post_para_probs = [get_prob(model_new, tokenizer, query, template, explicit_toks, wrong_toks) for query in paraphrase_queries]
        # print("\n\n".join([paraphrase_queries[i] + " " + post_update_text[i] + "\n" +str(request["target"])+" prob(post-pre):" + str((post_para_probs[i][0]-pre_para_probs[i][0])/pre_para_probs[i][0] * 100)+'%' +',' + str(wrong_answer)+" prob(post-pre):" + str((post_para_probs[i][1]-pre_para_probs[i][1])/pre_para_probs[i][1] * 100) +'%' for i in range(len(paraphrase_queries))]))
        output_datum["paraphrase"]["post_probs"]["target"] = [post_para_probs[i][0] for i in range(len(paraphrase_queries))]
        output_datum["paraphrase"]["post_probs"]["wrong"] = [post_para_probs[i][1] for i in range(len(paraphrase_queries))]

    output_datum["generalize"]["post_probs"] = dict()
    for key in ["target"]:
        output_datum["generalize"]["post_probs"][key] = list()

    if len(generalize_queries) > 0:
        # print_loud("Generating post-update text for generalization cases")
        # post_update_text = generate_fast(model_new, tokenizer, generalize_queries, template, max_length=50)
        post_gen_probs = [get_prob(model_new, tokenizer, generalize_queries[i], template, enc_tok(generalize_answers[i], True) ) for i in range(len(generalize_queries))]
        # print("\n\n".join([generalize_queries[i] + " " + post_update_text[i] + "\n" +str(generalize_answers[i])+" prob(post-pre):" + str((post_gen_probs[i][0]-pre_gen_probs[i][0])/pre_gen_probs[i][0] * 100)+'%'  for i in range(len(generalize_queries))]))
        output_datum["generalize"]["post_probs"]["target"] = [post_gen_probs[i][0] for i in range(len(generalize_queries))]


    output_datum["irrelevant"]["post_probs"] = dict()
    for key in ["target", "wrong"]:
        output_datum["irrelevant"]["post_probs"][key] = list()

    if len(irrelevant_queries) > 0:
        # print_loud("Generating pre-update text for generalization cases")
        # pre_update_text = generate_fast(model_old, tokenizer, irrelevant_queries, template, max_length=100)
        post_irre_probs = [get_prob(model_new, tokenizer, irrelevant_queries[i], template, enc_tok(irrelevant_answers[i], True), explicit_toks ) for i in range(len(irrelevant_queries))]
        # print("\n\n".join([irrelevant_queries[i] + " " + pre_update_text[i] + "\n" +str(irrelevant_answers[i])+" prob:" + str(pre_irre_probs[i][0])  for i in range(len(irrelevant_queries))]))
        output_datum["irrelevant"]["post_probs"]["target"] = [post_irre_probs[i][0] for i in range(len(irrelevant_queries))]
        output_datum["irrelevant"]["post_probs"]["wrong"] = [post_irre_probs[i][1] for i in range(len(irrelevant_queries))]

    # print_loud("Key possibilities for comp, guide and first-hop with the pre-update model")
    after_prob_comp, after_prob_wrong_comp, after_prob_guide, after_prob_f_hop\
        = get_key_probs(model_new, tokenizer,cloze_comp, cloze_guide, cloze_f_hop, template, explicit_toks, implicit_toks, wrong_toks)

    # print('comp:',cloze_comp, request["target"], str((after_prob_comp-pre_prob_comp)/pre_prob_comp * 100)+'%')
    # print('comp:',cloze_comp, wrong_answer, str((after_prob_wrong_comp-pre_prob_wrong_comp)/pre_prob_wrong_comp * 100)+'%')
    # print('guide:',cloze_guide, request["target"], str((after_prob_guide-pre_prob_guide)/pre_prob_guide * 100) + '%')
    # print('first-hop:',cloze_f_hop,request["subject_guide"],str((after_prob_f_hop-pre_prob_f_hop)/pre_prob_f_hop) + '%')

    output_datum["correctness"]["post_probs"]["target"]=[after_prob_comp]
    output_datum["correctness"]["post_probs"]["wrong"]=[after_prob_wrong_comp]
    
    output_datum["second-hop"]["post_prob"] = after_prob_guide
    output_datum["first-hop"]["post_prob"] = after_prob_f_hop

    for type in ["correctness", "paraphrase", "generalize","irrelevant"]:
        output_datum[type]["delta"] = dict()
        for key in ["target", "wrong"]:
            if key in output_datum[type]["post_probs"]:
                # print(type,key,output_datum[type]["post_probs"][key],output_datum[type]["pre_probs"][key])
                output_datum[type]["delta"][key] = list()
                for j in range(len(output_datum[type]["pre_probs"][key])):
                    if output_datum[type]["pre_probs"][key][j] == 0.:
                        output_datum[type]["delta"][key].append(0.)
                    else:
                        output_datum[type]["delta"][key].append((output_datum[type]["post_probs"][key][j]-output_datum[type]["pre_probs"][key][j])/output_datum[type]["pre_probs"][key][j])
                # output_datum[type]["delta"][key] = [(output_datum[type]["post_probs"][key][j]-output_datum[type]["pre_probs"][key][j])/output_datum[type]["pre_probs"][key][j] for j in range(len(output_datum[type]["pre_probs"][key]))]

    global output_data
    output_data.append(output_datum)

    f.close()

if __name__ == "__main__":
    # print(sys.argv[2]) # /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/llama2.suf.json
    read_path = sys.argv[2]
    fr = open(read_path, "r")
    data = json.load(fr)
    total_num = len(data)
    # print(total_num)
    fr.close()
    write_path = read_path.replace('make_dataset', "results/rebuttal")
    fw = open(write_path, "w")
    output_data = list()
    for _ in range(total_num):
        fire.Fire(test_rome)
    json.dump(output_data, fw, indent=4)
