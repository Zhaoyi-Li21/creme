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


    for type in ["irrelevant"]:
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
    # total_num = 1
    # print(total_num)
    fr.close()
    write_path = read_path.replace('make_dataset', "results/v2")
    fw = open(write_path, "w")
    output_data = list()
    for _ in range(total_num):
        fire.Fire(test_rome)
    json.dump(output_data, fw, indent=4)
