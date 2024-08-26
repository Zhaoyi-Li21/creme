import os
import fire
import json
from typing import Optional
import sys
from rome import ROMEHyperParams, apply_rome_to_model
from utils.prints import print_loud
from utils.template import Template
from utils.mtloader import load_model_and_tokenizer
from utils.generate import generate_fast, generate_interactive, get_key_probs, get_prob, get_ppl

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
    # print(tokenizer.encode('Donald Trump'))
    # print(tokenizer.encode(' Donald Trump'))
    # print(tokenizer.encode('of USA is Donald Trump'))
    # print(tokenizer.encode('of USA isDonald Trump'))
    # raise Exception('debug')
    # input()
    template = Template(name=template)

    print_loud("Retrieving hyperparameters")
    hparams = ROMEHyperParams.from_name(config)
    print(hparams)

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




    request_comp = dict()
    request_comp["prompt"] = request["prompt_comp"]
    request_comp["subject"] = request["subject_comp"]
    request_guide = dict()
    request_guide["prompt"] = request["prompt_guide"]
    request_guide["subject"] = request["subject_guide"]
    my_requests = [(request_comp, request_guide)]

    
    output_datum["correctness"] = dict()
    output_datum["correctness"]["query"] = original_query
    output_datum["correctness"]["guide"] = request["prompt_guide"].format(request["subject_guide"])
    output_datum["correctness"]["target"] = request["target"]
    output_datum["correctness"]["guide_ppl"] = 0.
    output_datum["correctness"]["pre_predict"] = ""
    output_datum["correctness"]["pre_ppl"] = 0.
    output_datum["correctness"]["post_predict"] = ""
    output_datum["correctness"]["post_ppl"] = 0.

    output_datum["paraphrase"] = list()
    for query in paraphrase_queries:
        datum = dict()
        if "Q: " not in query:
            datum["query"] = query
        else:
            query_ = query.split('Q: ')[-1]
            query_ = query_.split('\nA:')[0]
            datum["query"] = query_

        datum["target"] = request["target"]
        datum["pre_predict"] = ""
        datum["pre_ppl"] = 0.
        datum["post_predict"] = ""
        datum["post_ppl"] = 0.
        output_datum["paraphrase"].append(datum)

    output_datum["generalization"] = list()
    for i in range(len(generalize_queries)):
        datum = dict()
        if "Q: " not in query:
            datum["query"] = query
        else:
            query_ = query.split('Q: ')[-1]
            query_ = query_.split('\nA:')[0]
            datum["query"] = query_

        datum["query"] = generalize_queries[i]
        datum["target"] = generalize_answers[i]
        datum["pre_predict"] = ""
        datum["pre_ppl"] = 0.
        datum["post_predict"] = ""
        datum["post_ppl"] = 0.
        output_datum["generalization"].append(datum)



    if len(paraphrase_queries) > 0:
        pre_update_text = generate_fast(model_old, tokenizer, paraphrase_queries, template, max_length=100)
        # pre_para_probs = [get_prob(model_old, tokenizer, query, template, explicit_toks, wrong_toks) for query in paraphrase_queries]
        for idx in range(len(paraphrase_queries)):
            query = paraphrase_queries[idx]
            if 'Q: ' in query:
                
                pre_text = pre_update_text[idx].split(query)[0]
                if '\nQ' in pre_text:
                    pre_text = pre_text.split('\nQ')[0]
                else:
                    pre_text = 'Wrong Format'
                output_datum["paraphrase"][idx]["pre_predict"] += pre_text

            output_datum["paraphrase"][idx]["pre_ppl"] = get_ppl(model_old, tokenizer, query, template, request["target"], None)



    if len(generalize_queries) > 0:
        pre_update_text = generate_fast(model_old, tokenizer, generalize_queries, template, max_length=100)
        # pre_para_probs = [get_prob(model_old, tokenizer, query, template, explicit_toks, wrong_toks) for query in paraphrase_queries]
        for idx in range(len(generalize_queries)):
            query = generalize_queries[idx]
            if 'Q: ' in query:
                pre_text = pre_update_text[idx].split(query)[0]
                if '\nQ' in pre_text:
                    pre_text = pre_text.split('\nQ')[0]
                else:
                    pre_text = 'Wrong Format'
                output_datum["generalization"][idx]["pre_predict"] += pre_text

            output_datum["generalization"][idx]["pre_ppl"] = get_ppl(model_old, tokenizer, query, template, generalize_answers[idx], None)

    output_datum["correctness"]["pre_ppl"] = get_ppl(model_old, tokenizer, original_query, template, request["target"], None)
    output_datum["correctness"]["guide_ppl"] = get_ppl(model_old, tokenizer, output_datum["correctness"]["guide"], template, request["target"], None)

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

    output_datum["correctness"]["post_ppl"] = get_ppl(model_new, tokenizer, original_query, template, request["target"], None)


    if len(paraphrase_queries) > 0:
        post_update_text = generate_fast(model_new, tokenizer, paraphrase_queries, template, max_length=100)
        # pre_para_probs = [get_prob(model_old, tokenizer, query, template, explicit_toks, wrong_toks) for query in paraphrase_queries]
        for idx in range(len(paraphrase_queries)):
            query = paraphrase_queries[idx]
            if 'Q: ' in query:
                post_text = post_update_text[idx].split(query)[0]
                if '\nQ' in post_text:
                    post_text = post_text.split('\nQ')[0]
                else:
                    post_text = 'Wrong Format'
                output_datum["paraphrase"][idx]["post_predict"] += post_text

            output_datum["paraphrase"][idx]["post_ppl"] = get_ppl(model_new, tokenizer, query, template, request["target"], None)



    if len(generalize_queries) > 0:
        post_update_text = generate_fast(model_new, tokenizer, generalize_queries, template, max_length=100)
        # pre_para_probs = [get_prob(model_old, tokenizer, query, template, explicit_toks, wrong_toks) for query in paraphrase_queries]
        for idx in range(len(generalize_queries)):
            query = generalize_queries[idx]
            if 'Q: ' in query:
                post_text = post_update_text[idx].split(query)[0]
                if '\nQ' in post_text:
                    post_text = post_text.split('\nQ')[0]
                else:
                    post_text = 'Wrong Format'
                output_datum["generalization"][idx]["post_predict"] += post_text

            output_datum["generalization"][idx]["post_ppl"] = get_ppl(model_new, tokenizer, query, template, generalize_answers[idx], None)




    # for type in ["correctness", "paraphrase", "generalize","irrelevant"]:
    #     output_datum[type]["delta"] = dict()
    #     for key in ["target", "wrong"]:
    #         if key in output_datum[type]["post_probs"]:
    #             # print(type,key,output_datum[type]["post_probs"][key],output_datum[type]["pre_probs"][key])
    #             output_datum[type]["delta"][key] = list()
    #             for j in range(len(output_datum[type]["pre_probs"][key])):
    #                 if output_datum[type]["pre_probs"][key][j] == 0.:
    #                     output_datum[type]["delta"][key].append(0.)
    #                 else:
    #                     output_datum[type]["delta"][key].append((output_datum[type]["post_probs"][key][j]-output_datum[type]["pre_probs"][key][j])/output_datum[type]["pre_probs"][key][j])
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
    # total_num = 1
    fr.close()
    write_path = read_path.replace('make_dataset', "results/gen_errors_2")
    fw = open(write_path, "w")
    output_data = list()
    for _ in range(total_num):
        fire.Fire(test_rome)
    json.dump(output_data, fw, indent=4)
