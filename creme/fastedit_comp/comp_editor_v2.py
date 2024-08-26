import os
import fire
import json
from typing import Optional

from rome import ROMEHyperParams, apply_rome_to_model_v2
from utils.prints import print_loud
from utils.template import Template
from utils.mtloader import load_model_and_tokenizer
from utils.generate import generate_fast, generate_interactive, get_key_probs, get_prob



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
        probs = [get_prob(model_old, tokenizer, query, template, explicit_toks, wrong_toks) for query in paraphrase_queries]
        print("\n\n".join([paraphrase_queries[i] + " " + pre_update_text[i] + "\n" +str(request["target"])+" prob:" + str(probs[i][0]) +',' + str(wrong_answer)+" prob:" + str(probs[i][1]) for i in range(len(paraphrase_queries))]))

    if len(generalize_queries) > 0:
        print_loud("Generating pre-update text for generalization cases")
        pre_update_text = generate_fast(model_old, tokenizer, generalize_queries, template, max_length=50)
        probs = [get_prob(model_old, tokenizer, generalize_queries[i], template, enc_tok(generalize_answers[i], True) ) for i in range(len(generalize_queries))]
        print("\n\n".join([generalize_queries[i] + " " + pre_update_text[i] + "\n" +str(generalize_answers[i])+" prob:" + str(probs[i][0])  for i in range(len(generalize_queries))]))

    print_loud("Key possibilities for comp, guide and first-hop with the pre-update model")
    pre_prob_comp, pre_prob_wrong_comp, pre_prob_guide,  pre_prob_f_hop\
        = get_key_probs(model_old, tokenizer, cloze_comp, cloze_guide, cloze_f_hop, template, explicit_toks, implicit_toks, wrong_toks)
    
    print('comp:',cloze_comp, request["target"], pre_prob_comp)
    print('comp:',cloze_comp, wrong_answer, pre_prob_wrong_comp)
    print('guide:',cloze_guide, request["target"],pre_prob_guide)
    print('first-hop:',cloze_f_hop,request["subject_guide"],pre_prob_f_hop)

    print_loud(f"Applying rome to model")
    model_new, _ = apply_rome_to_model_v2(
        model_old,
        tokenizer,
        my_requests,
        implicit_toks,
        explicit_toks,
        hparams,
        hparams.edit_mode,
        batch_first,
        return_diff_weights=False
    )

    if len(paraphrase_queries) > 0:
        print_loud("Generating post-update text for paraphrase cases")
        post_update_text = generate_fast(model_new, tokenizer, paraphrase_queries, template, max_length=50)
        probs = [get_prob(model_new, tokenizer, query, template, explicit_toks, wrong_toks) for query in paraphrase_queries]
        print("\n\n".join([paraphrase_queries[i] + " " + post_update_text[i] + "\n" +str(request["target"])+" prob:" + str(probs[i][0]) +',' + str(wrong_answer)+" prob:" + str(probs[i][1]) for i in range(len(paraphrase_queries))]))

    if len(generalize_queries) > 0:
        print_loud("Generating post-update text for generalization cases")
        post_update_text = generate_fast(model_new, tokenizer, generalize_queries, template, max_length=50)
        probs = [get_prob(model_new, tokenizer, generalize_queries[i], template, enc_tok(generalize_answers[i], True) ) for i in range(len(generalize_queries))]
        print("\n\n".join([generalize_queries[i] + " " + post_update_text[i] + "\n" +str(generalize_answers[i])+" prob:" + str(probs[i][0])  for i in range(len(generalize_queries))]))

    print_loud("Key possibilities for comp, guide and first-hop with the pre-update model")
    after_prob_comp, after_prob_wrong_comp, after_prob_guide, after_prob_f_hop\
        = get_key_probs(model_new, tokenizer,cloze_comp, cloze_guide, cloze_f_hop, template, explicit_toks, implicit_toks, wrong_toks)

    print('comp:',cloze_comp, request["target"], after_prob_comp)
    print('comp:',cloze_comp, wrong_answer, after_prob_wrong_comp)
    print('guide:',cloze_guide, request["target"],after_prob_guide)
    print('first-hop:',cloze_f_hop,request["subject_guide"],after_prob_f_hop)

    print_loud("Starting interactively generation interface")
    generate_interactive(model_new, tokenizer, template)

    if output is not None:
        model_new.config.use_cache = True
        model_new.save_pretrained(output, max_shard_size="10GB")
        tokenizer.save_pretrained(output)


if __name__ == "__main__":
    fire.Fire(test_rome)
