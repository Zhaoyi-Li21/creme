import torch
from typing import List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

from .template import Template


def generate_interactive(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    template: Template,
    top_k: Optional[int] = 50,
    max_length: Optional[int] = 200
):
    r"""
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    print("Enter `exit` to exit the interface.")

    while True:
        query = input("Input: ").strip()

        if query == "exit":
            break

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("Output: ", end="", flush=True)
        generate_fast(model, tokenizer, [query], template, top_k=top_k, max_length=max_length, streamer=streamer)[0]
        print()


def generate_fast(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    queries: List[str],
    template: Template,
    n_gen_per_prompt: Optional[int] = 1,
    top_k: Optional[int] = 50,
    max_length: Optional[int] = 200,
    streamer: Optional[TextStreamer] = None
):
    r"""
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [template.get_prompt(query) for query in queries for _ in range(n_gen_per_prompt)]
    inp_tok = tokenizer(inp, padding=True, return_token_type_ids=False, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inp_tok,
            temperature=0.1,
            top_k=top_k,
            max_length=max_length,
            do_sample=True,
            streamer=streamer
        )

    responses = tokenizer.batch_decode(
        generated_ids[:, inp_tok["input_ids"].size(1):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return responses

def get_key_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    query_comp: str,
    template: Template,
    explicit_toks: List[int],
    wrong_toks: List[int],
):
    inp = [template.get_prompt(query) for query in [query_comp]]
    inp_tok = tokenizer(inp, padding=True, return_token_type_ids=False, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**inp_tok)["logits"] # out.shape = (1, seq_len, vocab_size)
        logits_comp = torch.softmax(out[0, -1], dim=0).tolist() # len = vocab_size
        # max_v = 0
        # max_id = 0
        # for i in range(len(logits_comp)):
        #     if logits_comp[i] > max_v:
        #         max_v = logits_comp[i]
        #         max_id = i
        # print(explicit_toks, max(logits_comp), logits_comp[explicit_toks[0]], max_id, tokenizer.decode(max_id), tokenizer.decode(explicit_toks[0]))
        origin_prob_comp = 0
        wrong_prob_comp = 0

        for tok in explicit_toks:
            origin_prob_comp += logits_comp[tok]

        for tok in wrong_toks:
            wrong_prob_comp += logits_comp[tok]

        origin_prob_comp /= len(explicit_toks)
        wrong_prob_comp /= len(wrong_toks)


        # origin_rank_comp = get_rank(logits_comp, explicit_toks)
        # origin_rank_guide = get_rank(logits_guide, explicit_toks)
        # origin_rank_f_hop = get_rank(logits_f_hop, explicit_toks)

    # return origin_prob_comp, origin_rank_comp, origin_prob_guide, origin_rank_guide, origin_prob_f_hop, origin_rank_f_hop
    return origin_prob_comp, wrong_prob_comp

def get_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    query: str,
    template: Template,
    toks: List[int],
    wrong_toks: List[int]=None,
):
    inp = [template.get_prompt(query) for query in [query]]
    inp_tok = tokenizer(inp, padding=True, return_token_type_ids=False, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**inp_tok)["logits"] # out.shape = (1, seq_len, vocab_size)
        logits_comp = torch.softmax(out[0, -1], dim=0).tolist() # len = vocab_size
        prob = 0
        wrong_prob = 0
        for tok in toks:
            prob += logits_comp[tok]
        prob /= len(toks)

        if wrong_toks:
            for tok in wrong_toks:
                wrong_prob += logits_comp[tok]
            wrong_prob /= len(wrong_toks)
    # return origin_prob_comp, origin_rank_comp, origin_prob_guide, origin_rank_guide, origin_prob_f_hop, origin_rank_f_hop
    return prob,wrong_prob