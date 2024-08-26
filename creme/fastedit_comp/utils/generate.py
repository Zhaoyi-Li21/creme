import torch
from typing import List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer
from torch.nn import CrossEntropyLoss
from .template import Template

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

def get_key_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    query_comp: str,
    query_guide: str,
    query_first_hop: str,
    template: Template,
    explicit_toks: List[int],
    implicit_toks: List[int],
    wrong_toks: List[int],
):
    inp = [template.get_prompt(query) for query in [query_comp, query_guide, query_first_hop]]
    inp_tok = tokenizer(inp, padding=True, return_token_type_ids=False, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**inp_tok)["logits"] # out.shape = (1, seq_len, vocab_size)
        logits_comp = torch.softmax(out[0, -1], dim=0).tolist() # len = vocab_size
        logits_guide = torch.softmax(out[1, -1], dim=0).tolist() # len = vocab_size
        logits_f_hop = torch.softmax(out[2, -1], dim=0).tolist() # len = vocab_size

        origin_prob_comp = 0
        wrong_prob_comp = 0
        origin_prob_guide = 0
        origin_prob_f_hop = 0

        for tok in explicit_toks:
            origin_prob_comp += logits_comp[tok]
            origin_prob_guide += logits_guide[tok]

        for tok in wrong_toks:
            wrong_prob_comp += logits_comp[tok]
   
        for tok in implicit_toks:
            origin_prob_f_hop += logits_f_hop[tok]

        origin_prob_comp /= len(explicit_toks)
        wrong_prob_comp /= len(wrong_toks)
        origin_prob_guide /= len(explicit_toks)
        origin_prob_f_hop /= len(implicit_toks)

        # origin_rank_comp = get_rank(logits_comp, explicit_toks)
        # origin_rank_guide = get_rank(logits_guide, explicit_toks)
        # origin_rank_f_hop = get_rank(logits_f_hop, explicit_toks)

    # return origin_prob_comp, origin_rank_comp, origin_prob_guide, origin_rank_guide, origin_prob_f_hop, origin_rank_f_hop
    return origin_prob_comp, wrong_prob_comp, origin_prob_guide, origin_prob_f_hop

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

def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    # print('input_encs:', token_lists)
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # print('length of input_ids:', maxlen)
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def get_ppl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    query: str,
    template: Template,
    target: str,
    wrong_target: str=None,
    ):
    with torch.no_grad():
        
        target_temp = target
        prompt = query + ' ' + target_temp
        prefix = query
        prompt_tokens = tokenizer.encode(prompt)
        prefix_tokens = tokenizer.encode(prefix)
        output_tokens = tokenizer.encode(target_temp)[1:] # Is this true? check if output_tokens == prompt_tokens - prefix_tokens
        output_length = len(output_tokens)
        output_labels = torch.tensor(output_tokens).to('cuda') # shape = [label_len]
        # input = make_inputs(mt.tokenizer, [prompt])
    
        input = make_inputs(tokenizer,[prompt])
        logits = model(**input)["logits"] # figure it out, shape = [bs, seq_len, vocab_size]
        logits = logits[0] # shape = [seq_len, vocab_size]
        assert logits.shape[0] == len(prompt_tokens)
        shift_logits = logits[len(prefix_tokens)-1 : len(prompt_tokens)-1, :] # shift - 1
        assert shift_logits.shape[0] == output_length
        loss_fct = CrossEntropyLoss()
        ppl = loss_fct(shift_logits, output_labels).cpu().float()

        # ppl_wrong = 0.
        # if wrong_target:
        #     target_temp = wrong_target
        #     prompt = query + ' ' + target_temp
        #     prefix = query
        #     prompt_tokens = tokenizer.encode(prompt)
        #     prefix_tokens = tokenizer.encode(prefix)
        #     output_tokens = tokenizer.encode(target_temp)[1:] # Is this true? check if output_tokens == prompt_tokens - prefix_tokens
        #     output_length = len(output_tokens)
        #     output_labels = torch.tensor(output_tokens).to('cuda') # shape = [label_len]
        #     # input = make_inputs(mt.tokenizer, [prompt])
        #     input = tokenizer(prompt)
        #     logits = model(**input)["logits"] # figure it out, shape = [bs, seq_len, vocab_size]
        #     logits = logits[0] # shape = [seq_len, vocab_size]
        #     assert logits.shape[0] == len(prompt_tokens)
        #     shift_logits = logits[len(prefix_tokens)-1 : len(prompt_tokens)-1, :] # shift - 1
        #     assert shift_logits.shape[0] == output_length
        #     loss_fct = CrossEntropyLoss()
        #     ppl_wrong = loss_fct(shift_logits, output_labels)
        
    # return ppl, ppl_wrong
    return float(ppl)

    pass


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
