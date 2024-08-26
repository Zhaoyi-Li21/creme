import sys
sys.path.append(".") 
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

from .repr_tools import get_reprs_at_idxs, get_reprs_at_word_tokens, get_words_idxs_in_templates
from .rome_hparams import ROMEHyperParams
from utils import nethook


def compute_v(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
    batch_first: Optional[bool] = True
) -> torch.Tensor:
    r"""
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")

    prompt_tok = tokenizer.tokenize(context_templates[0].format(request["prompt"]))
    compl_tok = tokenizer.tokenize(context_templates[0].format(request["prompt"]) + request["target"])
    target_len = len(compl_tok) - len(prompt_tok)

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = [context.format(request["prompt"]) + request["target"] for context in context_templates]
    kl_prompts = ["{} is a", "{}是一个"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tokenizer(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        padding=True,
        return_token_type_ids=False,
        return_tensors="pt"
    ).to(model.device)

    # Compute rewriting targets for left-padded sequences
    rewriting_targets = torch.tensor(-100).repeat(len(rewriting_prompts), *input_tok["input_ids"].shape[1:]).to(model.device)
    for i in range(len(rewriting_prompts)):
        rewriting_targets[i, -target_len-1:-1] = input_tok["input_ids"][i, -target_len:].clone() # build labels

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [find_fact_lookup_idx(prompt, request["subject"], tokenizer,
                                        hparams.fact_token if i <= len(context_templates) else "last", verbose=(i == 0))
                                        for i, prompt in enumerate(all_prompts)]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    n_embed = model.config.n_embd if hasattr(model.config, "n_embed") else model.config.hidden_size # for LLaMA model
    delta = torch.zeros((n_embed,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        # Store initial value of the vector of interest
        if target_init is None:
            print("Recording initial value of v*")
            # Initial value is recorded for the clean sentence
            target_init = cur_out[0, lookup_idxs[0]].detach().clone()

        for i, idx in enumerate(lookup_idxs):
            cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr, weight_decay=hparams.v_weight_decay)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.Trace(
            module=model,
            layer=hparams.mlp_module_tmp.format(layer),
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [logits[i - len(kl_prompts), idx, :] for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])],
                dim=0
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = log_probs.gather(2, torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2)).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(dim=1) / target_len
        nll_loss = nll_loss_each.mean()
        kl_loss = torch.nn.functional.kl_div(kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean")
        kl_loss *= hparams.kl_factor
        loss = nll_loss + kl_loss
        print(f"loss {np.round(loss.item(), 3)} = "
              f"{np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} "
              f"avg prob of [{request['target']}] {np.round(torch.exp(-nll_loss_each).mean().item(), 4)}")

        if loss < 5e-3: # early-stopping
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tokenizer,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
        batch_first=batch_first
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {np.round((target - cur_output).norm().item(), 3)}")
    print(f"Change in target norm: {np.round(target_init.norm().item(), 3)} to {np.round(target.norm().item(), 3)} => "
          f"{np.round((target.norm() - target_init.norm()).item(), 3)}")
    print(f"Division Factor: {np.round(torch.dot(cur_input, left_vector).item(), 3)}")
    print(f"Right vector norm: {np.round(right_vector.norm().item(), 3)}")

    return right_vector

def compute_v_compedit(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request_comp: Dict[str, str], # request["comp"]
    request_guide: Dict[str, str],
    hparams: ROMEHyperParams,
    edit_mode: str, # 'early-mlp' OR 'middle-attention'
    layer: int,
    context_templates: List[str],
    left_vector: torch.Tensor,
    batch_first: Optional[bool] = True
) -> torch.Tensor:
    r"""
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing right vector (v)...")

    if edit_mode == 'early-mlp':
        edit_module_tmp = hparams.rewrite_module_tmp_mlp
    elif edit_mode == 'middle-attention':
        edit_module_tmp = hparams.rewrite_module_tmp_attn

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        module_template=edit_module_tmp,
        track="out",
        batch_first=batch_first
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request_guide["subject"]
        print(f"Selected u projection object {word}")
        cur_repr = get_reprs_at_word_tokens(
            context_templates=[templ.format(request_guide["prompt"]) for templ in context_templates],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_"):],
            **word_repr_args
        ).mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = get_reprs_at_idxs(
            contexts=[templ.format(request_guide["prompt"].format(request_guide["subject"])) for templ in context_templates],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args
        ).mean(0)
        print("Selected v projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment
    target = cur_repr

    # # Calculate k*? or use the method in compute_u_compedit()?
    # cur_input, cur_output = get_module_input_output_at_word(
    #     model,
    #     tokenizer,
    #     layer,
    #     context_template=request_comp["prompt"],
    #     word=request_comp["subject"],
    #     module_template=edit_module_tmp,
    #     fact_token_strategy=hparams.fact_token,
    #     batch_first=batch_first
    # )

    word_repr_args = dict(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        module_template=edit_module_tmp,
        track="both",
        batch_first=batch_first
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request_comp["subject"]
        print(f"Selected u projection object {word}")
        cur_input, cur_output = get_reprs_at_word_tokens(
            context_templates=[templ.format(request_comp["prompt"]) for templ in context_templates],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_"):],
            **word_repr_args
        )
        cur_input = cur_input.mean(0)
        cur_output = cur_output.mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_input, cur_output = get_reprs_at_idxs(
            contexts=[templ.format(request_comp["prompt"].format(request_comp["subject"])) for templ in context_templates],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args
        )
        cur_input = cur_input.mean(0)
        cur_output = cur_output.mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")


    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)

    if hparams.check_updated_vector == True:
        return right_vector, cur_input, cur_output, target
    else:
        return right_vector, None, None, None

def compute_v_compedit_v2(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request_comp: Dict[str, str], # request["comp"]
    request_guide: Dict[str, str],
    edit_toks: List[int],
    preserve_toks: List[int],
    unembed: torch.Tensor,
    hparams: ROMEHyperParams,
    edit_mode: str, # 'early-mlp' OR 'middle-attention'
    layer: int,
    context_templates: List[str],
    left_vector: torch.Tensor,
    batch_first: Optional[bool] = True
) -> torch.Tensor:
    r"""
    Computes the right vector used in constructing the rank-1 update matrix.
    version 2 for compositional editing:
    vector v is derived through optimization:

    """

    print("Computing right vector (v)...")

    if edit_mode == 'early-mlp':
        edit_module_tmp = hparams.rewrite_module_tmp_mlp
    elif edit_mode == 'middle-attention':
        edit_module_tmp = hparams.rewrite_module_tmp_attn

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        module_template=edit_module_tmp,
        track="out",
        batch_first=batch_first
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request_guide["subject"]
        print(f"Selected u projection object {word}")
        cur_repr = get_reprs_at_word_tokens(
            context_templates=[templ.format(request_guide["prompt"]) for templ in context_templates],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_"):],
            **word_repr_args
        ).mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = get_reprs_at_idxs(
            contexts=[templ.format(request_guide["prompt"].format(request_guide["subject"])) for templ in context_templates],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args
        ).mean(0)
        print("Selected v projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")


    target_guide = cur_repr # target_guide.shape = [hid_dim]


    word_repr_args = dict(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        module_template=edit_module_tmp,
        track="both",
        batch_first=batch_first
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request_comp["subject"]
        print(f"Selected u projection object {word}")
        cur_input, cur_output = get_reprs_at_word_tokens(
            context_templates=[templ.format(request_comp["prompt"]) for templ in context_templates],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_"):],
            **word_repr_args
        )
        cur_input = cur_input.mean(0)
        cur_output = cur_output.mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_input, cur_output = get_reprs_at_idxs(
            contexts=[templ.format(request_comp["prompt"].format(request_comp["subject"])) for templ in context_templates],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args
        )
        cur_input = cur_input.mean(0)
        cur_output = cur_output.mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    target_old = cur_output
    '''
    now we have target_guide, target_old, and we aim to get target_new through an optimization procedure
    optimization objective: 
    \lambda_1(e.g.,=10) \Sigma_{j in edit_toks} ((W \cdot target_new)[j] - (W \cdot target_guide)[j])^2 
    + \lambda_2(e.g.,=10) \Sigma_{j in preserve_toks} ((W \cdot target_new)[j] - (W \cdot target_old)[j])^2
    + \lambda_3(e.g.,=1) \Sigma_{j in other_toks} ((W \cdot target_new)[j] - (W \cdot target_old)[j])^2
    '''
    lambda_edit = 1.
    lambda_preserve = 30.
    lambda_others = 1.

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    n_embed = model.config.n_embd if hasattr(model.config, "n_embed") else model.config.hidden_size # for LLaMA model
    print('n_embed:', n_embed)
    unembed = unembed.float()
    target_guide = target_guide.float()
    target_old = target_old.float()
    target_new = torch.zeros((n_embed,), dtype= torch.float32, requires_grad=True, device="cuda")

    # Optimizer
    # opt = torch.optim.Adam([target_new], lr=hparams.v_lr, weight_decay=hparams.v_weight_decay)
    # opt = torch.optim.Adam([target_new], lr=1.)
    opt = torch.optim.SGD([target_new], lr=100.)

    logits_guide = torch.matmul(unembed, target_guide.unsqueeze(1))
    p_guide = torch.softmax(logits_guide, 0)
    logits_old = torch.matmul(unembed, target_old.unsqueeze(1))
    p_old = torch.softmax(logits_old, 0)
    # Execute optimization
    for it in range(25):
        opt.zero_grad()

        print(target_new.grad)
        print(target_new)

        # Forward propagation
        logits_new = torch.matmul(unembed, target_new.unsqueeze(1))
        p_new = torch.softmax(logits_new, 0)

        loss = 0
        edit_loss = 0
        preserve_loss = 0
        others_loss = 0
        print('shape:',p_new.shape[0])
        print(edit_toks)
        print(preserve_toks)
        for j in range(p_new.shape[0]):
            if j in edit_toks:
                edit_loss += torch.abs((p_new[j] - p_guide[j])) 
                print('edit toks:',p_new[j].item(),p_guide[j].item(),p_old[j].item())
            elif j in preserve_toks:
                preserve_loss += torch.abs((p_new[j] - p_old[j]))
                print('preserve toks:',p_new[j].item(),p_guide[j].item(), p_old[j].item())
            else:
                others_loss += torch.abs((p_new[j] - p_old[j]))
        others_toks_num = p_new.shape[0] - len(edit_toks) - len(preserve_toks)
        edit_loss = edit_loss / len(edit_toks)
        preserve_loss = preserve_loss / len(preserve_toks)
        others_loss = others_loss / others_toks_num
        loss = (edit_loss) * lambda_edit \
                + (preserve_loss) * lambda_preserve \
                + (others_loss) * lambda_others
        
        '''
        print training info
        '''
        print(f"total loss = {np.round(loss.item(), 7)}")
        print(f"edit loss = {np.round(edit_loss.item(), 7)}")
        print(f"preserve loss = {np.round(preserve_loss.item(), 7)}")
        print(f"others loss = {np.round(others_loss.item(), 7)}")
        
        # if loss < 9e-6 and edit_loss < 1e-7: # early-stopping
        #     break

            

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()
        print(target_new.grad)
        print(target_new)
        # Project within L2 ball
        # max_norm = hparams.clamp_norm_factor * target_init.norm()
        # if delta.norm() > max_norm:
        #     with torch.no_grad():
        #         delta[...] = delta * max_norm / delta.norm()


    target_new = target_new.half()

    right_vector = (target_new - cur_output) / torch.dot(cur_input, left_vector)

    if hparams.check_updated_vector == True:
        return right_vector, cur_input, cur_output, target_new
    else:
        return right_vector, None, None, None

def get_module_input_output_at_word(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
    batch_first: Optional[bool] = True
) -> Tuple[torch.Tensor]:
    r"""
    Retrieves detached representations for a word at the input and output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        module_template=module_template,
        track="both",
        batch_first=batch_first
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        l_input, l_output = get_reprs_at_word_tokens(
            context_templates=[context_template],
            words=[word],
            subtoken=fact_token_strategy[len("subject_"):],
            **word_repr_args
        )
    elif fact_token_strategy == "last":
        l_input, l_output = get_reprs_at_idxs(
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tokenizer: PreTrainedTokenizer,
    fact_token_strategy: str,
    verbose: Optional[bool] = True,
) -> int:
    r"""
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        ret = get_words_idxs_in_templates(
            tokenizer=tokenizer,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_"):],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
              tokenizer.decode(tokenizer(sentence)["input_ids"][ret]))

    return ret
