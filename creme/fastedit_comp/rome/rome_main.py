import sys
sys.path.append(".") 
import time
import torch
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_u import compute_u, compute_u_compedit
from .compute_v import compute_v, compute_v_compedit, compute_v_compedit_v2
from .rome_hparams import ROMEHyperParams, ROMEHyperParams_Original

from utils import nethook
from utils.context import CONTEXT_TEMPLATES


def apply_rome_to_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    requests: List[Tuple],
    hparams: ROMEHyperParams,
    edit_mode: str,
    batch_first: Optional[bool] = True,
    copy: Optional[bool] = False,
    return_diff_weights: Optional[bool] = False
) -> Tuple[PreTrainedModel, Dict[str, torch.Tensor]]:
    r"""
    Edits a pre-trained model using model-editing algorithms.

    Args:
        model (`PreTrainedModel`):
            The pre-trained transformer model to be edited.
        tokeniser (`PreTrainedTokenizer`):
            The pre-trained tokenizer of the model.
        requests (`List[Dict[str, Union[List[str], str]]]`):
            The samples for editing.
        hparams (`ROMEHyperParams`):
            The hyper-parameters of the ROME algorithm.
        batch_first (`bool`, *optional*, defaults to `True`):
            If true, the first dimension of the inputs/outputs of MLP is the batch dimension.
        copy (`bool`, *optional*, defaults to `False`):
            If true, will preserve the original model while creating a new one to edit.
            Note that you are responsible for deallocating the new model's memory to avoid leaks.
        return_diff_weights (`bool`, *optional*, defaults to `False`):
            If true, will return the difference between the updated weights and the original weights.

    Returns:
        model (`PreTrainedModel`):
            The updated transformer model.
        diff_weights (`Dict[str, Tensor]`):
            A dict of diff weights that have been changed.
    """

    model = deepcopy(model) if copy else model

    weights_diff = {}

    for request_comp, request_guide in requests:

        deltas = execute_rome(model, tokenizer, request_comp, request_guide, hparams, edit_mode, batch_first)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                
                # upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                if hparams.transpose == True:
                    # in case: o_proj.shape = [4096, 4096] can not judge
                    upd_matrix = upd_matrix.T
                else:
                    upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                w[...] += upd_matrix

                if return_diff_weights:
                    if w_name in weights_diff:
                        weights_diff[w_name] += upd_matrix.detach().clone()
                    else:
                        weights_diff[w_name] = upd_matrix.detach().clone()

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_diff

def execute_rome(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request_comp: Dict,
    request_guide: Dict,
    hparams: ROMEHyperParams,
    edit_mode: str,
    batch_first: Optional[bool] = True
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    # request = deepcopy(request)

    print("Executing ROME algorithm for the update: " ,
        "[{}] -> [{}]".format(request_comp["prompt"].format(request_comp["subject"]), 
                              request_guide["prompt"].format(request_guide["subject"])))

    start_time = time.time()

    if edit_mode == 'early-mlp':
        edit_module_tmp = hparams.rewrite_module_tmp_mlp
    elif edit_mode == 'middle-attention':
        edit_module_tmp = hparams.rewrite_module_tmp_attn

    # Retrieve weights that user desires to change
    weights = {f"{edit_module_tmp.format(layer)}.weight":
               nethook.get_parameter(model, f"{edit_module_tmp.format(layer)}.weight")
               for layer in hparams.layers}

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u_compedit(
            model,
            tokenizer,
            request_comp,
            hparams,
            edit_mode,
            layer,
            CONTEXT_TEMPLATES,
            batch_first
        )
        print("Left vector shape:", left_vector.shape)
        right_vector, cur_input, cur_output, target = compute_v_compedit(
            model,
            tokenizer,
            request_comp,
            request_guide,
            hparams,
            edit_mode,
            layer,
            CONTEXT_TEMPLATES,
            left_vector,
            batch_first
        )
        print("Right vector shape:", right_vector.shape)
        right_vector = right_vector.to(torch.float16)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{edit_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            print(weights[weight_name].shape)
            if hparams.transpose == True:
                # in case: o_proj.shape = [4096, 4096] can not judge
                upd_matrix = upd_matrix.T
            else:
                upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )
        
            v_new = (weights[weight_name] @ cur_input.unsqueeze(1)).squeeze(1) # shape = [4096]
            v_old = (weights_copy[weight_name] @ cur_input.unsqueeze(1)).squeeze(1) # shape = [4096]
            print('Distance between v_new and v_old:', torch.norm(v_old - v_new))
            print('Distance between v_new and v_target:', torch.norm(target - v_new))
            print('Distance between pre v_old and now v_old:', torch.norm(v_old-cur_output))

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    end_time = time.time()
    print("Time elapsed: {:.2f} seconds".format(end_time - start_time))

    return deltas

def apply_rome_to_model_v2(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    requests: List[Tuple],
    edit_toks: List[int],
    preserve_toks: List[int],
    hparams: ROMEHyperParams,
    edit_mode: str,
    batch_first: Optional[bool] = True,
    copy: Optional[bool] = False,
    return_diff_weights: Optional[bool] = False
) -> Tuple[PreTrainedModel, Dict[str, torch.Tensor]]:
    r"""
    Edits a pre-trained model using model-editing algorithms.

    Args:
        model (`PreTrainedModel`):
            The pre-trained transformer model to be edited.
        tokeniser (`PreTrainedTokenizer`):
            The pre-trained tokenizer of the model.
        requests (`List[Dict[str, Union[List[str], str]]]`):
            The samples for editing.
        hparams (`ROMEHyperParams`):
            The hyper-parameters of the ROME algorithm.
        batch_first (`bool`, *optional*, defaults to `True`):
            If true, the first dimension of the inputs/outputs of MLP is the batch dimension.
        copy (`bool`, *optional*, defaults to `False`):
            If true, will preserve the original model while creating a new one to edit.
            Note that you are responsible for deallocating the new model's memory to avoid leaks.
        return_diff_weights (`bool`, *optional*, defaults to `False`):
            If true, will return the difference between the updated weights and the original weights.

    Returns:
        model (`PreTrainedModel`):
            The updated transformer model.
        diff_weights (`Dict[str, Tensor]`):
            A dict of diff weights that have been changed.
    """

    model = deepcopy(model) if copy else model

    weights_diff = {}

    for request_comp, request_guide in requests:

        deltas = execute_rome_v2(model, tokenizer, request_comp, request_guide, edit_toks, preserve_toks, hparams, edit_mode, batch_first)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                
                # upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                if hparams.transpose == True:
                    # in case: o_proj.shape = [4096, 4096] can not judge
                    upd_matrix = upd_matrix.T
                else:
                    upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                w[...] += upd_matrix

                if return_diff_weights:
                    if w_name in weights_diff:
                        weights_diff[w_name] += upd_matrix.detach().clone()
                    else:
                        weights_diff[w_name] = upd_matrix.detach().clone()

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_diff

def execute_rome_v2(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request_comp: Dict,
    request_guide: Dict,
    edit_toks: List[int],
    preserve_toks: List[int],
    hparams: ROMEHyperParams,
    edit_mode: str,
    batch_first: Optional[bool] = True
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    # request = deepcopy(request)

    print("Executing ROME algorithm for the update: " ,
        "[{}] -> [{}]".format(request_comp["prompt"].format(request_comp["subject"]), 
                              request_guide["prompt"].format(request_guide["subject"])))
    
    unembed_matrix = model.state_dict()['lm_head.weight']

    start_time = time.time()


    if edit_mode == 'early-mlp':
        edit_module_tmp = hparams.rewrite_module_tmp_mlp
    elif edit_mode == 'middle-attention':
        edit_module_tmp = hparams.rewrite_module_tmp_attn

    # Retrieve weights that user desires to change
    weights = {f"{edit_module_tmp.format(layer)}.weight":
               nethook.get_parameter(model, f"{edit_module_tmp.format(layer)}.weight")
               for layer in hparams.layers}

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u_compedit(
            model,
            tokenizer,
            request_comp,
            hparams,
            edit_mode,
            layer,
            CONTEXT_TEMPLATES,
            batch_first
        )
        print("Left vector shape:", left_vector.shape)
        right_vector, cur_input, cur_output, target = compute_v_compedit_v2(
            model,
            tokenizer,
            request_comp,
            request_guide,
            edit_toks,
            preserve_toks,
            unembed_matrix,
            hparams,
            edit_mode,
            layer,
            CONTEXT_TEMPLATES,
            left_vector,
            batch_first
        )
        print("Right vector shape:", right_vector.shape)
        right_vector = right_vector.to(torch.float16)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{edit_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            print(weights[weight_name].shape)
            if hparams.transpose == True:
                # in case: o_proj.shape = [4096, 4096] can not judge
                upd_matrix = upd_matrix.T
            else:
                upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )
        
            v_new = (weights[weight_name] @ cur_input.unsqueeze(1)).squeeze(1) # shape = [4096]
            v_old = (weights_copy[weight_name] @ cur_input.unsqueeze(1)).squeeze(1) # shape = [4096]
            print('Distance between v_new and v_old:', torch.norm(v_old - v_new))
            print('Distance between v_new and v_target:', torch.norm(target - v_new))
            print('Distance between pre v_old and now v_old:', torch.norm(v_old-cur_output))

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    end_time = time.time()
    print("Time elapsed: {:.2f} seconds".format(end_time - start_time))

    return deltas

def apply_rome_to_model_v3(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    requests: List[Dict[str, Union[List[str], str]]],
    hparams: ROMEHyperParams,
    batch_first: Optional[bool] = True,
    copy: Optional[bool] = False,
    return_diff_weights: Optional[bool] = False
) -> Tuple[PreTrainedModel, Dict[str, torch.Tensor]]:
    r"""
    Edits a pre-trained model using model-editing algorithms.

    Args:
        model (`PreTrainedModel`):
            The pre-trained transformer model to be edited.
        tokeniser (`PreTrainedTokenizer`):
            The pre-trained tokenizer of the model.
        requests (`List[Dict[str, Union[List[str], str]]]`):
            The samples for editing.
        hparams (`ROMEHyperParams`):
            The hyper-parameters of the ROME algorithm.
        batch_first (`bool`, *optional*, defaults to `True`):
            If true, the first dimension of the inputs/outputs of MLP is the batch dimension.
        copy (`bool`, *optional*, defaults to `False`):
            If true, will preserve the original model while creating a new one to edit.
            Note that you are responsible for deallocating the new model's memory to avoid leaks.
        return_diff_weights (`bool`, *optional*, defaults to `False`):
            If true, will return the difference between the updated weights and the original weights.

    Returns:
        model (`PreTrainedModel`):
            The updated transformer model.
        diff_weights (`Dict[str, Tensor]`):
            A dict of diff weights that have been changed.
    """

    model = deepcopy(model) if copy else model

    weights_diff = {}

    for request in requests:
        deltas = execute_rome_v3(model, tokenizer, request, hparams, batch_first)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                w[...] += upd_matrix

                if return_diff_weights:
                    if w_name in weights_diff:
                        weights_diff[w_name] += upd_matrix.detach().clone()
                    else:
                        weights_diff[w_name] = upd_matrix.detach().clone()

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_diff

def execute_rome_v3(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    batch_first: Optional[bool] = True
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)

    print("Executing ROME algorithm for the update: "
          "[{}] -> [{}]".format(request["prompt"].format(request["subject"]), request["target"]))

    start_time = time.time()

    # Retrieve weights that user desires to change
    weights = {f"{hparams.rewrite_module_tmp.format(layer)}.weight":
               nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(layer)}.weight")
               for layer in hparams.layers}

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tokenizer,
            request,
            hparams,
            layer,
            CONTEXT_TEMPLATES,
            batch_first
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tokenizer,
            request,
            hparams,
            layer,
            left_vector,
            CONTEXT_TEMPLATES,
            batch_first
        )
        print("Right vector shape:", right_vector.shape)
        right_vector = right_vector.to(torch.float16)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    end_time = time.time()
    print("Time elapsed: {:.2f} seconds".format(end_time - start_time))

    return deltas

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    r"""
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """
    
    if matrix.shape == shape:
        print('update without transpose')
        return matrix
    elif matrix.T.shape == shape:
        print('update with transpose')
        return matrix.T
    else:
        raise ValueError("Update matrix computed by ROME does not match original weight shape. "
                         "Check for bugs in the code?")
