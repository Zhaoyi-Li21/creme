import sys
sys.path.append(".") 
from dataclasses import dataclass
from typing import List

from utils.hparams import HyperParams


@dataclass
class ROMEHyperParams(HyperParams):
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    # Comp Edit
    edit_mode:str
    rewrite_module_tmp_mlp:str
    rewrite_module_tmp_attn:str
    transpose:bool
    # debug
    check_updated_vector:bool
    @classmethod
    def from_name(cls, name: str):
        data = dict(
            layers=[5],
            fact_token="subject_last",
            v_num_grad_steps=20,
            v_lr=1e-1,
            v_loss_layer=27,
            v_weight_decay=1e-3,
            clamp_norm_factor=4,
            kl_factor=0.0625,
            mom2_adjustment=False,
            edit_mode=None,
            check_updated_vector=False,
            transpose=False,
            rewrite_module_tmp="transformer.h.{}.mlp.fc_out",
            rewrite_module_tmp_mlp="transformer.h.{}.mlp.fc_out",
            rewrite_module_tmp_attn="transformer.h.{}.attn.c_proj",
            layer_module_tmp="transformer.h.{}",
            mlp_module_tmp="transformer.h.{}.mlp",
            attn_module_tmp="transformer.h.{}.attn",
            ln_f_module="transformer.ln_f",
            lm_head_module="lm_head",
            mom2_dataset="wikipedia",
            mom2_n_samples=100000,
            mom2_dtype="float16"
        )

        if name == "gpt-j-6b":
            pass
        elif name == "llama-7b":
            r"""
            Supports: LLaMA-7B, LLaMA-2-7B, Baichuan-7B, InternLM-7B...
            """
            data.update(dict(
                # layers = [15, 16, 17, 18, 19, 20, 21, 22, 23],
                # layers = [17, 18, 19, 20, 21, 22, 23],
                # layers=[13, 14, 15, 16, 17, 18],
                layers=[17, 18, 19, 20, 21, 22, 23], # for llama2-7b and openalpaca-3b
                # layers=[14, 15, 16, 17, 18, 19, 20], # for llama2-13b
                # layers=[2, 3, 4, 5, 6, 7, 8],
                # layers=[27, 28, 29, 30, 31],
                # layers=[7,8,9,10,11,12],
                # layers=[22, 23, 24, 25],
                # layers=[7, 8, 9, 10, 11, 12, 13],
                # fact_token="subject_last",
                fact_token="last",
                v_loss_layer=31,
                rewrite_module_tmp_mlp="model.layers.{}.mlp.down_proj",
                rewrite_module_tmp_attn="model.layers.{}.self_attn.o_proj",
                # edit_mode="early-mlp",
                edit_mode="middle-attention",
                check_updated_vector=True,
                transpose=True,
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name == "llama-13b":
            r"""
            Supports LLaMA-13B, LLaMA-2-13B, Baichuan-13B...
            """
            data.update(dict(
                layers=[10],
                v_loss_layer=39,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name == "falcon-7b":
            data.update(dict(
                v_loss_layer=31,
                rewrite_module_tmp="transformer.h.{}.mlp.dense_4h_to_h",
                attn_module_tmp="transformer.h.{}.self_attention"
            ))
        elif name == "bloom-7b1":
            data.update(dict(
                v_lr=2e-1,
                v_loss_layer=29,
                rewrite_module_tmp="transformer.h.{}.mlp.dense_4h_to_h",
                attn_module_tmp="transformer.h.{}.self_attention"
            ))
        else:
            raise NotImplementedError

        return cls(**data)


@dataclass
class ROMEHyperParams_Original(HyperParams):
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    @classmethod
    def from_name(cls, name: str):
        data = dict(
            layers=[5],
            fact_token="subject_last",
            v_num_grad_steps=20,
            v_lr=1e-1,
            v_loss_layer=27,
            v_weight_decay=1e-3,
            clamp_norm_factor=4,
            kl_factor=0.0625,
            mom2_adjustment=False,
            rewrite_module_tmp="transformer.h.{}.mlp.fc_out",
            layer_module_tmp="transformer.h.{}",
            mlp_module_tmp="transformer.h.{}.mlp",
            attn_module_tmp="transformer.h.{}.attn",
            ln_f_module="transformer.ln_f",
            lm_head_module="lm_head",
            mom2_dataset="wikipedia",
            mom2_n_samples=100000,
            mom2_dtype="float16"
        )

        if name == "gpt-j-6b":
            pass
        elif name == "llama-7b":
            r"""
            Supports: LLaMA-7B, LLaMA-2-7B, Baichuan-7B, InternLM-7B...
            """
            data.update(dict(
                v_loss_layer=31,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name == "llama-13b":
            r"""
            Supports LLaMA-13B, LLaMA-2-13B, Baichuan-13B...
            """
            data.update(dict(
                layers=[10],
                v_loss_layer=39,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name == "falcon-7b":
            data.update(dict(
                v_loss_layer=31,
                rewrite_module_tmp="transformer.h.{}.mlp.dense_4h_to_h",
                attn_module_tmp="transformer.h.{}.self_attention"
            ))
        elif name == "bloom-7b1":
            data.update(dict(
                v_lr=2e-1,
                v_loss_layer=29,
                rewrite_module_tmp="transformer.h.{}.mlp.dense_4h_to_h",
                attn_module_tmp="transformer.h.{}.self_attention"
            ))
        else:
            raise NotImplementedError

        return cls(**data)
