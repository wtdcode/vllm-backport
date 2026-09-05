# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PP-missing layers must be skipped by the glm5next FP8 dequant load helpers.

With pipeline parallelism every rank iterates the full checkpoint; layers not
held by this rank have no parameters, so the helpers must not buffer them or
look their params up.
"""

import torch

from vllm.models.glm5next.nvidia.model import (
    _try_load_fp8_attn_proj,
    _try_load_fp8_indexer_wk,
)

PP_MISSING = ["model.layers.0."]


class _ParamStub:
    def __init__(self):
        self.calls = []

    def weight_loader(self, param, weight, *args):
        self.calls.append((weight, args))


def _fp8_pair(name_prefix):
    weight = torch.full((128, 128), 1.0, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, 1)
    return (
        (f"{name_prefix}.weight", weight),
        (f"{name_prefix}.weight_scale_inv", scale),
    )


def test_indexer_wk_skips_pp_missing_layer():
    params_dict: dict = {}  # layer lives on another PP rank
    buf, loaded = {}, set()
    (w_name, w), (s_name, s) = _fp8_pair("model.layers.0.self_attn.indexer.wk")

    assert _try_load_fp8_indexer_wk(w_name, w, buf, params_dict, loaded, PP_MISSING)
    assert _try_load_fp8_indexer_wk(s_name, s, buf, params_dict, loaded, PP_MISSING)
    assert buf == {}
    assert loaded == set()


def test_indexer_wk_dequantizes_owned_layer():
    param = _ParamStub()
    params_dict = {"model.layers.1.self_attn.indexer.wk_weights_proj.weight": param}
    buf, loaded = {}, set()
    (w_name, w), (s_name, s) = _fp8_pair("model.layers.1.self_attn.indexer.wk")

    assert _try_load_fp8_indexer_wk(w_name, w, buf, params_dict, loaded, [])
    assert _try_load_fp8_indexer_wk(s_name, s, buf, params_dict, loaded, [])
    [(weight, args)] = param.calls
    assert args == (0,)
    assert weight.dtype == torch.bfloat16
    assert weight.shape == (128, 128)
    assert "model.layers.1.self_attn.indexer.wk_weights_proj.weight" in loaded


def test_attn_proj_skips_pp_missing_layer():
    params_dict: dict = {}
    buf, loaded = {}, set()
    (w_name, w), (s_name, s) = _fp8_pair("model.layers.0.self_attn.q_a_proj")

    assert _try_load_fp8_attn_proj(w_name, w, buf, params_dict, loaded, 0, PP_MISSING)
    assert _try_load_fp8_attn_proj(s_name, s, buf, params_dict, loaded, 0, PP_MISSING)
    assert buf == {}
    assert loaded == set()


def test_attn_proj_dequantizes_owned_layer():
    param = _ParamStub()
    params_dict = {"model.layers.1.self_attn.fused_qkv_a_proj.weight": param}
    buf, loaded = {}, set()
    (w_name, w), (s_name, s) = _fp8_pair("model.layers.1.self_attn.q_a_proj")

    assert _try_load_fp8_attn_proj(w_name, w, buf, params_dict, loaded, 0, [])
    assert _try_load_fp8_attn_proj(s_name, s, buf, params_dict, loaded, 0, [])
    [(weight, args)] = param.calls
    assert args == (0,)
    assert weight.dtype == torch.bfloat16
    assert weight.shape == (128, 128)
    assert "model.layers.1.self_attn.fused_qkv_a_proj.weight" in loaded
