# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fp8 fused-QKV sharding in ``mimo_v2``.

Xiaomi exports the fused QKV pre-sharded for the quant-time TP degree NB=4:
NB contiguous blocks of [Q | K | V] per block. For full-attention layers
(num_kv_heads=4) a block equals one KV head's group; for the SWA layers
(num_kv_heads=8) a block carries TWO KV heads and per-KV-head slicing is
wrong. Tests cover both layer types x TP{1,2,4,8} x all ranks against a
reference dequantization, with ``scaled_quantize`` stubbed to an identity.
"""

import pytest
import torch

import vllm.model_executor.models.mimo_v2 as mimo_v2
from vllm.model_executor.models.mimo_v2 import _shard_fp8_qkv_proj

BLOCK = 128
COLS = 4096
NB = 4  # quant-time TP the fused qkv is pre-sharded for

LAYER_CASES = [
    ("full_attn", 64, 4, 192, 128),  # block == one KV-head group
    ("swa", 64, 8, 192, 128),  # block == two KV-head groups
]

TP_SIZES = [1, 2, 4, 8]


def _identity_scaled_quantize(w, group, dtype, compute_dtype=torch.float32):
    rows = -(-w.shape[0] // group[0])
    cols = -(-w.shape[1] // group[1])
    return w.to(compute_dtype), torch.ones(rows, cols, dtype=torch.float32)


@pytest.fixture
def stubbed_quantize(monkeypatch):
    monkeypatch.setattr(mimo_v2, "scaled_quantize", _identity_scaled_quantize)


def _block_shapes(nh, nk, hd, vd):
    bq = (nh // NB) * hd
    bk = (nk // NB) * hd
    bv = (nk // NB) * vd
    return bq, bk, bv


def _make(nh, nk, hd, vd, seed=0):
    gen = torch.Generator().manual_seed(seed)
    bq, bk, bv = _block_shapes(nh, nk, hd, vd)
    rpb = bq + bk + bv
    total = NB * rpb
    w = (torch.randn(total, COLS, generator=gen) * 0.05).to(torch.float8_e4m3fn)
    s_rows = NB * -(-rpb // BLOCK)
    s = torch.rand(s_rows, COLS // BLOCK, generator=gen) + 0.5
    return w, s


def _reference_parts(w, s, nh, nk, hd, vd):
    """Dequantize and split into head-ordered all_q/all_k/all_v."""
    bq, bk, bv = _block_shapes(nh, nk, hd, vd)
    rpb = bq + bk + bv
    per_block = -(-rpb // BLOCK)
    qs, ks, vs = [], [], []
    for b in range(NB):
        s_b = s[b * per_block : (b + 1) * per_block]
        exp = s_b.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[
            :rpb, :COLS
        ]
        w_b = w[b * rpb : (b + 1) * rpb].to(torch.float32) * exp
        qs.append(w_b[:bq])
        ks.append(w_b[bq : bq + bk])
        vs.append(w_b[bq + bk :])
    return torch.cat(qs), torch.cat(ks), torch.cat(vs)


def _expected_rank_rows(all_q, all_k, all_v, rank, tp, nh, nk, hd, vd):
    nhr = nh // tp
    q = all_q[rank * nhr * hd : (rank + 1) * nhr * hd]
    if nk >= tp:
        kvh = nk // tp
        k = all_k[rank * kvh * hd : (rank + 1) * kvh * hd]
        v = all_v[rank * kvh * vd : (rank + 1) * kvh * vd]
    else:
        kvi = rank // (tp // nk)
        k = all_k[kvi * hd : (kvi + 1) * hd]
        v = all_v[kvi * vd : (kvi + 1) * vd]
    return torch.cat([q, k, v])


@pytest.mark.parametrize("label,nh,nk,hd,vd", LAYER_CASES)
@pytest.mark.parametrize("tp", TP_SIZES)
def test_shard_fp8_qkv_proj_matches_reference(
    stubbed_quantize, label, nh, nk, hd, vd, tp
):
    w, s = _make(nh, nk, hd, vd)
    all_q, all_k, all_v = _reference_parts(w, s, nh, nk, hd, vd)
    nhr = nh // tp
    if nk >= tp:
        exp_rows = nhr * hd + (nk // tp) * (hd + vd)
    else:
        exp_rows = nhr * hd + hd + vd

    for rank in range(tp):
        w_r, s_r = _shard_fp8_qkv_proj(w, s, nh, nk, hd, vd, tp_rank=rank, tp_size=tp)
        # Replica shards (tp > num_kv_heads) are zero-padded to whole 128-row
        # scale blocks; the model loader truncates to the parameter size.
        w_r = w_r[:exp_rows]
        assert tuple(w_r.shape) == (exp_rows, COLS)
        assert s_r.shape[0] == -(-exp_rows // BLOCK)
        s_exp = s_r.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        deq = w_r.to(torch.float32) * s_exp[:exp_rows, :COLS]
        exp = _expected_rank_rows(all_q, all_k, all_v, rank, tp, nh, nk, hd, vd)
        assert torch.allclose(deq, exp, atol=1e-6, rtol=1e-5)


def test_shard_fp8_qkv_proj_rejects_unknown_scale_rows(stubbed_quantize):
    w, s = _make(64, 8, 192, 128)
    with pytest.raises(ValueError, match="scale has 115 rows"):
        _shard_fp8_qkv_proj(w, s[:-1], 64, 8, 192, 128, tp_rank=0, tp_size=8)


def test_shard_fp8_qkv_proj_rejects_indivisible_heads(stubbed_quantize):
    w = torch.zeros(4 * (2 * 192 + 2 * 192 + 2 * 128), COLS, dtype=torch.float8_e4m3fn)
    s = torch.ones(4 * 29, COLS // BLOCK)
    with pytest.raises(ValueError, match="must be divisible"):
        # num_kv_heads=2 is not divisible by NB=4.
        _shard_fp8_qkv_proj(w, s, 8, 2, 192, 128, tp_rank=0, tp_size=1)
