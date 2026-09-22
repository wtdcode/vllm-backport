# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharding of MiMo-V2 fused fp8 qkv_proj checkpoints.

The checkpoint stores the fused QKV pre-sharded for ``ckpt_tp`` ranks as
``[Q_i | K_i | V_i]`` per shard, each shard block-quantized on its own, so
the block tiling restarts at every shard boundary.
"""

import math

import pytest
import torch

from vllm.model_executor.models.mimo_v2 import _shard_fp8_qkv_proj

BLOCK = 128
FP8 = torch.float8_e4m3fn


def _quantize_blocks(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    pad = -rows % BLOCK
    xp = torch.nn.functional.pad(x, (0, 0, 0, pad))
    blocks = xp.view(xp.shape[0] // BLOCK, BLOCK, cols // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / torch.finfo(FP8).max
    q = (blocks / scale).to(FP8).view(xp.shape)[:rows]
    return q, scale.view(blocks.shape[0], blocks.shape[2])


def _dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    s = s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return w.float() * s[: w.shape[0], : w.shape[1]]


def _make_checkpoint(num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp):
    torch.manual_seed(0)
    hidden = 256
    q = [torch.randn(head_dim, hidden) for _ in range(num_heads)]
    # Give each KV head a distinct magnitude so a mixed-up head is obvious.
    k = [torch.randn(head_dim, hidden) * (1 + h) for h in range(num_kv_heads)]
    v = [torch.randn(v_head_dim, hidden) * (2 + h) for h in range(num_kv_heads)]
    q_per, kv_per = num_heads // ckpt_tp, max(1, num_kv_heads // ckpt_tp)
    ws, ss = [], []
    for i in range(ckpt_tp):
        if num_kv_heads >= ckpt_tp:
            kv_ids = range(i * kv_per, (i + 1) * kv_per)
        else:
            kv_ids = [i * num_kv_heads // ckpt_tp]
        shard = torch.cat(
            q[i * q_per : (i + 1) * q_per]
            + [k[h] for h in kv_ids]
            + [v[h] for h in kv_ids]
        )
        w, s = _quantize_blocks(shard)
        ws.append(w)
        ss.append(s)
    return torch.cat(ws), torch.cat(ss), q, k, v


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,v_head_dim",
    [
        (64, 4, 192, 128),  # MiMo-V2.6-Flash full attention
        (64, 8, 192, 128),  # MiMo-V2.6-Flash sliding window attention
    ],
)
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_shard_fp8_qkv_proj(num_heads, num_kv_heads, head_dim, v_head_dim, tp_size):
    ckpt_tp = 4
    w, s, q, k, v = _make_checkpoint(
        num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
    )
    shard_rows = w.shape[0] // ckpt_tp
    assert s.shape[0] == ckpt_tp * math.ceil(shard_rows / BLOCK)

    # Reference heads as stored (i.e. after the checkpoint's own fp8 rounding).
    q_per, kv_per = num_heads // ckpt_tp, max(1, num_kv_heads // ckpt_tp)
    q_rows = q_per * head_dim
    s_rows = math.ceil(shard_rows / BLOCK)
    ref_q, ref_k, ref_v = {}, {}, {}
    for i in range(ckpt_tp):
        d = _dequant(
            w[i * shard_rows : (i + 1) * shard_rows], s[i * s_rows : (i + 1) * s_rows]
        )
        for j in range(q_per):
            ref_q[i * q_per + j] = d[j * head_dim : (j + 1) * head_dim]
        for j in range(kv_per):
            h = i * kv_per + j
            ref_k[h] = d[q_rows + j * head_dim : q_rows + (j + 1) * head_dim]
            v_lo = q_rows + kv_per * head_dim + j * v_head_dim
            ref_v[h] = d[v_lo : v_lo + v_head_dim]

    for rank in range(tp_size):
        w_rank, s_rank = _shard_fp8_qkv_proj(
            w, s, num_heads, num_kv_heads, head_dim, v_head_dim,
            tp_rank=rank, tp_size=tp_size, ckpt_tp=ckpt_tp,
        )  # fmt: skip
        q_ids = range(rank * num_heads // tp_size, (rank + 1) * num_heads // tp_size)
        if tp_size <= num_kv_heads:
            per = num_kv_heads // tp_size
            kv_ids = range(rank * per, (rank + 1) * per)
        else:
            kv_ids = [rank // (tp_size // num_kv_heads)]
        expected = torch.cat(
            [ref_q[h] for h in q_ids]
            + [ref_k[h] for h in kv_ids]
            + [ref_v[h] for h in kv_ids]
        )
        assert w_rank.dtype == FP8
        assert w_rank.shape == expected.shape
        assert s_rank.shape[0] == math.ceil(expected.shape[0] / BLOCK)
        got = _dequant(w_rank, s_rank)
        rel = (got - expected).norm() / expected.norm()
        if tp_size == ckpt_tp:
            # One checkpoint shard per rank: no re-quantization error.
            assert rel < 1e-6
        else:
            assert rel < 3e-2
