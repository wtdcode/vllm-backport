# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 SM8x sparse MLA: logical top-k rows with holes, read through a paged
BF16 or fp8_ds_mla cache, match sink-softmax attention over each row's live
keys."""

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

capability = current_platform.get_device_capability()
if not current_platform.is_cuda() or capability is None or capability.major != 8:
    pytest.skip("HYV4 SM8x sparse MLA", allow_module_level=True)

from vllm.models.hy_v4.ampere.sparse_mla import (  # noqa: E402
    HYV4AmpereMLASparseImpl,
    HYV4AmpereMLASparseMetadata,
)

DEVICE = "cuda"
RANK, DIM, BLOCK, TOPK, SCALE = 512, 576, 64, 2048, 0.07
FP8_ROW_BYTES = 656


def _fp8_ds_mla_pool(values):
    """Packs BF16 rows with the engine's writer and decodes them in torch.

    Returns the `[blocks, slots, 656]` byte pool and the rows it stores:
    512 e4m3 latents, four fp32 scales (one per 128 latents), 64 BF16 RoPE.
    """
    num_blocks, slots, _ = values.shape
    pool = torch.zeros(
        num_blocks, slots, FP8_ROW_BYTES, dtype=torch.uint8, device=DEVICE
    )
    flat = values.reshape(-1, DIM)
    ops.concat_and_cache_mla(
        flat[:, :RANK].contiguous(),
        flat[:, RANK:].contiguous(),
        pool,
        torch.arange(flat.shape[0], device=DEVICE),
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1, device=DEVICE),
    )
    latent = pool[..., :RANK].contiguous().view(torch.float8_e4m3fn).float()
    scales = pool[..., RANK : RANK + 16].contiguous().view(torch.float32)
    latent = (latent.unflatten(-1, (4, 128)) * scales[..., None]).flatten(-2)
    rope = pool[..., RANK + 16 :].contiguous().view(torch.bfloat16).float()
    stored = torch.cat((latent, rope), -1)
    # e4m3 keeps three mantissa bits; subnormals step by scale * 2**-9.
    bound = values[..., :RANK].float().abs() / 16
    bound += scales.repeat_interleave(128, -1) * 2**-9
    assert ((latent - values[..., :RANK].float()).abs() <= bound).all()
    assert torch.equal(rope, values[..., RANK:].float())
    return pool, stored


def _reference(q, rows, valid, sinks):
    scores = torch.einsum("thd,tkd->thk", q.float(), rows.float()) * SCALE
    scores.masked_fill_(~valid[:, None], -torch.inf)
    if sinks is not None:
        sink = sinks[None, :, None].expand(q.shape[0], -1, 1)
        scores = torch.cat((scores, sink), -1)
    probs = scores.softmax(-1).nan_to_num(0.0)[..., : rows.shape[1]]
    return torch.einsum("thk,tkd->thd", probs, rows.float()[..., :RANK])


# 120 prefill rows exceed one CTA per SM, where the fp8 read narrows its tile.
@pytest.mark.parametrize("query_len", [1, 40], ids=["decode", "prefill"])
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("with_sink", [True, False])
@pytest.mark.parametrize("kv_cache_dtype", ["bfloat16", "fp8_ds_mla"])
def test_forward_mqa_matches_sink_softmax(query_len, heads, with_sink, kv_cache_dtype):
    torch.manual_seed(0)
    num_reqs, seq_len, num_blocks = 3, 3000, 160
    num_tokens = num_reqs * query_len
    # Blocks of other layers sit between this cache's blocks in the pool.
    pool = (torch.randn(num_blocks, 2 * BLOCK, DIM, device=DEVICE) * 0.5).bfloat16()
    # Per-tile fp8 scales differ when the latent tiles do.
    pool[..., :RANK] *= torch.tensor([0.05, 1.0, 2.0, 0.3], device=DEVICE)[
        torch.randint(4, (num_blocks, 2 * BLOCK, 4), device=DEVICE)
    ].repeat_interleave(128, -1)
    stored = pool
    if kv_cache_dtype == "fp8_ds_mla":
        pool, stored = _fp8_ds_mla_pool(pool)
    cache, stored = pool[:, :BLOCK], stored[:, :BLOCK]
    block_table = (
        torch.randperm(num_blocks, device=DEVICE)[: num_reqs * 47]
        .reshape(num_reqs, 47)
        .int()
    )
    logical = torch.stack(
        [torch.randperm(seq_len, device=DEVICE)[:TOPK] for _ in range(num_tokens)]
    ).int()
    logical.masked_fill_(torch.rand(logical.shape, device=DEVICE) < 0.3, -1)
    logical[1] = -1
    valid = logical >= 0
    req = torch.arange(num_reqs, device=DEVICE).repeat_interleave(query_len).int()
    safe = logical.clamp_min(0).long()
    rows = stored[block_table[req.long()[:, None], safe // BLOCK].long(), safe % BLOCK]
    q = (torch.randn(num_tokens, heads, DIM, device=DEVICE) * 0.5).bfloat16()
    sinks = torch.randn(heads, device=DEVICE) if with_sink else None

    impl = object.__new__(HYV4AmpereMLASparseImpl)
    impl.sinks, impl.scale, impl.kv_lora_rank = sinks, SCALE, RANK
    impl.kv_cache_dtype = kv_cache_dtype
    impl.topk_indices_buffer = logical.clone()
    metadata = SimpleNamespace(
        block_size=BLOCK,
        block_table=block_table,
        req_id_per_token=req,
        topk_start=0,
        paged_kv_indices=torch.zeros(
            num_tokens * TOPK, dtype=torch.int32, device=DEVICE
        ),
        paged_kv_indptr=torch.zeros(num_tokens + 1, dtype=torch.int32, device=DEVICE),
    )
    assert HYV4AmpereMLASparseMetadata.__dataclass_fields__.keys() >= {
        "paged_kv_indices",
        "paged_kv_indptr",
    }

    out, _ = impl.forward_mqa(q, cache, metadata, None)

    expected = _reference(q, rows, valid, sinks)
    torch.testing.assert_close(out.float(), expected, rtol=2**-6, atol=4e-3)
