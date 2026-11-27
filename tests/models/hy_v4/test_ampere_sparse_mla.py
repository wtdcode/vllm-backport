# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 SM8x sparse MLA: logical top-k rows with holes, read through a paged
BF16 cache, match sink-softmax attention over each row's live keys."""

from types import SimpleNamespace

import pytest
import torch

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


def _reference(q, rows, valid, sinks):
    scores = torch.einsum("thd,tkd->thk", q.float(), rows.float()) * SCALE
    scores.masked_fill_(~valid[:, None], -torch.inf)
    if sinks is not None:
        sink = sinks[None, :, None].expand(q.shape[0], -1, 1)
        scores = torch.cat((scores, sink), -1)
    probs = scores.softmax(-1).nan_to_num(0.0)[..., : rows.shape[1]]
    return torch.einsum("thk,tkd->thd", probs, rows.float()[..., :RANK])


@pytest.mark.parametrize("query_len", [1, 5], ids=["decode", "prefill"])
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("with_sink", [True, False])
def test_forward_mqa_matches_sink_softmax(query_len, heads, with_sink):
    torch.manual_seed(0)
    num_reqs, seq_len, num_blocks = 3, 3000, 160
    num_tokens = num_reqs * query_len
    # Blocks of other layers sit between this cache's blocks in the pool.
    pool = (torch.randn(num_blocks, 2 * BLOCK, DIM, device=DEVICE) * 0.5).bfloat16()
    cache = pool[:, :BLOCK]
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
    rows = cache[block_table[req.long()[:, None], safe // BLOCK].long(), safe % BLOCK]
    q = (torch.randn(num_tokens, heads, DIM, device=DEVICE) * 0.5).bfloat16()
    sinks = torch.randn(heads, device=DEVICE) if with_sink else None

    impl = object.__new__(HYV4AmpereMLASparseImpl)
    impl.sinks, impl.scale, impl.kv_lora_rank = sinks, SCALE, RANK
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
