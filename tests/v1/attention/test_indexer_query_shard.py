# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rows an indexer Q path owns under ShardQ (the TP prefill query shard)."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerPrefillMetadata
from vllm.v1.attention.ops import shardq
from vllm.v1.attention.ops.shardq import ShardQSplit, zigzag_split


def test_q_row_ranges_is_none_without_shardq():
    assert DeepseekV32IndexerPrefillMetadata(chunks=[]).q_row_ranges is None


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("num_decode_tokens", [0, 6])
def test_q_row_ranges_cover_exactly_the_gathered_rows(
    world, num_decode_tokens, monkeypatch
):
    """The ranges are the decode rows plus this rank's ShardQ tokens, so a Q path
    that computes only them fills every row the indexer op gathers."""
    query_lens = np.array([2048, 1500, 37])
    token_ids, per_rank = zigzag_split(query_lens, world)
    split = ShardQSplit(num_decode_tokens, token_ids, per_rank)
    owned = np.zeros(num_decode_tokens + query_lens.sum(), dtype=np.int64)
    for rank in range(world):
        monkeypatch.setattr(
            shardq,
            "get_tp_group",
            lambda rank=rank: SimpleNamespace(rank_in_group=rank),
        )
        prefill = DeepseekV32IndexerPrefillMetadata(chunks=[], shardq_split=split)
        ranges = prefill.q_row_ranges
        assert ranges is not None
        assert all(lo < hi for lo, hi in ranges)
        assert all(a[1] < b[0] for a, b in zip(ranges, ranges[1:]))
        rows = np.concatenate([np.arange(lo, hi) for lo, hi in ranges])
        expected = np.concatenate(
            (np.arange(num_decode_tokens), split.local_token_ids + num_decode_tokens)
        )
        assert np.array_equal(rows, expected)
        owned[rows] += 1
    # Decode rows are scored by every rank, prefill rows by exactly one.
    assert (owned[:num_decode_tokens] == world).all()
    assert (owned[num_decode_tokens:] == 1).all()
