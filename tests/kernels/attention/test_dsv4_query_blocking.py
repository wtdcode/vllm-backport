# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structure tests for the query-blocked ratio-128 sparse-attention path.

The blocked kernels do not read an index list: they re-derive the rows from
the query positions, because the layers they serve have no indexer and their
"top-k" is the positional identity prefix. That makes the row derivation a
*second* implementation of an index constructor that already exists, so the
tests here pin the two together, on CPU, without needing a GPU:

* :func:`test_blocked_rows_match_combined_indices` -- the prefill derivation
  against a transcription of ``_combine_topk_swa_indices_kernel``.
* :func:`test_blocked_flash_matches_plain_attention` -- the masked/mask-free
  tile loop against plain softmax attention.
* :func:`test_decode_union_reproduces_each_query_list` -- the decode
  derivation, which reconstructs a group's index lists from ``indptr`` alone.
* the gating tests -- every condition that must send a layer back to the
  per-query kernel.
"""

import pytest
import torch

from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_ops
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_query_blocks,
    decode_block_tile,
    decode_query_block_size,
    prefill_query_block_size,
)

# The query-blocked *decode* kernel and its metadata gate are not wired into
# this tree (only the prefill half and the shared gating helpers are), so the
# tests that need them skip instead of failing at import.
_BLOCKED_DECODE_UNAVAILABLE = "query-blocked decode path is not wired in this tree"

COMPRESS_RATIO = 128
WINDOW_SIZE = 128
NEG_LARGE = -3.4028234663852886e38


def combined_rows(
    pos: int,
    top_k: int,
    row_stride: int,
    swa_offset: int,
    batch: int,
    gather_start: int,
) -> list[int]:
    """The rows ``_combine_topk_swa_indices_kernel`` writes for one query.

    Transcribed from ``vllm/models/deepseek_v4/amd/rocm.py`` (the compressed
    half of which is itself the identity prefix that
    ``sparse_mla.py::_build_c128a_topk_metadata_kernel`` stores). Kept as a
    separate transcription on purpose: if either constructor moves, this test
    fails rather than following it.
    """
    topk_len = min((pos + 1) // COMPRESS_RATIO, top_k)
    swa_len = min(pos + 1, WINDOW_SIZE)
    rows = [row_stride * batch + i for i in range(topk_len)]
    base = row_stride * batch + swa_offset + pos - swa_len + 1 - gather_start
    rows += [base + j for j in range(swa_len)]
    return rows


def blocked_geometry(positions: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
    """The per-block geometry ``_sparse_attn_prefill_blocked_kernel`` derives."""
    topk_len = torch.clamp((positions + 1) // COMPRESS_RATIO, max=top_k)
    swa_len = torch.clamp(positions + 1, max=WINDOW_SIZE)
    swa_start = positions + 1 - swa_len
    window_base = int(swa_start.min())
    return dict(
        topk_len=topk_len,
        prefix_len=int(topk_len.max()),
        prefix_shared=int(topk_len.min()),
        window_base=window_base,
        window_len=int((swa_start + swa_len).max()) - window_base,
        window_lo=swa_start - window_base,
        window_hi=swa_start - window_base + swa_len,
    )


def blocked_rows(
    positions: torch.Tensor,
    top_k: int,
    row_stride: int,
    swa_offset: int,
    batch: int,
    gather_start: int,
) -> list[list[int]]:
    """Rows the blocked kernel scores for each query of one block."""
    geo = blocked_geometry(positions, top_k)
    slab = row_stride * batch
    out = []
    for m in range(positions.numel()):
        rows = [slab + k for k in range(geo["prefix_len"]) if k < geo["topk_len"][m]]
        rows += [
            slab + swa_offset + geo["window_base"] + k - gather_start
            for k in range(geo["window_len"])
            if geo["window_lo"][m] <= k < geo["window_hi"][m]
        ]
        out.append(rows)
    return out


@pytest.mark.parametrize("block_m", [2, 4, 8, 16])
@pytest.mark.parametrize(
    "start_pos",
    # Every boundary the derivation can trip on: the first query of a request,
    # the window filling up, a compressed row appearing, a 128-multiple falling
    # inside the block, and a deep chunk.
    [0, 1, 120, 126, 127, 128, 255, 256, 1000, 200_000],
)
@pytest.mark.parametrize("gather_start", [0, 37])
def test_blocked_rows_match_combined_indices(
    block_m: int, start_pos: int, gather_start: int
) -> None:
    top_k = 2048
    row_stride, swa_offset, batch = 300_000, 262_144, 3
    positions = torch.arange(start_pos, start_pos + block_m)
    got = blocked_rows(positions, top_k, row_stride, swa_offset, batch, gather_start)
    for m, pos in enumerate(positions.tolist()):
        expected = combined_rows(
            int(pos), top_k, row_stride, swa_offset, batch, gather_start
        )
        assert got[m] == expected, f"query at position {pos} of a {block_m}-block"


def test_blocked_rows_at_the_topk_cap() -> None:
    """Past the cap the prefixes stop growing, so the block shares every row."""
    top_k = 16
    positions = torch.arange(4096, 4096 + 8)
    got = blocked_rows(positions, top_k, 1 << 20, 1 << 19, 0, 0)
    for m, pos in enumerate(positions.tolist()):
        assert got[m] == combined_rows(int(pos), top_k, 1 << 20, 1 << 19, 0, 0)
    geo = blocked_geometry(positions, top_k)
    assert geo["prefix_len"] == geo["prefix_shared"] == top_k


def blocked_flash(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    top_k: int,
    swa_offset: int,
    gather_start: int,
    scale: float,
    block_k: int,
) -> torch.Tensor:
    """Replay of the blocked kernel's tile loop, including its two fast paths.

    Same tiling, same mask predicates, same order of accumulation, in torch.
    """
    geo = blocked_geometry(positions, top_k)
    rows = q.shape[0]
    m_i = torch.full((rows,), NEG_LARGE, dtype=torch.float32)
    l_i = torch.zeros(rows, dtype=torch.float32)
    acc = torch.zeros(rows, kv.shape[1], dtype=torch.float32)

    def step(scores: torch.Tensor, keep: torch.Tensor, tile: torch.Tensor) -> None:
        nonlocal m_i, l_i, acc
        scores = torch.where(keep, scores, torch.tensor(NEG_LARGE))
        m_new = torch.maximum(m_i, scores.max(dim=1).values)
        alpha = torch.exp(m_i - m_new)
        p = torch.where(keep, torch.exp(scores - m_new[:, None]), torch.zeros(()))
        l_i = l_i * alpha + p.sum(dim=1)
        acc = acc * alpha[:, None] + p @ tile
        m_i = m_new

    k_off = torch.arange(block_k)
    for k_start in range(0, geo["prefix_len"], block_k):
        k_pos = k_start + k_off
        tile = kv[torch.where(k_pos < geo["prefix_len"], k_pos, torch.zeros(()).long())]
        scores = (q @ tile.T) * scale
        if k_start + block_k <= geo["prefix_shared"]:
            keep = torch.ones_like(scores, dtype=torch.bool)
        else:
            keep = k_pos[None, :] < geo["topk_len"][:, None]
        step(scores, keep, tile)

    lo_max = int(geo["window_lo"].max())
    hi_min = int(geo["window_hi"].min())
    for k_start in range(0, geo["window_len"], block_k):
        k_pos = k_start + k_off
        row = (
            swa_offset
            + geo["window_base"]
            - gather_start
            + torch.where(k_pos < geo["window_len"], k_pos, torch.zeros(()).long())
        )
        tile = kv[row]
        scores = (q @ tile.T) * scale
        if k_start >= lo_max and k_start + block_k <= hi_min:
            keep = torch.ones_like(scores, dtype=torch.bool)
        else:
            keep = (k_pos[None, :] >= geo["window_lo"][:, None]) & (
                k_pos[None, :] < geo["window_hi"][:, None]
            )
        step(scores, keep, tile)

    denom = torch.clamp(l_i, min=1e-30)
    return torch.where((l_i > 0)[:, None], acc / denom[:, None], torch.zeros(()))


@pytest.mark.parametrize("block_m", [4, 8])
@pytest.mark.parametrize("start_pos", [0, 100, 126, 254, 640])
@pytest.mark.parametrize("block_k", [8, 32])
def test_blocked_flash_matches_plain_attention(
    block_m: int, start_pos: int, block_k: int
) -> None:
    """The tiled loop, masks and all, equals softmax over each query's own rows.

    ``start_pos=0`` is the case that made the mask on ``p`` load-bearing: the
    first query of a request has an empty compressed prefix while a later one
    in the same block does not, so its whole first tile is masked and its
    running max is still -inf.
    """
    torch.manual_seed(block_m * 1000 + start_pos + block_k)
    top_k, swa_offset, gather_start, dim = 8, 64, 0, 16
    positions = torch.arange(start_pos, start_pos + block_m)
    # SWA rows sit at `swa_offset + position - gather_start`, so the workspace
    # has to reach the block's last position.
    kv = torch.randn(swa_offset + start_pos + block_m + 1, dim, dtype=torch.float32)
    q = torch.randn(block_m, dim, dtype=torch.float32)
    scale = dim**-0.5

    got = blocked_flash(
        q, kv, positions, top_k, swa_offset, gather_start, scale, block_k
    )

    for m, pos in enumerate(positions.tolist()):
        rows = combined_rows(int(pos), top_k, 0, swa_offset, 0, gather_start)
        scores = (q[m] @ kv[rows].T) * scale
        expected = torch.softmax(scores, dim=0) @ kv[rows]
        torch.testing.assert_close(got[m], expected, rtol=1e-5, atol=1e-5)


def decode_group_geometry(
    main_lens: list[int], extra_lens: list[int]
) -> dict[str, int | list[int]]:
    """The union geometry ``_sparse_attn_decode_partial_blocked_kernel`` derives.

    Everything comes from the per-query segment lengths, i.e. from ``indptr``:
    the compressed lists nest, and the windows slide by one position per query.
    """
    group = len(main_lens)
    front = min(max(group - 1 + main_lens[0] - main_lens[-1], 0), main_lens[0])
    main_union = front + main_lens[-1]
    main_hi = [min(m + main_lens[0], main_union) for m in range(group)]
    main_lo = [max(main_hi[m] - main_lens[m], 0) for m in range(group)]
    extra_union = extra_lens[-1]
    return dict(
        front=front,
        main_union=main_union,
        main_lo=main_lo,
        main_hi=main_hi,
        extra_union=extra_union,
        extra_hi=[min(extra_lens[m], extra_union) for m in range(group)],
    )


@pytest.mark.parametrize("start_pos", [200, 127, 128, 1023, 1024, 200_000])
@pytest.mark.parametrize("group", [2, 6, 8])
def test_decode_union_reproduces_each_query_list(start_pos: int, group: int) -> None:
    """Reading the union at a query's own range returns that query's own rows.

    Slots are a per-request function of position (block-table lookup), so the
    test builds them that way and then checks that the two-source union
    reconstruction hands each query back exactly the list the per-query kernel
    would have walked.
    """
    positions = list(range(start_pos, start_pos + group))
    slot_of = lambda p: 7919 * (p // 64) + (p % 64)  # noqa: E731 - a paged map

    main = [
        [slot_of(p) for p in range(pos + 1 - min(pos + 1, WINDOW_SIZE), pos + 1)]
        for pos in positions
    ]
    extra = [
        [slot_of(i) for i in range(min((pos + 1) // COMPRESS_RATIO, 2048))]
        for pos in positions
    ]
    geo = decode_group_geometry([len(x) for x in main], [len(x) for x in extra])

    # The union as the kernel materialises it: coordinates below `front` come
    # from the first query's list, the rest from the last query's.
    union = [
        main[0][j] if j < geo["front"] else main[-1][j - geo["front"]]
        for j in range(geo["main_union"])
    ]
    for m in range(group):
        assert union[geo["main_lo"][m] : geo["main_hi"][m]] == main[m], (
            f"SWA window of query {m} at position {positions[m]}"
        )
        assert extra[-1][: geo["extra_hi"][m]] == extra[m], (
            f"compressed prefix of query {m} at position {positions[m]}"
        )


def test_decode_geometry_survives_an_all_padding_group() -> None:
    """A captured graph's spare rows carry zero lengths and must read nothing."""
    geo = decode_group_geometry([0] * 6, [0] * 6)
    assert geo["front"] == 0
    assert geo["main_union"] == 0
    assert geo["extra_union"] == 0
    assert geo["main_lo"] == geo["main_hi"] == [0] * 6


def test_query_blocks_never_cross_a_request() -> None:
    qsl = torch.tensor([0, 10, 14, 30], dtype=torch.int32)
    block_req, block_qstart = build_query_blocks(qsl, 4, torch.device("cpu"))
    assert block_req.tolist() == [0, 0, 0, 1, 2, 2, 2, 2]
    assert block_qstart.tolist() == [0, 4, 8, 10, 14, 18, 22, 26]
    # Every block starts inside its own request and no query is covered twice.
    starts = qsl[:-1].tolist()
    ends = qsl[1:].tolist()
    covered = []
    for req, start in zip(block_req.tolist(), block_qstart.tolist()):
        assert starts[req] <= start < ends[req]
        covered += [t for t in range(start, start + 4) if t < ends[req]]
    assert covered == list(range(int(qsl[-1])))


def test_query_blocks_are_offset_by_the_chunks_first_query() -> None:
    """Chunk slices are absolute; the kernel indexes q from the chunk's start."""
    qsl = torch.tensor([100, 106, 118], dtype=torch.int32)
    _, block_qstart = build_query_blocks(qsl, 4, torch.device("cpu"))
    assert block_qstart.tolist() == [0, 4, 6, 10, 14]


@pytest.mark.parametrize(
    "flag,expected",
    [(0, 0), (-1, 8), (1, 1), (4, 4), (6, 8), (16, 16)],
)
def test_prefill_block_size_reads_the_flag(
    monkeypatch: pytest.MonkeyPatch, flag: int, expected: int
) -> None:
    monkeypatch.setenv("VLLM_SPARSE_DENSE_QUERY_BLOCK", str(flag))
    prefill_query_block_size.cache_clear()
    if flag == -1:
        # The auto default declines on parts without A100-class shared
        # memory (the fixed KV/index tiles alone exceed sm86/sm89's ~99 KB).
        smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        expected = 8 if smem >= 160 * 1024 else 0
    assert prefill_query_block_size(8, 512) == expected
    prefill_query_block_size.cache_clear()


def test_prefill_block_size_declines_shapes_the_kernel_cannot_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SPARSE_DENSE_QUERY_BLOCK", "8")
    prefill_query_block_size.cache_clear()
    assert prefill_query_block_size(8, 512) == 8
    assert prefill_query_block_size(5, 512) == 0  # heads are not a whole tile
    assert prefill_query_block_size(8, 500) == 0  # head_dim is not a power of two
    prefill_query_block_size.cache_clear()


@pytest.mark.parametrize(
    # The decode tile defaults OFF: it measured 1.2-1.9x slower than the
    # per-query kernel at every residency (see `decode_query_block_size`), so
    # -1 keeps the old path and only an explicit width turns it on.
    "flag,group,expected",
    [(0, 6, 0), (-1, 6, 0), (-1, 1, 0), (6, 6, 8), (8, 6, 8), (2, 6, 2)],
)
def test_decode_block_size_reads_the_flag(
    monkeypatch: pytest.MonkeyPatch, flag: int, group: int, expected: int
) -> None:
    monkeypatch.setenv("VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE", str(flag))
    decode_query_block_size.cache_clear()
    assert decode_query_block_size(group) == expected
    decode_query_block_size.cache_clear()


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a GPU"
)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _ragged_from_rows(rows: list[list[int]], device: torch.device):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    width = max(len(r) for r in rows)
    dense = torch.full((len(rows), width), -1, dtype=torch.int32, device=device)
    for i, r in enumerate(rows):
        dense[i, : len(r)] = torch.tensor(r, dtype=torch.int32, device=device)
    lens = torch.tensor([len(r) for r in rows], dtype=torch.int32, device=device)
    return build_ragged_indices_from_dense(dense, lens), lens


@requires_cuda
@pytest.mark.parametrize("block_m", [1, 2, 4, 8])
@pytest.mark.parametrize("start_pos", [0, 126, 1000])
def test_blocked_prefill_kernel_matches_the_per_query_kernel(
    block_m: int, start_pos: int
) -> None:
    """Both kernels, same rows, on a two-request chunk with a ragged tail.

    The comparison is to the kernel this replaces rather than to a torch
    reference, because that is the pair the serving gate cares about; the
    tolerance is bf16 flash-rescaling noise, since the blocked kernel tiles the
    two segments separately and so accumulates in a different order.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _sparse_attn_prefill_blocked_kernel,
        _sparse_attn_prefill_ragged_kernel,
        build_query_blocks,
    )

    device = torch.device("cuda")
    torch.manual_seed(start_pos + block_m)
    heads, dim, block_h, block_k = 8, 64, 8, 16
    top_k, swa_offset, row_stride = 64, 4096, 8192
    query_lens = [7, 5]
    gather_starts = [0, 3]

    positions, rows = [], []
    for req, qlen in enumerate(query_lens):
        for t in range(qlen):
            pos = start_pos + t
            positions.append(pos)
            rows.append(
                combined_rows(
                    pos, top_k, row_stride, swa_offset, req, gather_starts[req]
                )
            )
    num_queries = len(rows)
    (indices, indptr), _ = _ragged_from_rows(rows, device)

    q = torch.randn(num_queries, heads, dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(2 * row_stride, dim, dtype=torch.bfloat16, device=device)
    sink = torch.randn(heads, dtype=torch.float32, device=device)
    scale = dim**-0.5
    expected = torch.empty_like(q)
    got = torch.empty_like(q)

    _sparse_attn_prefill_ragged_kernel[(num_queries, heads // block_h)](
        q, kv, indices, indptr, sink, expected,
        q.stride(0), q.stride(1), q.stride(2), kv.stride(0), kv.stride(1),
        expected.stride(0), expected.stride(1), expected.stride(2),
        heads, dim, kv.shape[0], scale,
        HAS_ATTN_SINK=True, IS_MXFP4=False, OUT_DV=dim,
        BLOCK_H=block_h, BLOCK_D=dim, BLOCK_K=block_k,
        EXACT_TILE=True, num_warps=4,
    )  # fmt: skip

    qsl = torch.tensor([0, query_lens[0], num_queries], dtype=torch.int32)
    block_req, block_qstart = build_query_blocks(qsl, block_m, device)
    seq_lens = torch.tensor(
        [start_pos + query_lens[0], start_pos + query_lens[1]],
        dtype=torch.int32,
        device=device,
    )
    gather_lens = seq_lens - torch.tensor(
        gather_starts, dtype=torch.int32, device=device
    )
    _sparse_attn_prefill_blocked_kernel[(block_req.numel(), heads // block_h)](
        q, kv, block_req, block_qstart, qsl.to(device), seq_lens, gather_lens,
        sink, got,
        q.stride(0), q.stride(1), q.stride(2), kv.stride(0), kv.stride(1),
        got.stride(0), got.stride(1), got.stride(2),
        top_k, row_stride, swa_offset, scale,
        HAS_ATTN_SINK=True, COMPRESS_RATIO=COMPRESS_RATIO,
        WINDOW_SIZE=WINDOW_SIZE, BLOCK_M=block_m, BLOCK_H=block_h,
        BLOCK_D=dim, BLOCK_K=block_k, num_warps=4,
    )  # fmt: skip

    torch.testing.assert_close(
        got.to(torch.float32), expected.to(torch.float32), rtol=2e-2, atol=2e-2
    )


@requires_cuda
@pytest.mark.parametrize("group", [2, 6])
@pytest.mark.parametrize("num_splits", [1, 4])
def test_blocked_decode_kernel_matches_the_per_query_kernel(
    group: int, num_splits: int
) -> None:
    """Same ragged buffers into both partial kernels, same reduce after them.

    The blocked kernel derives its union from ``indptr``, so feeding both the
    identical index lists is what makes this a test of that derivation.

    ``group=6`` runs on an 8-row tile with two padding rows, which is what
    production does at ``next_n=6``: ``BLOCK_M`` indexes a ``tl.arange`` and
    must be a power of two, while ``group_size`` is the real query count.
    """
    smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    block_m = 1 << (group - 1).bit_length()
    if block_m * 17920 > smem:
        pytest.skip(
            f"blocked tile BLOCK_M={block_m} needs ~{block_m * 17920} B smem, "
            f"device limit {smem} B (sm86/sm89)"
        )

    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )
    from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _sparse_attn_decode_partial_kernel,
        _sparse_attn_decode_reduce_kernel,
    )

    _sparse_attn_decode_partial_blocked_kernel = getattr(
        sparse_ops, "_sparse_attn_decode_partial_blocked_kernel", None
    )
    if _sparse_attn_decode_partial_blocked_kernel is None:
        pytest.skip(_BLOCKED_DECODE_UNAVAILABLE)

    device = torch.device("cuda")
    torch.manual_seed(group * 10 + num_splits)
    heads, nope, rope, dim = 8, 448, 64, 512
    block_h, block_k, block_size = 8, 32, 64
    batch, depth, window = 3, 4096, WINDOW_SIZE
    num_queries = batch * group

    cache_rows = 8192
    k = torch.randn(cache_rows, dim, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(
        (cache_rows // block_size, block_size, 584), dtype=torch.uint8, device=device
    )
    quantize_and_insert_k_cache(
        k,
        cache,
        torch.arange(cache_rows, device=device),
        block_size=block_size,
        use_fnuz=False,
    )

    # Slots are a per-request function of position, which is what makes the
    # prefixes nest and the windows slide.
    slot = lambda req, i: (req * 2731 + i * 13) % cache_rows  # noqa: E731
    main_rows, extra_rows = [], []
    for req in range(batch):
        for t in range(group):
            pos = depth + t
            swa_len = min(pos + 1, window)
            main_rows.append([slot(req, p) for p in range(pos + 1 - swa_len, pos + 1)])
            extra_rows.append(
                [slot(req + 64, i) for i in range((pos + 1) // COMPRESS_RATIO)]
            )
    (main_indices, main_indptr), _ = _ragged_from_rows(main_rows, device)
    (extra_indices, extra_indptr), _ = _ragged_from_rows(extra_rows, device)

    q = torch.randn(num_queries, heads, dim, dtype=torch.bfloat16, device=device)
    sink = torch.randn(heads, dtype=torch.float32, device=device)
    lut = get_e4m3fn_bf16_lut(device)
    scale = dim**-0.5

    def run(blocked: bool) -> torch.Tensor:
        part_m = torch.empty(
            (num_queries, num_splits, heads), dtype=torch.float32, device=device
        )
        part_l = torch.empty_like(part_m)
        part_acc = torch.empty(
            (num_queries, num_splits, heads, dim), dtype=torch.float32, device=device
        )
        out = torch.empty_like(q)
        kernel = (
            _sparse_attn_decode_partial_blocked_kernel
            if blocked
            else _sparse_attn_decode_partial_kernel
        )
        extra = {"BLOCK_M": _next_pow2(group)} if blocked else {}
        kernel[((batch if blocked else num_queries), num_splits, heads // block_h)](
            q, cache, main_indices, main_indptr, cache, extra_indices, extra_indptr,
            part_m, part_l, part_acc, lut,
            q.stride(0), q.stride(1), cache.stride(0), cache.stride(0),
            part_m.stride(0), part_m.stride(1),
            part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
            cache_rows, cache_rows, block_size, block_size,
            scale, group if blocked else heads,
            HAS_EXTRA=True, NOPE_DIM=nope, NOPE_BLOCK=512, ROPE_DIM=rope,
            IS_FNUZ_MAIN=False, IS_FNUZ_EXTRA=False,
            BLOCK_H=block_h, BLOCK_K=block_k, NUM_SPLITS=num_splits,
            NUM_STAGES=1, num_warps=8, **extra,
        )  # fmt: skip
        _sparse_attn_decode_reduce_kernel[(num_queries, heads)](
            part_m, part_l, part_acc, sink, out,
            out.stride(0), out.stride(1), part_m.stride(0), part_m.stride(1),
            part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
            heads,
            HAS_ATTN_SINK=True, COMB_DIM=dim, BLOCK_H=1,
            NUM_SPLITS=num_splits, SPLITS_PAD=num_splits, num_warps=4,
        )  # fmt: skip
        return out

    torch.testing.assert_close(
        run(True).to(torch.float32), run(False).to(torch.float32),
        rtol=2e-2, atol=2e-2,
    )  # fmt: skip


def test_decode_block_tile_declines_what_the_kernel_cannot_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE", "6")
    decode_query_block_size.cache_clear()
    # 27 requests x next_n 6, 8 heads on an 8-wide head tile: the served shape.
    # On low-shared-memory parts (sm86/sm89, ~99 KB vs A100's 163 KB) the
    # 8-row tile cannot compile, so the chooser declines it there instead.
    smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    expected = 8 if smem >= 8 * 17920 else 0
    assert decode_block_tile(6, 162, 8, 8) == expected
    # A group is never split across CTAs, so a forced tile below next_n
    # declines rather than halving the group.
    monkeypatch.setenv("VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE", "4")
    decode_query_block_size.cache_clear()
    assert decode_block_tile(6, 162, 8, 8) == 0
    monkeypatch.setenv("VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE", "6")
    decode_query_block_size.cache_clear()
    assert decode_block_tile(6, 160, 8, 8) == 0  # rows are not whole groups
    assert decode_block_tile(6, 162, 5, 8) == 0  # heads are not a whole tile
    assert decode_block_tile(1, 27, 8, 8) == 0  # nothing to block
    assert decode_block_tile(0, 27, 8, 8) == 0
    decode_query_block_size.cache_clear()


def test_uniform_group_size_gates_the_decode_block() -> None:
    try:
        from vllm.models.deepseek_v4.amd.rocm import uniform_decode_group_size
    except ImportError:
        pytest.skip(_BLOCKED_DECODE_UNAVAILABLE)
    uniform = torch.tensor([0, 6, 12, 18], dtype=torch.int32)
    assert uniform_decode_group_size(True, 3, 18, uniform) == 6
    # The DSpark draft step is non-causal: its per-token SWA lists are not
    # slices of one sliding run.
    assert uniform_decode_group_size(False, 3, 18, uniform) == 0
    # A ragged batch would put a block across two requests.
    ragged = torch.tensor([0, 6, 7, 13], dtype=torch.int32)
    assert uniform_decode_group_size(True, 3, 13, ragged) == 0
    # One query per request is the un-speculated case: nothing to block.
    single = torch.tensor([0, 1, 2], dtype=torch.int32)
    assert uniform_decode_group_size(True, 2, 2, single) == 0
    assert uniform_decode_group_size(True, 0, 0, uniform) == 0
    assert uniform_decode_group_size(True, 3, 18, None) == 0
