# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmarks for the DeepSeek-V4-Flash SM8x kernels.

Shapes default to DSv4-Flash-0731 at TP=8 (8 local heads, head_dim 512,
hidden 4096, hc_mult 4), the configuration whose profile motivated the tuning
knobs each sweep exposes. Sweeps launch the kernels directly so the tuning
parameters (split count, head block, worker count) are explicit rather than
whatever the production heuristic picks.

Examples:
    python benchmarks/kernels/benchmark_dsv4_sm80.py --kernel sparse-decode
    python benchmarks/kernels/benchmark_dsv4_sm80.py --kernel all

"""

import argparse
import itertools
from collections.abc import Callable
from functools import partial

import torch

from vllm.triton_utils import tl, triton

# Every value below that also exists in the checkpoint is named after its
# config.json key, because guessing one has already produced a wrong conclusion
# here: a placeholder sinkhorn count of 3 against the real 20 changed the mHC pre
# per-launch cost from 5.2 to 8.3 us *and* changed which component dominated.
# DeepSeek-V4-Flash-0731 config.json, verified 2026-08-01.
CFG_HEAD_DIM = 512  # head_dim
CFG_QK_ROPE_HEAD_DIM = 64  # qk_rope_head_dim
CFG_HIDDEN_SIZE = 4096  # hidden_size
CFG_NUM_ATTENTION_HEADS = 64  # num_attention_heads
CFG_NUM_HIDDEN_LAYERS = 43  # num_hidden_layers
CFG_N_ROUTED_EXPERTS = 256  # n_routed_experts
CFG_NUM_EXPERTS_PER_TOK = 6  # num_experts_per_tok
CFG_HC_MULT = 4  # hc_mult
CFG_HC_SINKHORN_ITERS = 20  # hc_sinkhorn_iters
CFG_SLIDING_WINDOW = 128  # sliding_window
CFG_INDEX_TOPK = 512  # index_topk
CFG_INDEX_N_HEADS = 64  # index_n_heads (replicated, not TP-sharded)
CFG_INDEX_HEAD_DIM = 128  # index_head_dim
CFG_Q_LORA_RANK = 1024  # q_lora_rank
CFG_O_LORA_RANK = 1024  # o_lora_rank
CFG_O_GROUPS = 8  # o_groups
CFG_MOE_INTERMEDIATE_SIZE = 2048  # moe_intermediate_size (n_shared_experts=1)
CFG_WEIGHT_BLOCK_SIZE = 128  # quantization_config.weight_block_size = [128, 128]

TP_SIZE = 8

# compress_ratios in config.json holds 21 fours: only those layers carry an
# indexer, so only they have three input GEMMs to merge.
_ATTN_INPUT_LAYERS = 21

# MLA head geometry. The nope width is the remainder, not an independent number.
HEAD_DIM = CFG_HEAD_DIM
ROPE_DIM = CFG_QK_ROPE_HEAD_DIM
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448
SCALE = HEAD_DIM**-0.5

# fp8_ds_mla cache entry: nope as fp8, rope as bf16, one ue8m0 scale per
# quantization block (7 real + 1 pad).
QUANT_BLOCK = 64
SCALE_DIM = 8
CACHE_TOKEN_BYTES = NOPE_DIM + ROPE_DIM * 2  # 576
CACHE_ENTRY_BYTES = CACHE_TOKEN_BYTES + SCALE_DIM  # 584
CACHE_BLOCK_SIZE = 64  # vLLM paged block size, not a checkpoint value

NUM_HEADS = CFG_NUM_ATTENTION_HEADS // TP_SIZE  # 8 local heads
HIDDEN_SIZE = CFG_HIDDEN_SIZE
HC_MULT = CFG_HC_MULT
HC_MULT3 = HC_MULT * (2 + HC_MULT)  # 24 prenorm GEMM outputs
SINKHORN_ITERS = CFG_HC_SINKHORN_ITERS

# Decode walks sliding_window SWA tokens plus index_topk compressed tokens per
# query, independent of context length.
SWA_LEN = CFG_SLIDING_WINDOW
TOPK_LEN = CFG_INDEX_TOPK

# The ratio-128 layers' compression ratio: their index list is the identity
# prefix (pos+1)//128 of the compressed cache, not a selection.
_PREFILL_COMPRESS_RATIO = 128

# mHC pre runs twice per layer (attention and FFN) plus one broadcast variant for
# the first layer, which is what per-step totals below are scaled by.
MHC_PRE_LAUNCHES_PER_STEP = 2 * CFG_NUM_HIDDEN_LAYERS + 1  # 87


def _time_us(fn: Callable[[], None]) -> float:
    """Median latency in us, or NaN if the configuration does not compile."""
    try:
        fn()
        torch.accelerator.synchronize()
        return triton.testing.do_bench_cudagraph(fn, rep=200) * 1000.0
    except Exception as exc:  # noqa: BLE001 - report and keep sweeping
        print(f"  skipped ({type(exc).__name__}: {exc})")
        return float("nan")


def _fmt(value: float, fmt: str = ".1f") -> str:
    return "-" if value != value else format(value, fmt)


def _print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(header[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(header))
    ]
    print(f"\n=== {title} ===")
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    for row in rows:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))


def _make_ds_mla_cache(
    num_tokens: int, block_size: int, use_fnuz: bool, device: torch.device
) -> torch.Tensor:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    num_blocks = max(1, triton.cdiv(num_tokens, block_size))
    cache = torch.zeros(
        (num_blocks, block_size, CACHE_ENTRY_BYTES), dtype=torch.uint8, device=device
    )
    k = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device=device)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    quantize_and_insert_k_cache(
        k, cache, slots, block_size=block_size, use_fnuz=use_fnuz
    )
    return cache


def _ragged_indices(
    num_queries: int,
    seg_len: int,
    num_rows: int,
    scattered: bool,
    device: torch.device,
    live_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged (indices, indptr) with ``seg_len`` slots per query.

    ``scattered`` models the compressed top-k gather (random rows); contiguous
    models the SWA window.

    ``live_rows`` separates *how much data the sequence owns* from *how far
    apart it sits*. Serving gathers ctx/compress_ratio rows out of a cache pool
    two orders of magnitude larger, in `block_size`-row pages handed out by a
    free list, so the bytes are L2-sized while the addresses are not. Passing
    live_rows < num_rows reproduces that; leaving it 0 keeps the old compact
    behaviour where the pool IS the sequence.
    """
    if scattered:
        if live_rows and live_rows < num_rows:
            page = CACHE_BLOCK_SIZE
            n_pages = max(1, triton.cdiv(live_rows, page))
            pool_pages = max(n_pages, num_rows // page)
            pages = torch.randperm(pool_pages, device=device)[:n_pages]
            slot = torch.randint(
                0, page, (num_queries * seg_len,), dtype=torch.int64, device=device
            )
            pick = torch.randint(
                0, n_pages, (num_queries * seg_len,), dtype=torch.int64, device=device
            )
            indices = (pages[pick] * page + slot).clamp_(max=num_rows - 1)
            indices = indices.to(torch.int32)
        else:
            indices = torch.randint(
                0, num_rows, (num_queries * seg_len,), dtype=torch.int32, device=device
            )
    else:
        starts = torch.randint(
            0, max(1, num_rows - seg_len), (num_queries, 1), device=device
        )
        indices = (starts + torch.arange(seg_len, device=device)).to(torch.int32)
        indices = indices.reshape(-1)
    indptr = torch.arange(
        0, num_queries * seg_len + 1, seg_len, dtype=torch.int32, device=device
    )
    return indices.contiguous(), indptr


# ---------------------------------------------------------------------------
# Sparse MLA decode: single-pass vs split-K partial+reduce
# ---------------------------------------------------------------------------


def _decode_operating_point(ctx: int, compress_ratio: int = 4) -> dict:
    """Segment lengths and gather-pool size for a decode step at ``ctx`` tokens.

    Every quantity here follows from the context length, which is the point:
    a fixed pool measures one spot on a context-dependent curve, and a
    node-granularity A/B put the live kernel on BOTH sides of the old fixed
    pool (17% faster than the bench at ~94 tokens, 20% slower at 107k). The
    bench cannot guide tuning in either direction until the operating point is
    an input.

    ``topk_len``/``swa_len`` are the *true* walked lengths, which is what the
    kernel does via indptr. They are NOT what production's split heuristic
    sees -- see `bench_sparse_decode`.
    """
    compressed = max(1, ctx // compress_ratio)
    return dict(
        ctx=ctx,
        topk_len=min(TOPK_LEN, compressed),
        # Rows the sequence owns; the pool it sits in is a separate knob,
        # because bytes and address spread are separate effects.
        topk_rows=max(1, compressed),
        swa_len=min(SWA_LEN, ctx),
    )


def _decode_inputs(
    batch: int,
    splits: list[int],
    device: torch.device,
    topk_len: int = TOPK_LEN,
    topk_rows: int = 26875,
    swa_rows: int = 4096,
    swa_len: int = SWA_LEN,
    pool_rows: int = 0,
) -> dict:
    """Decode inputs matching the live caller: SWA is the *main* segment
    (contiguous window) and the compressed top-k is *extra* (scattered) --
    see `rocm_sparse_attn_decode`, which passes `main_cache=swa_k_cache,
    main_lengths=swa_lens` and `extra_cache=kv_cache, extra_lengths=topk_lens`.

    ``topk_rows`` is the row count the scattered gather indexes into, and it
    is the single most misleading knob here. It used to be hardcoded at
    128*1024 rows = 76.6 MB of fp8_ds_mla cache against an A100's 40 MB L2,
    so every measurement was taken in a DRAM-bound regime the server never
    enters: serving gathers out of ctx/compress_ratio rows, which is 4.8 MB
    at 32k and 19.1 MB at 128k -- L2-resident throughout. That inflated the
    kernel roughly 2.6x against the live 17.30 us and is why bench-derived
    per-call numbers disagreed with the trace.
    """
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut

    is_fnuz = current_platform.is_fp8_fnuz()
    # The cache the gather indexes into is the pool when one is given, and the
    # sequence's own rows otherwise; `live_rows` keeps the touched bytes the
    # same either way, so the two arms isolate address spread from volume.
    extra_cache_rows = max(topk_rows, pool_rows)
    main_indices, main_indptr = _ragged_indices(batch, swa_len, swa_rows, False, device)
    extra_indices, extra_indptr = _ragged_indices(
        batch, topk_len, extra_cache_rows, True, device, live_rows=topk_rows
    )
    q = torch.randn(batch, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    part = {
        s: (
            torch.empty((batch, s, NUM_HEADS), dtype=torch.float32, device=device),
            torch.empty((batch, s, NUM_HEADS), dtype=torch.float32, device=device),
            torch.empty(
                (batch, s, NUM_HEADS, HEAD_DIM), dtype=torch.float32, device=device
            ),
        )
        for s in splits
    }
    return dict(
        q=q,
        out=torch.empty_like(q),
        main_cache=_make_ds_mla_cache(swa_rows, CACHE_BLOCK_SIZE, is_fnuz, device),
        extra_cache=_make_ds_mla_cache(
            extra_cache_rows, CACHE_BLOCK_SIZE, False, device
        ),
        main_indices=main_indices,
        main_indptr=main_indptr,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
        attn_sink=torch.randn(NUM_HEADS, dtype=torch.float32, device=device),
        fp8_lut=get_e4m3fn_bf16_lut(device),
        is_fnuz=is_fnuz,
        part=part,
    )


def _launch_single_pass(inp: dict, block_h: int, block_k: int) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _sparse_attn_decode_ragged_kernel,
    )

    q, out = inp["q"], inp["out"]
    main_cache, extra_cache = inp["main_cache"], inp["extra_cache"]
    _sparse_attn_decode_ragged_kernel[(q.shape[0], triton.cdiv(NUM_HEADS, block_h))](
        q,
        main_cache,
        inp["main_indices"],
        inp["main_indptr"],
        extra_cache,
        inp["extra_indices"],
        inp["extra_indptr"],
        inp["attn_sink"],
        inp["fp8_lut"],
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        SCALE,
        NUM_HEADS,
        HAS_ATTN_SINK=True,
        HAS_EXTRA=True,
        NOPE_DIM=NOPE_DIM,
        NOPE_BLOCK=triton.next_power_of_2(NOPE_DIM),
        ROPE_DIM=ROPE_DIM,
        IS_FNUZ_MAIN=inp["is_fnuz"],
        IS_FNUZ_EXTRA=False,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=8,
    )


def _launch_split_k(
    inp: dict, block_h: int, block_k: int, num_splits: int, num_warps: int
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _sparse_attn_decode_partial_kernel,
        _sparse_attn_decode_reduce_kernel,
    )

    q, out = inp["q"], inp["out"]
    part_m, part_l, part_acc = inp["part"][num_splits]
    main_cache, extra_cache = inp["main_cache"], inp["extra_cache"]
    _sparse_attn_decode_partial_kernel[
        (q.shape[0], num_splits, triton.cdiv(NUM_HEADS, block_h))
    ](
        q,
        main_cache,
        inp["main_indices"],
        inp["main_indptr"],
        extra_cache,
        inp["extra_indices"],
        inp["extra_indptr"],
        part_m,
        part_l,
        part_acc,
        inp["fp8_lut"],
        q.stride(0),
        q.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        SCALE,
        NUM_HEADS,
        HAS_EXTRA=True,
        NOPE_DIM=NOPE_DIM,
        NOPE_BLOCK=triton.next_power_of_2(NOPE_DIM),
        ROPE_DIM=ROPE_DIM,
        IS_FNUZ_MAIN=inp["is_fnuz"],
        IS_FNUZ_EXTRA=False,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        NUM_STAGES=1,
        num_warps=num_warps,
    )
    _sparse_attn_decode_reduce_kernel[(q.shape[0], NUM_HEADS)](
        part_m,
        part_l,
        part_acc,
        inp["attn_sink"],
        out,
        out.stride(0),
        out.stride(1),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        NUM_HEADS,
        HAS_ATTN_SINK=True,
        COMB_DIM=HEAD_DIM,
        BLOCK_H=1,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=triton.next_power_of_2(num_splits),
        num_warps=4,
    )


def bench_sparse_decode(
    batches: list[int],
    block_hs: list[int],
    splits: list[int],
    warps: list[int],
    device: torch.device,
    topk_len: int = TOPK_LEN,
    topk_rows: int = 26875,
    swa_len: int = SWA_LEN,
    pool_rows: int = 0,
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _decode_num_splits

    rows = []
    for batch in batches:
        for block_h in block_hs:
            heads_blocks = triton.cdiv(NUM_HEADS, block_h)
            # Serving picks the split count from this same heuristic, but it
            # feeds it `main_indices.numel() / num_queries`, and both ragged
            # buffers are allocated at their DENSE width (rocm.py:262 sizes the
            # top-k pack at num_tokens * topk; :299 slices the SWA graph buffer
            # at num_rows * window). So the heuristic always sees 128 and 512
            # no matter how short the context is -- which is why the split
            # count is context-independent in the traces. Mark that arm live,
            # and mark the split the TRUE segment lengths would have chosen so
            # the two can be compared directly.
            live_splits = _decode_num_splits(
                batch, heads_blocks, float(SWA_LEN), float(TOPK_LEN), 32
            )
            true_splits = _decode_num_splits(
                batch, heads_blocks, float(swa_len), float(topk_len), 32
            )
            all_splits = sorted({*splits, live_splits, true_splits})
            inp = _decode_inputs(
                batch,
                all_splits,
                device,
                topk_len,
                topk_rows,
                swa_len=swa_len,
                pool_rows=pool_rows,
            )
            variants: list[tuple[str, int, Callable[[], None]]] = [
                (
                    "single-pass",
                    batch * heads_blocks,
                    partial(_launch_single_pass, inp, block_h, 32),
                )
            ]

            def _tag(s: int, live: int = live_splits, true: int = true_splits) -> str:
                marks = []
                if s == live:
                    marks.append("live")
                if s == true and true != live:
                    marks.append("true-len")
                return f" ({'/'.join(marks)})" if marks else ""

            variants += [
                (
                    f"split-k s{s} w{w}{_tag(s)}",
                    batch * s * heads_blocks,
                    partial(_launch_split_k, inp, block_h, 32, s, w),
                )
                for s in all_splits
                for w in warps
            ]
            for label, ctas, fn in variants:
                rows.append(
                    [
                        str(batch),
                        str(block_h),
                        label,
                        str(ctas),
                        _fmt(_time_us(fn)),
                    ]
                )
    _print_table(
        f"sparse MLA decode (heads={NUM_HEADS}, swa={swa_len}, topk={topk_len}, "
        f"live rows={topk_rows} = {topk_rows * CACHE_ENTRY_BYTES / 2**20:.1f} MB"
        + (
            f", spread over a {pool_rows * CACHE_ENTRY_BYTES / 2**20:.0f} MB pool)"
            if pool_rows > topk_rows
            else ")"
        ),
        ["batch", "block_h", "impl", "CTAs", "us"],
        rows,
    )


# ---------------------------------------------------------------------------
# Sparse MLA prefill: reported at 17.9% occupancy, capped at 3 CTAs/SM by BOTH
# 168 regs/thread and 49,664 B smem. Both figures are trace metadata, so this
# mode reads registers, spills and smem off the compiled kernel instead and
# prints which of the two caps actually binds at each config.
# ---------------------------------------------------------------------------

# A100 (SM80) per-SM limits. sharedMemPerMultiprocessor is 164 KB; a single
# block may opt into at most 163 KB of it.
_SM80_REGS_PER_SM = 65536
_SM80_SMEM_PER_SM = 164 * 1024
_SM80_WARPS_PER_SM = 64


def _occupancy(n_regs: int, shared: int, num_warps: int) -> tuple[int, str, float]:
    """CTAs/SM, which resource caps it, and warp occupancy."""
    threads = 32 * num_warps
    # Registers are allocated per warp in granular chunks; the per-thread
    # figure times the thread count is the closest an outside model can get,
    # so treat a tie as "both" rather than pretending to resolve it.
    by_regs = _SM80_REGS_PER_SM // max(1, n_regs * threads)
    by_smem = _SM80_SMEM_PER_SM // max(1, shared) if shared else 32
    ctas = max(0, min(by_regs, by_smem, 32))
    if by_regs < by_smem:
        binder = "regs"
    elif by_smem < by_regs:
        binder = "smem"
    else:
        binder = "both"
    return ctas, binder, 100.0 * ctas * num_warps / _SM80_WARPS_PER_SM


def _prefill_inputs(
    m_tokens: int,
    ctx: int,
    kv_len: int,
    device: torch.device,
    index_mode: str = "topk",
) -> dict:
    """One chunk of ragged sparse prefill.

    ``index_mode`` picks which of the two layer populations the indices come
    from, and it is not a cosmetic knob -- it decides whether adjacent queries
    share rows, which is the whole property a query-blocked kernel would
    exploit:

    ``topk`` (the 21 ratio-4 layers): every query gathers ``kv_len``
    scattered rows out of a ``ctx``-row pool. Those layers' top-k sets are
    genuine per-query selections, so scattered indices are faithful.

    ``prefix`` (the 20 ratio-128 layers): the index list is not a selection at
    all -- ``sparse_mla.py`` builds it positionally as ``(pos+1)//128``, the
    identity prefix of the compressed cache. Adjacent queries then share all
    but at most one row. ``torch.randint`` destroys exactly that sharing,
    which is why R3 S3.2's standalone bench predicted 485 ms/chunk against a
    traced 315 ms and why its absolute rates were not quotable.
    """
    if index_mode == "prefix":
        ratio = _PREFILL_COMPRESS_RATIO
        kv_rows = max(1, ctx // ratio)
        # Queries sit at the deep end of the context, one position apart.
        positions = ctx - m_tokens + torch.arange(m_tokens, device=device)
        lens = ((positions + 1) // ratio).clamp(1, kv_rows)
        indptr = torch.zeros(m_tokens + 1, dtype=torch.int64, device=device)
        torch.cumsum(lens, 0, out=indptr[1:])
        total = int(indptr[-1].item())
        starts = indptr[:-1].repeat_interleave(lens)
        indices = (torch.arange(total, device=device) - starts).to(torch.int32)
        indptr = indptr.to(torch.int32)
    else:
        kv_rows = ctx
        indices = torch.randint(
            0, ctx, (m_tokens * kv_len,), dtype=torch.int32, device=device
        )
        indptr = torch.arange(
            0, m_tokens * kv_len + 1, kv_len, dtype=torch.int32, device=device
        )
    return dict(
        q=torch.randn(
            m_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        ),
        kv=torch.randn(kv_rows, HEAD_DIM, dtype=torch.bfloat16, device=device),
        indices=indices,
        indptr=indptr,
        attn_sink=torch.randn(NUM_HEADS, dtype=torch.float32, device=device),
    )


def _launch_sparse_prefill(
    inp: dict,
    block_h: int,
    block_k: int,
    num_warps: int,
    maxnreg: int,
    num_stages: int,
    out: torch.Tensor,
) -> object:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _sparse_attn_prefill_ragged_kernel,
    )

    q, kv = inp["q"], inp["kv"]
    extra = {"maxnreg": maxnreg} if maxnreg else {}
    # num_stages=0 means "pass nothing", i.e. Triton's default -- which is what
    # the serving launch does. It is not the same as num_stages=1: the default
    # pipelines to 49,664 B of smem, the exact figure the trace reported, while
    # 1 compiles a different (slower) kernel. Benchmarking against 1 would
    # invent a win that production already has.
    if num_stages:
        extra["num_stages"] = num_stages
    return _sparse_attn_prefill_ragged_kernel[
        (q.shape[0], triton.cdiv(NUM_HEADS, block_h))
    ](
        q,
        kv,
        inp["indices"],
        inp["indptr"],
        inp["attn_sink"],
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_HEADS,
        HEAD_DIM,
        kv.shape[0],
        HEAD_DIM**-0.5,
        HAS_ATTN_SINK=True,
        BLOCK_H=block_h,
        BLOCK_D=triton.next_power_of_2(HEAD_DIM),
        BLOCK_K=block_k,
        num_warps=num_warps,
        **extra,
    )


def bench_sparse_prefill(
    ms: list[int],
    ctx_ns: list[int],
    block_hs: list[int],
    block_ks: list[int],
    warps: list[int],
    maxnregs: list[int],
    stages: list[int],
    device: torch.device,
    index_modes: list[str] | None = None,
) -> None:
    index_modes = ["topk"] if index_modes is None else index_modes
    rows = []
    for m_tokens, ctx, index_mode in itertools.product(ms, ctx_ns, index_modes):
        kv_len = min(TOPK_LEN, ctx)
        inp = _prefill_inputs(m_tokens, ctx, kv_len, device, index_mode)
        out = torch.empty_like(inp["q"])
        rows_per_q = (inp["indptr"][-1].item()) / m_tokens
        for block_h, block_k, num_warps, maxnreg, num_stages in itertools.product(
            block_hs, block_ks, warps, maxnregs, stages
        ):
            fn = partial(
                _launch_sparse_prefill,
                inp,
                block_h,
                block_k,
                num_warps,
                maxnreg,
                num_stages,
                out,
            )
            n_regs = n_spills = shared = 0
            try:
                compiled = fn()
                torch.accelerator.synchronize()
                n_regs = getattr(compiled, "n_regs", 0)
                n_spills = getattr(compiled, "n_spills", 0)
                shared = getattr(compiled, "metadata", None)
                shared = getattr(shared, "shared", 0) if shared else 0
            except Exception as exc:  # noqa: BLE001 - report and keep sweeping
                print(f"  skipped ({type(exc).__name__}: {exc})")
            ctas, binder, occ = _occupancy(n_regs, shared, num_warps)
            rows.append(
                [
                    str(m_tokens),
                    str(ctx),
                    index_mode,
                    f"{rows_per_q:.0f}",
                    str(block_h),
                    str(block_k),
                    str(num_warps),
                    str(maxnreg) if maxnreg else "-",
                    str(num_stages) if num_stages else "def",
                    str(n_regs),
                    str(n_spills),
                    str(shared),
                    f"{ctas} ({binder})",
                    _fmt(occ),
                    _fmt(_time_us(fn)),
                ]
            )
    _print_table(
        f"sparse MLA prefill (heads={NUM_HEADS}, D={HEAD_DIM}, topk={TOPK_LEN})",
        [
            "M",
            "ctx",
            "indices",
            "rows/q",
            "bH",
            "bK",
            "warps",
            "maxnreg",
            "stages",
            "regs",
            "spill",
            "smem",
            "CTA/SM",
            "occ%",
            "us",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Sparse MLA at the ratio-128 operating point (K2). These 20 layers have no
# indexer: their index list is `(pos+1)//128` rows of identity prefix plus a
# 128-row sliding window, so consecutive queries read nested rows. That is the
# whole premise of the query-blocked path, and it is invisible to any arm that
# builds indices with randint -- which is why these are separate arms rather
# than new points on the ones above.
#
# `--prefill-index-mode prefix` above already puts the *existing* kernel on the
# compressed half of this row rule, and that is the arm to use for "what does
# the per-query kernel do when the rows are shared". These arms exist because
# the blocked kernel reads no index list: it re-derives the rows from the query
# positions, so measuring it needs the rest of the production geometry the
# index-list arms never have to model -- the SWA window, the per-request slab
# layout of the bf16 workspace, and the top-k cap. Both rungs here share one
# input, which is what makes the comparison an A/B rather than two arms.
#
# The ladder is one switch per rung (canon rule 17): the ragged kernel on real
# prefixes, then the blocked kernel at BLOCK_M=1 (same tile, no index list --
# it derives rows from positions), then the tile itself.
# ---------------------------------------------------------------------------

C128_MAX_MODEL_LEN = 262_144


def _c128_prefill_inputs(m_tokens: int, depth: int, device: torch.device) -> dict:
    """One ratio-128 prefill chunk of a single request at context ``depth``.

    Mirrors what `_forward_prefill` hands the kernel: a `[row_stride, 512]`
    bf16 workspace holding the dequantized compressed rows at `[0, n_rows)`
    and the gathered SWA rows at `[n_rows, ...)`, and one ragged index list per
    query built the way `_combine_topk_swa_indices_kernel` builds it.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    n_rows = triton.cdiv(C128_MAX_MODEL_LEN, _PREFILL_COMPRESS_RATIO)
    seq_len = depth + m_tokens
    gather_len = m_tokens + min(depth, SWA_LEN - 1)
    gather_start = seq_len - gather_len
    row_stride = n_rows + SWA_LEN + m_tokens
    ratio = _PREFILL_COMPRESS_RATIO
    top_k = min(max(triton.next_power_of_2(max(seq_len // ratio, 1)), ratio), n_rows)

    positions = torch.arange(depth, seq_len, device=device, dtype=torch.int32)
    topk_len = torch.clamp(
        torch.div(positions + 1, ratio, rounding_mode="floor"), max=top_k
    )
    swa_len = torch.clamp(positions + 1, max=SWA_LEN)
    swa_row = n_rows + positions + 1 - swa_len - gather_start
    lens = (topk_len + swa_len).to(torch.int32)

    width = int(lens.max())
    col = torch.arange(width, device=device, dtype=torch.int32)
    dense = torch.where(
        col[None, :] < topk_len[:, None],
        col[None, :],
        swa_row[:, None] + col[None, :] - topk_len[:, None],
    )
    dense = torch.where(col[None, :] < lens[:, None], dense, -1).to(torch.int32)
    indices, indptr = build_ragged_indices_from_dense(dense, lens, num_rows=row_stride)

    q = torch.randn(m_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    query_start_loc = torch.tensor([0, m_tokens], dtype=torch.int32, device=device)
    return dict(
        q=q,
        out=torch.empty_like(q),
        kv=torch.randn(row_stride, HEAD_DIM, dtype=torch.bfloat16, device=device),
        dense=dense,
        lens=lens,
        indices=indices,
        indptr=indptr,
        attn_sink=torch.randn(NUM_HEADS, dtype=torch.float32, device=device),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        gather_lens=torch.tensor([gather_len], dtype=torch.int32, device=device),
        top_k=top_k,
        row_stride=row_stride,
        swa_offset=n_rows,
        rows_per_query=float(lens.float().mean()),
    )


def _launch_c128_prefill_ragged(inp: dict, block_h: int, block_k: int, num_warps: int):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _sparse_attn_prefill_ragged_kernel,
    )

    q, out, kv = inp["q"], inp["out"], inp["kv"]
    return _sparse_attn_prefill_ragged_kernel[
        (q.shape[0], triton.cdiv(NUM_HEADS, block_h))
    ](
        q,
        kv,
        inp["indices"],
        inp["indptr"],
        inp["attn_sink"],
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_HEADS,
        HEAD_DIM,
        kv.shape[0],
        SCALE,
        HAS_ATTN_SINK=True,
        BLOCK_H=block_h,
        BLOCK_D=triton.next_power_of_2(HEAD_DIM),
        BLOCK_K=block_k,
        EXACT_TILE=block_h == NUM_HEADS,
        num_warps=num_warps,
    )


def _launch_c128_prefill_blocked(
    inp: dict, block_m: int, block_h: int, block_k: int, num_warps: int
):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _sparse_attn_prefill_blocked_kernel,
        build_query_blocks,
    )

    q, out, kv = inp["q"], inp["out"], inp["kv"]
    blocks = inp["blocks"].get(block_m)
    if blocks is None:
        blocks = build_query_blocks(inp["query_start_loc_cpu"], block_m, q.device)
        inp["blocks"][block_m] = blocks
    return _sparse_attn_prefill_blocked_kernel[
        (blocks[0].numel(), triton.cdiv(NUM_HEADS, block_h))
    ](
        q,
        kv,
        blocks[0],
        blocks[1],
        inp["query_start_loc"],
        inp["seq_lens"],
        inp["gather_lens"],
        inp["attn_sink"],
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        inp["top_k"],
        inp["row_stride"],
        inp["swa_offset"],
        SCALE,
        HAS_ATTN_SINK=True,
        COMPRESS_RATIO=_PREFILL_COMPRESS_RATIO,
        WINDOW_SIZE=SWA_LEN,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_D=triton.next_power_of_2(HEAD_DIM),
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def _c128_prefill_fp32_error(inp: dict, out: torch.Tensor, samples: int = 24) -> float:
    """Error of sampled query rows against fp32 softmax attention.

    Scaled by the row's own largest component, not per element. An attention
    output over ~1,800 rows of random KV has components scattered around zero,
    so a per-element ratio divides by noise and reports ~0.2 for a kernel that
    is exactly right -- which is what the per-query kernel, carried in this
    same check as its own control, measured before this was fixed.

    Sampled rather than exhaustive: the gathered KV of one 15,360-query chunk
    does not fit anywhere. Rows are drawn across the chunk so the sample spans
    short and long prefixes, and the first and last query are always in it --
    those are the two the block's masks treat specially.
    """
    m_tokens = inp["q"].shape[0]
    picks = sorted({0, m_tokens - 1, *torch.randint(0, m_tokens, (samples,)).tolist()})
    worst = 0.0
    for t in picks:
        n = int(inp["lens"][t])
        rows = inp["dense"][t, :n].to(torch.int64)
        kv = inp["kv"][rows].to(torch.float32)
        scores = (inp["q"][t].to(torch.float32) @ kv.T) * SCALE
        sink = inp["attn_sink"].to(torch.float32)[:, None]
        weights = torch.softmax(torch.cat([scores, sink], dim=1), dim=1)[:, :-1]
        expected = weights @ kv
        got = out[t].to(torch.float32)
        scale = expected.abs().max().clamp(min=1e-6)
        worst = max(worst, float((got - expected).abs().max() / scale))
    return worst


def bench_sparse_prefill_c128(
    ms: list[int],
    depths: list[int],
    block_ms: list[int],
    block_ks: list[int],
    warps: list[int],
    device: torch.device,
    check: bool = True,
) -> None:
    rows = []
    for m_tokens, depth in itertools.product(ms, depths):
        inp = _c128_prefill_inputs(m_tokens, depth, device)
        inp["blocks"] = {}
        block_h = min(16, max(8, triton.next_power_of_2(NUM_HEADS)))
        rows_per_query = inp["rows_per_query"]
        # 2 dots of 2*BLOCK_D per (query, head, row).
        flop = 4.0 * HEAD_DIM * NUM_HEADS * rows_per_query * m_tokens

        for block_k, num_warps in itertools.product(block_ks, warps):
            fn = partial(_launch_c128_prefill_ragged, inp, block_h, block_k, num_warps)
            us = _time_us(fn)
            err = _c128_prefill_fp32_error(inp, inp["out"]) if check else float("nan")
            rows.append(
                [
                    str(m_tokens),
                    str(depth),
                    "ragged (per query)",
                    str(block_k),
                    str(num_warps),
                    f"{m_tokens}",
                    _fmt(us),
                    _fmt(flop / us * 1e-6) if us == us else "-",
                    f"{err:.1e}",
                ]
            )

        for block_m, block_k, num_warps in itertools.product(block_ms, block_ks, warps):
            fn = partial(
                _launch_c128_prefill_blocked,
                inp,
                block_m,
                block_h,
                block_k,
                num_warps,
            )
            us = _time_us(fn)
            err = _c128_prefill_fp32_error(inp, inp["out"]) if check else float("nan")
            rows.append(
                [
                    str(m_tokens),
                    str(depth),
                    f"blocked M={block_m}",
                    str(block_k),
                    str(num_warps),
                    str(triton.cdiv(m_tokens, block_m)),
                    _fmt(us),
                    _fmt(flop / us * 1e-6) if us == us else "-",
                    f"{err:.1e}",
                ]
            )
    _print_table(
        f"sparse MLA prefill, ratio-128 dense prefix (heads={NUM_HEADS}, "
        f"D={HEAD_DIM}, window={SWA_LEN})",
        ["M", "depth", "impl", "bK", "warps", "CTAs", "us", "TFLOP/s", "relerr"],
        rows,
    )


def _c128_paged_slots(
    num_reqs: int, per_req: int, pool_rows: int, device: torch.device
) -> torch.Tensor:
    """`[num_reqs, per_req]` slot ids, paged the way a free list hands them out.

    Row `i` of a request always lands in the same slot regardless of which
    query reads it -- that is what makes the prefixes nest -- while the pages
    themselves are scattered through the pool, which is what keeps the address
    spread honest.
    """
    pages_per_req = triton.cdiv(per_req, CACHE_BLOCK_SIZE)
    pool_pages = max(num_reqs * pages_per_req, pool_rows // CACHE_BLOCK_SIZE)
    pages = torch.randperm(pool_pages, device=device)[: num_reqs * pages_per_req]
    pages = pages.reshape(num_reqs, pages_per_req)
    off = torch.arange(per_req, device=device)
    return (
        pages[:, off // CACHE_BLOCK_SIZE] * CACHE_BLOCK_SIZE
        + (off % CACHE_BLOCK_SIZE)[None, :]
    ).to(torch.int32)


def _c128_decode_inputs(
    batch: int,
    next_n: int,
    depth: int,
    splits: list[int],
    device: torch.device,
    pool_rows: int = 0,
) -> dict:
    """`batch` requests of `next_n` query tokens each at consecutive positions.

    Both segments are built from a per-request positional slot map, so the
    compressed lists nest and the SWA windows slide by one -- the structure the
    blocked kernel derives from `indptr`. Both kernels then take *the same*
    ragged buffers, so the arm compares implementations and nothing else.
    """
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    is_fnuz = current_platform.is_fp8_fnuz()
    positions = (
        depth + torch.arange(next_n, device=device, dtype=torch.int32)[None, :]
    ).expand(batch, next_n)

    comp_per_req = max(1, (depth + next_n) // _PREFILL_COMPRESS_RATIO)
    comp_slots = _c128_paged_slots(batch, comp_per_req, pool_rows, device)
    comp_lens = torch.clamp(
        torch.div(positions + 1, _PREFILL_COMPRESS_RATIO, rounding_mode="floor"),
        max=comp_per_req,
    ).reshape(-1)
    comp_dense = (
        comp_slots[:, None, :]
        .expand(batch, next_n, comp_per_req)
        .reshape(batch * next_n, comp_per_req)
    )
    comp_dense = torch.where(
        torch.arange(comp_per_req, device=device)[None, :] < comp_lens[:, None],
        comp_dense,
        -1,
    ).to(torch.int32)

    swa_span = min(depth + next_n, SWA_LEN + next_n - 1)
    swa_first = depth + next_n - swa_span
    swa_slots = _c128_paged_slots(batch, swa_span, 0, device)
    swa_lens = torch.clamp(positions + 1, max=SWA_LEN).reshape(-1)
    swa_start = (positions + 1 - torch.clamp(positions + 1, max=SWA_LEN)).reshape(
        -1
    ) - swa_first
    col = torch.arange(SWA_LEN, device=device)
    gather = torch.clamp(swa_start[:, None] + col[None, :], 0, swa_span - 1)
    swa_dense = torch.gather(
        swa_slots.repeat_interleave(next_n, dim=0), 1, gather.to(torch.int64)
    )
    swa_dense = torch.where(col[None, :] < swa_lens[:, None], swa_dense, -1).to(
        torch.int32
    )

    comp_rows = int(comp_slots.max()) + 1
    swa_rows = int(swa_slots.max()) + 1
    main_indices, main_indptr = build_ragged_indices_from_dense(
        swa_dense, swa_lens.to(torch.int32), num_rows=swa_rows
    )
    extra_indices, extra_indptr = build_ragged_indices_from_dense(
        comp_dense, comp_lens.to(torch.int32), num_rows=comp_rows
    )

    num_queries = batch * next_n
    q = torch.randn(
        num_queries, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    part = {
        s: (
            torch.empty(
                (num_queries, s, NUM_HEADS), dtype=torch.float32, device=device
            ),
            torch.empty(
                (num_queries, s, NUM_HEADS), dtype=torch.float32, device=device
            ),
            torch.empty(
                (num_queries, s, NUM_HEADS, HEAD_DIM),
                dtype=torch.float32,
                device=device,
            ),
        )
        for s in splits
    }
    return dict(
        q=q,
        out=torch.empty_like(q),
        main_cache=_make_ds_mla_cache(swa_rows, CACHE_BLOCK_SIZE, is_fnuz, device),
        extra_cache=_make_ds_mla_cache(comp_rows, CACHE_BLOCK_SIZE, False, device),
        main_indices=main_indices,
        main_indptr=main_indptr,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
        attn_sink=torch.randn(NUM_HEADS, dtype=torch.float32, device=device),
        fp8_lut=get_e4m3fn_bf16_lut(device),
        is_fnuz=is_fnuz,
        part=part,
        next_n=next_n,
        rows_per_query=float((comp_lens + swa_lens).float().mean()),
    )


def _launch_c128_decode_blocked(
    inp: dict, block_m: int, block_h: int, block_k: int, num_splits: int, warps: int
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _sparse_attn_decode_partial_blocked_kernel,
        _sparse_attn_decode_reduce_kernel,
    )

    q, out = inp["q"], inp["out"]
    part_m, part_l, part_acc = inp["part"][num_splits]
    main_cache, extra_cache = inp["main_cache"], inp["extra_cache"]
    next_n = inp["next_n"]
    _sparse_attn_decode_partial_blocked_kernel[
        (q.shape[0] // next_n, num_splits, triton.cdiv(NUM_HEADS, block_h))
    ](
        q,
        main_cache,
        inp["main_indices"],
        inp["main_indptr"],
        extra_cache,
        inp["extra_indices"],
        inp["extra_indptr"],
        part_m,
        part_l,
        part_acc,
        inp["fp8_lut"],
        q.stride(0),
        q.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        SCALE,
        next_n,
        HAS_EXTRA=True,
        NOPE_DIM=NOPE_DIM,
        NOPE_BLOCK=triton.next_power_of_2(NOPE_DIM),
        ROPE_DIM=ROPE_DIM,
        IS_FNUZ_MAIN=inp["is_fnuz"],
        IS_FNUZ_EXTRA=False,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        NUM_STAGES=1,
        num_warps=warps,
    )
    _sparse_attn_decode_reduce_kernel[(q.shape[0], NUM_HEADS)](
        part_m,
        part_l,
        part_acc,
        inp["attn_sink"],
        out,
        out.stride(0),
        out.stride(1),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        NUM_HEADS,
        HAS_ATTN_SINK=True,
        COMB_DIM=HEAD_DIM,
        BLOCK_H=1,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=triton.next_power_of_2(num_splits),
        num_warps=4,
    )


def bench_sparse_decode_c128(
    batches: list[int],
    next_n: int,
    depths: list[int],
    block_ms: list[int],
    splits: list[int],
    warps: list[int],
    device: torch.device,
    pool_rows: int = 0,
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _decode_num_splits

    block_h = min(16, max(8, triton.next_power_of_2(NUM_HEADS)))
    heads_blocks = triton.cdiv(NUM_HEADS, block_h)
    rows = []
    for batch, depth in itertools.product(batches, depths):
        num_queries = batch * next_n
        # The per-query kernel's own split choice, and the blocked one's: the
        # block count is `next_n` times smaller, so the heuristic sees a
        # different device fill and this is where that shows up.
        live = _decode_num_splits(num_queries, heads_blocks, SWA_LEN, TOPK_LEN, 32)
        blocked_live = _decode_num_splits(batch, heads_blocks, SWA_LEN, TOPK_LEN, 32)
        all_splits = sorted({*splits, live, blocked_live})
        inp = _c128_decode_inputs(
            batch, next_n, depth, all_splits, device, pool_rows=pool_rows
        )
        flop = 4.0 * HEAD_DIM * NUM_HEADS * inp["rows_per_query"] * num_queries

        reference = None

        def _err(got: torch.Tensor, reference: torch.Tensor | None) -> str:
            # Scaled by the output's own magnitude, not per element: attention
            # over ~1,700 rows of random KV has components scattered around
            # zero, and a per-element ratio divides by that noise.
            if reference is None:
                return "-"
            scale = reference.abs().max().clamp(min=1e-6)
            err = (got.to(torch.float32) - reference).abs().max() / scale
            return f"{float(err):.1e}"

        for s in all_splits:
            for w in warps:
                fn = partial(_launch_split_k, inp, block_h, 32, s, w)
                us = _time_us(fn)
                # The per-query kernel measured against its own first split
                # count is the metric's floor: whatever it reports is split-K
                # reassociation, not a property of the blocked path.
                err = _err(inp["out"], reference) if us == us else "-"
                if reference is None and us == us:
                    reference = inp["out"].clone().to(torch.float32)
                    err = "ref"
                rows.append(
                    [
                        str(batch),
                        str(depth),
                        f"per query s{s} w{w}" + (" (live)" if s == live else ""),
                        str(num_queries * s * heads_blocks),
                        _fmt(us),
                        _fmt(flop / us * 1e-6) if us == us else "-",
                        err,
                    ]
                )
        # A group is never split across CTAs, so a tile narrower than next_n
        # would leave the group's last queries unwritten. `decode_block_tile`
        # declines those in production; here they are simply not run.
        usable = [b for b in block_ms if b >= next_n]
        for block_m, s, w in itertools.product(usable, all_splits, warps):
            fn = partial(_launch_c128_decode_blocked, inp, block_m, block_h, 32, s, w)
            us = _time_us(fn)
            err = _err(inp["out"], reference) if us == us else "-"
            rows.append(
                [
                    str(batch),
                    str(depth),
                    f"blocked M={block_m} s{s} w{w}"
                    + (" (live)" if s == blocked_live else ""),
                    str(batch * s * heads_blocks),
                    _fmt(us),
                    _fmt(flop / us * 1e-6) if us == us else "-",
                    err,
                ]
            )
    _print_table(
        f"sparse MLA decode, ratio-128 (heads={NUM_HEADS}, next_n={next_n}, "
        f"window={SWA_LEN})",
        ["C", "depth", "impl", "CTAs", "us", "TFLOP/s", "relerr"],
        rows,
    )


# ---------------------------------------------------------------------------
# mHC prenorm GEMM: [T, hc_mult*hidden] x [24, hc_mult*hidden]^T
# ---------------------------------------------------------------------------


def _torch_prenorm(
    x: torch.Tensor, fn: torch.Tensor, out: torch.Tensor, sqrsum: torch.Tensor
) -> None:
    x_wide = x if fn.dtype == torch.bfloat16 else x.float()
    out[0].copy_(x_wide @ fn.t())
    sqrsum[0].copy_(x.float().square().sum(-1))


def bench_prenorm_gemm(
    tokens: list[int], configs: list[tuple[int, int]], device: torch.device
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        hc_prenorm_gemm_block_m_tilelang,
        hc_prenorm_gemm_tilelang,
    )

    k = HC_MULT * HIDDEN_SIZE
    fn = torch.randn(HC_MULT3, k, dtype=torch.float32, device=device)
    fn_bf16 = fn.to(torch.bfloat16)
    rows = []
    for num_tokens in tokens:
        x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device=device)
        out = torch.empty(1, num_tokens, HC_MULT3, dtype=torch.float32, device=device)
        sqrsum = torch.empty(1, num_tokens, dtype=torch.float32, device=device)
        # fn dominates: it is re-read by every CTA that owns a token tile.
        bytes_min = x.numel() * 2 + fn.numel() * 4
        variants: list[tuple[str, str, Callable[[], None]]] = [
            (
                "tilelang",
                "split-1",
                partial(
                    hc_prenorm_gemm_tilelang,
                    x,
                    fn,
                    out,
                    sqrsum,
                    HIDDEN_SIZE,
                    HC_MULT,
                    HC_MULT3,
                    512,
                    12,
                    1,
                ),
            )
        ]
        # `fn` traffic scales 1/block_m, `x` traffic 1/tile_n; registers run
        # ~block_m*tile_n + block_m + tile_n + 20 against a 128-reg ceiling, so
        # the pair must move together. (8, 12) is expected to fail to launch.
        variants += [
            (
                "tilelang",
                f"block_m={m},tile_n={t}",
                partial(
                    hc_prenorm_gemm_block_m_tilelang,
                    x,
                    fn,
                    out,
                    sqrsum,
                    HIDDEN_SIZE,
                    HC_MULT,
                    HC_MULT3,
                    512,
                    t,
                    m,
                ),
            )
            for m, t in configs
        ]
        # Reference routes for the fp32 -> bf16 tensor-core question.
        variants += [
            ("torch", "fp32", partial(_torch_prenorm, x, fn, out, sqrsum)),
            ("torch", "bf16", partial(_torch_prenorm, x, fn_bf16, out, sqrsum)),
        ]
        # The production cuBLAS route: bf16 GEMM + one-pass Triton sqrsum.
        from vllm.model_executor.kernels.mhc.triton import hc_prenorm_gemm_cublas

        variants += [
            (
                "cublas",
                "gemm+sqrsum",
                partial(hc_prenorm_gemm_cublas, x, fn, out, sqrsum),
            )
        ]
        for impl, cfg, fn_ in variants:
            us = _time_us(fn_)
            gbs = float("nan") if us != us else bytes_min / (us * 1e-6) / 1e9
            rows.append([str(num_tokens), impl, cfg, _fmt(us), _fmt(gbs, ".0f")])
    _print_table(
        f"mHC prenorm GEMM (K={k}, N={HC_MULT3}, fp32 weight)",
        ["tokens", "impl", "cfg", "us", "GB/s"],
        rows,
    )


# ---------------------------------------------------------------------------
# mHC pre big-fuse. The grid is one CTA per token, but widening the worker warps
# was measured to do nothing (8.33 us at both 64 and 256 workers at batch 1):
# warp 0 walks `sinkhorn_repeat` dependent 4x4 normalizations serially and is the
# critical path. This sweeps that count to keep the attribution honest -- the
# difference against repeat=1 is the addressable share.
# ---------------------------------------------------------------------------


def bench_mhc_pre(
    tokens: list[int], sinkhorn_iters: list[int], device: torch.device
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_pre_big_fuse_with_norm_tilelang,
    )

    n_splits = 1
    rows = []
    for num_tokens in tokens:
        tensors = (
            torch.randn(
                n_splits, num_tokens, HC_MULT3, dtype=torch.float32, device=device
            ),
            torch.rand(n_splits, num_tokens, dtype=torch.float32, device=device) + 1.0,
            torch.randn(3, dtype=torch.float32, device=device),
            torch.randn(HC_MULT3, dtype=torch.float32, device=device),
            torch.randn(
                num_tokens, HC_MULT, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
            ),
            torch.empty(num_tokens, HC_MULT, dtype=torch.float32, device=device),
            torch.empty(
                num_tokens, HC_MULT * HC_MULT, dtype=torch.float32, device=device
            ),
            torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
            torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        )
        # residual read + normed layer_input write
        moved = num_tokens * HIDDEN_SIZE * 2 * (HC_MULT + 1)
        for iters in sinkhorn_iters:
            us = _time_us(
                partial(
                    mhc_pre_big_fuse_with_norm_tilelang,
                    *tensors,
                    HIDDEN_SIZE,
                    1e-6,
                    1e-6,
                    1e-6,
                    2.0,
                    iters,
                    1e-6,
                    n_splits,
                    HC_MULT,
                )
            )
            gbs = float("nan") if us != us else moved / (us * 1e-6) / 1e9
            label = f"{iters} (checkpoint)" if iters == SINKHORN_ITERS else str(iters)
            rows.append(
                [
                    str(num_tokens),
                    label,
                    _fmt(us),
                    _fmt(us * MHC_PRE_LAUNCHES_PER_STEP / 1000, ".3f"),
                    _fmt(gbs, ".0f"),
                ]
            )
    _print_table(
        f"mHC pre big-fuse + norm (hidden={HIDDEN_SIZE}, hc_mult={HC_MULT}); "
        f"ms/step assumes {MHC_PRE_LAUNCHES_PER_STEP} launches/step",
        ["tokens", "sinkhorn_iters", "us", "ms/step", "GB/s"],
        rows,
    )


# ---------------------------------------------------------------------------
# Prefill K-cache dequantize + gather
# ---------------------------------------------------------------------------


def _launch_dequant_gather(
    out: torch.Tensor,
    cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_workers: int,
) -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        DequantizeAndGatherKCacheKernel,
    )

    DequantizeAndGatherKCacheKernel.kernel[(seq_lens.shape[0], num_workers)](
        out,
        out.stride(0),
        out.stride(1),
        cache,
        seq_lens,
        block_table,
        0,
        None,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=NOPE_DIM,
        bf16_dim=ROPE_DIM,
        scale_dim=SCALE_DIM,
        quant_block=QUANT_BLOCK,
        cache_block_size=CACHE_BLOCK_SIZE,
        token_data_size=CACHE_TOKEN_BYTES,
        block_stride=cache.stride(0),
        output_dim=512,
        fp8_max=448.0,
        fp8_block=triton.next_power_of_2(NOPE_DIM),
        use_fnuz=False,
    )


def bench_dequant_gather(
    gather_lens: list[int],
    num_reqs_list: list[int],
    workers: list[int],
    device: torch.device,
) -> None:
    rows = []
    for num_reqs, gather_len in itertools.product(num_reqs_list, gather_lens):
        cache = _make_ds_mla_cache(gather_len, CACHE_BLOCK_SIZE, False, device)
        blocks_per_seq = triton.cdiv(gather_len, CACHE_BLOCK_SIZE)
        block_table = (
            torch.arange(blocks_per_seq, dtype=torch.int32, device=device)
            .repeat(num_reqs, 1)
            .contiguous()
        )
        seq_lens = torch.full((num_reqs,), gather_len, dtype=torch.int32, device=device)
        out = torch.empty(
            num_reqs, gather_len, 512, dtype=torch.bfloat16, device=device
        )
        moved = num_reqs * gather_len * (CACHE_ENTRY_BYTES + 512 * 2)
        for num_workers in workers:
            us = _time_us(
                partial(
                    _launch_dequant_gather,
                    out,
                    cache,
                    seq_lens,
                    block_table,
                    num_workers,
                )
            )
            gbs = float("nan") if us != us else moved / (us * 1e-6) / 1e9
            rows.append(
                [
                    str(num_reqs),
                    str(gather_len),
                    str(num_workers),
                    _fmt(us),
                    _fmt(gbs, ".0f"),
                ]
            )
    _print_table(
        "dequantize_and_gather_k_cache (Triton)",
        ["reqs", "gather_len", "workers", "us", "GB/s"],
        rows,
    )


# ---------------------------------------------------------------------------
# Indexer MQA logits (prefill). Register-limited: 132 regs/thread caps 3 CTAs/SM
# and Compute Warps in Flight sits at ~96% of that ceiling, so `maxnreg` is the
# knob this sweep exposes (128 is exactly the 4th-CTA/SM boundary at 128
# threads). Per-CTA cost is context-independent, so one long-N point per M is
# representative of any context length.
# ---------------------------------------------------------------------------


_LOGITS_BLOCK_N = 128  # production autotune space is BLOCK_N=128 only


def _launch_indexer_logits(
    inp: dict,
    grid: tuple[int, int],
    n_ctx: int,
    maxnreg: int,
    num_stages: int,
    kv_group: int,
    factor_k_scale: bool = True,
) -> object:
    from vllm.v1.attention.ops.mqa_logits_triton import _fp8_mqa_logits_kernel

    # Bypass @triton.autotune via .fn so maxnreg is an explicit knob.
    extra = {"maxnreg": maxnreg} if maxnreg else {}
    q, k, weights, logits = inp["q"], inp["k"], inp["weights"], inp["logits"]
    return _fp8_mqa_logits_kernel.fn[grid](
        q,
        k,
        inp["k_scales"],
        weights,
        inp["ks"],
        inp["ke"],
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_heads=CFG_INDEX_N_HEADS,
        head_dim=CFG_INDEX_HEAD_DIM,
        N=n_ctx,
        BLOCK_H=max(16, triton.next_power_of_2(CFG_INDEX_N_HEADS)),
        BLOCK_D=triton.next_power_of_2(CFG_INDEX_HEAD_DIM),
        BLOCK_N=_LOGITS_BLOCK_N,
        KV_GROUP=kv_group,
        FACTOR_K_SCALE=factor_k_scale,
        num_warps=4,
        num_stages=num_stages,
        **extra,
    )


def _check_factored_k_scale(inp: dict, n_ctx: int, kv_group: int) -> str:
    """Both epilogue variants, same inputs, one invocation.

    Reports the logit delta and whether the indexer's top-k SELECTION moved.
    The relu's active set cannot move (k_scale >= 0 cannot flip a sign), but
    the head sum's rounding order does change -- one scaling instead of
    BLOCK_H -- so near-ties can in principle reorder. This is a structural
    check on synthetic inputs; canon rule 34 requires the real gate to run on
    captured tensors (see /root/optim/k7_numerics.py).
    """
    m_tokens = inp["q"].shape[0]
    grid = (m_tokens, triton.cdiv(n_ctx, _LOGITS_BLOCK_N * kv_group))
    outs = {}
    for factored in (False, True):
        inp["logits"].zero_()
        _launch_indexer_logits(inp, grid, n_ctx, 0, 2, kv_group, factored)
        torch.accelerator.synchronize()
        outs[factored] = inp["logits"].clone()

    ref, new = outs[False], outs[True]
    finite = torch.isfinite(ref) & torch.isfinite(new)
    diff = (ref - new).abs()[finite]
    scale = ref.abs()[finite].clamp_min(1e-30)
    max_abs = diff.max().item() if diff.numel() else 0.0
    max_rel = (diff / scale).max().item() if diff.numel() else 0.0

    k = min(TOPK_LEN, n_ctx)
    sel_ref = torch.topk(ref, k, dim=-1).indices.sort(dim=-1).values
    sel_new = torch.topk(new, k, dim=-1).indices.sort(dim=-1).values
    moved = int((sel_ref != sel_new).any(dim=-1).sum().item())

    assert max_rel < 1e-4, f"factored k_scale changed logits by {max_rel:.2e} rel"
    return (
        f"M={m_tokens} N={n_ctx} G={kv_group}: max|d|={max_abs:.3e} "
        f"rel={max_rel:.3e}, top-{k} rows moved {moved}/{m_tokens}"
    )


def bench_indexer_logits(
    ms: list[int],
    ctx_ns: list[int],
    maxnregs: list[int],
    stages: list[int],
    kv_groups: list[int],
    device: torch.device,
    factor_scales: list[int] | None = None,
) -> None:
    factor_scales = [0, 1] if factor_scales is None else factor_scales
    rows = []
    checks = []
    for m_tokens, n_ctx in itertools.product(ms, ctx_ns):
        inp = dict(
            q=torch.randn(
                m_tokens,
                CFG_INDEX_N_HEADS,
                CFG_INDEX_HEAD_DIM,
                dtype=torch.bfloat16,
                device=device,
            ),
            k=torch.randn(
                n_ctx, CFG_INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device
            ),
            k_scales=torch.rand(n_ctx, dtype=torch.float32, device=device) + 0.5,
            weights=torch.rand(
                m_tokens, CFG_INDEX_N_HEADS, dtype=torch.float32, device=device
            ),
            # Full [0, N) range per row: at long context nearly every chunk row
            # attends the whole compressed prefix, so this is the loaded case
            # the trace measured, not an adversarial one.
            ks=torch.zeros(m_tokens, dtype=torch.int32, device=device),
            ke=torch.full((m_tokens,), n_ctx, dtype=torch.int32, device=device),
            logits=torch.empty(m_tokens, n_ctx, dtype=torch.float32, device=device),
        )
        for kv_group in kv_groups:
            checks.append(_check_factored_k_scale(inp, n_ctx, kv_group))
        for maxnreg, num_stages, kv_group, factored in itertools.product(
            maxnregs, stages, kv_groups, factor_scales
        ):
            grid = (m_tokens, triton.cdiv(n_ctx, _LOGITS_BLOCK_N * kv_group))
            fn = partial(
                _launch_indexer_logits,
                inp,
                grid,
                n_ctx,
                maxnreg,
                num_stages,
                kv_group,
                bool(factored),
            )
            n_regs = n_spills = 0
            try:
                compiled = fn()
                torch.accelerator.synchronize()
                n_regs = getattr(compiled, "n_regs", 0)
                n_spills = getattr(compiled, "n_spills", 0)
            except Exception as exc:  # noqa: BLE001 - report and keep sweeping
                print(f"  skipped ({type(exc).__name__}: {exc})")
            us = _time_us(fn)
            ns_per_cta = float("nan") if us != us else us * 1e3 / (grid[0] * grid[1])
            rows.append(
                [
                    str(m_tokens),
                    str(n_ctx),
                    str(maxnreg) if maxnreg else "-",
                    str(num_stages),
                    str(kv_group),
                    "yes" if factored else "no",
                    str(n_regs),
                    str(n_spills),
                    _fmt(us),
                    _fmt(ns_per_cta, ".1f"),
                ]
            )
    _print_table(
        f"indexer MQA logits (H={CFG_INDEX_N_HEADS}, D={CFG_INDEX_HEAD_DIM}, "
        f"BLOCK_N={_LOGITS_BLOCK_N}, num_warps=4)",
        [
            "M",
            "N",
            "maxnreg",
            "stages",
            "G",
            "k_scale out",
            "regs",
            "spill",
            "us",
            "ns/CTA",
        ],
        rows,
    )
    print("\n  factored-vs-scaled epilogue (same invocation, synthetic q/k):")
    for line in checks:
        print(f"    {line}")


# ---------------------------------------------------------------------------
# Indexer MQA logits (paged decode). Same math as the prefill kernel above but
# one query token against a paged K cache, and it runs 21 times per decode step
# (only the ratio-4 layers carry an indexer). A node-granularity trace measured
# it at 7.59 us short-context and 15.94 us at 107k, so the sweep is over the
# compressed context length -- which is what sets the CTA count, since the grid
# is (B * next_n, cdiv(n_compressed, block_size)).
# ---------------------------------------------------------------------------

# The indexer cache holds `cache_config.block_size // compress_ratio` compressed
# tokens per block (`kv_cache_interface.py:405`); this deployment runs 256 // 4,
# and 64 is also what the DEEPSEEK_SPARSE_SWA preferred block size reduces to.
# BLOCK_N follows block_size, so this is a shape constant, not a tuning knob.
_PAGED_BLOCK_SIZE = 64
_PAGED_COMPRESS_RATIO = 4


def _paged_logits_inputs(
    batch: int, n_compressed: int, device: torch.device, next_n: int = 6
) -> dict:
    """Decode inputs for `_fp8_paged_mqa_logits_kernel`, laid out exactly as
    `fp8_paged_mqa_logits_triton` re-derives them from the paged cache.

    ``next_n`` is the DSpark draft length: production runs 6 (5 speculative
    tokens plus the bonus), and the grid's first dimension is B * next_n, so
    the harness's old hardcoded 1 measured a grid 6x smaller than serving's
    at the same batch (canon rule 7).

    The block table scatters each sequence's blocks over a pool several times
    its size, because live block IDs come off a free list rather than being
    consecutive; the pool is deliberately not the full 48k-block deployment
    cache, whose 400 MB would measure a DRAM regime one decode never touches
    (a 107k sequence reads 418 blocks = 3.5 MB).
    """
    from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut

    head_dim = CFG_INDEX_HEAD_DIM
    block_size = _PAGED_BLOCK_SIZE
    seq_blocks = triton.cdiv(n_compressed, block_size)
    pool_blocks = max(2048, 4 * batch * seq_blocks)

    kv_cache = torch.randint(
        0,
        256,
        (pool_blocks, block_size, 1, head_dim + 4),
        dtype=torch.uint8,
        device=device,
    )
    kv_flat = kv_cache.view(pool_blocks, -1)
    k_end = block_size * head_dim
    kv_byte = kv_flat[:, :k_end].as_strided(
        (pool_blocks, block_size, head_dim),
        (kv_flat.stride(0), head_dim, 1),
    )
    kv_scale = kv_flat[:, k_end:].view(torch.float32)
    # Random bytes would decode to arbitrary fp8 scales including NaN; overwrite
    # the scale region with sane positive fp32 so the logits stay finite.
    kv_scale.copy_(torch.rand_like(kv_scale) + 0.5)

    block_tables = torch.stack(
        [
            torch.randperm(pool_blocks, device=device)[:seq_blocks].to(torch.int32)
            for _ in range(batch)
        ]
    )
    q = torch.randint(
        0,
        256,
        (batch, next_n, CFG_INDEX_N_HEADS, head_dim),
        dtype=torch.uint8,
        device=device,
    )
    lut = get_e4m3fn_bf16_lut(device, nan_value=480.0)
    return dict(
        q=q,
        # K1's treatment operand: the same table applied once on the host.
        q_bf16=lut.index_select(0, q.reshape(-1).to(torch.int32)).view(q.shape),
        next_n=next_n,
        kv_byte=kv_byte,
        kv_scale=kv_scale,
        weights=torch.rand(
            batch * next_n, CFG_INDEX_N_HEADS, dtype=torch.float32, device=device
        ),
        fp8_lut=lut,
        context_lens=torch.full(
            (batch,), n_compressed, dtype=torch.int32, device=device
        ),
        block_tables=block_tables.contiguous(),
        # clean_logits=False in serving: the buffer is left uninitialized and
        # the top-k reads only [:context_len].
        logits=torch.empty(
            batch * next_n, n_compressed, dtype=torch.float32, device=device
        ),
        seq_blocks=seq_blocks,
    )


def _launch_paged_logits(
    inp: dict,
    grid: tuple[int, int],
    maxnreg: int,
    num_stages: int,
    q_bf16: bool = True,
) -> object:
    from vllm.v1.attention.ops.mqa_logits_triton import _fp8_paged_mqa_logits_kernel

    # Bypass @triton.autotune via .fn so maxnreg is an explicit knob.
    extra = {"maxnreg": maxnreg} if maxnreg else {}
    q = inp["q_bf16"] if q_bf16 else inp["q"]
    kv_byte, kv_scale = inp["kv_byte"], inp["kv_scale"]
    weights, block_tables, logits = inp["weights"], inp["block_tables"], inp["logits"]
    return _fp8_paged_mqa_logits_kernel.fn[grid](
        q,
        kv_byte,
        kv_scale,
        weights,
        inp["fp8_lut"],
        inp["context_lens"],
        block_tables,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_byte.stride(0),
        kv_byte.stride(1),
        kv_byte.stride(2),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        next_n=inp["next_n"],
        num_heads=CFG_INDEX_N_HEADS,
        head_dim=CFG_INDEX_HEAD_DIM,
        block_size=_PAGED_BLOCK_SIZE,
        BLOCK_H=max(16, triton.next_power_of_2(CFG_INDEX_N_HEADS)),
        BLOCK_D=triton.next_power_of_2(CFG_INDEX_HEAD_DIM),
        BLOCK_N=triton.next_power_of_2(_PAGED_BLOCK_SIZE),
        Q_IS_BF16=q_bf16,
        num_warps=4,
        num_stages=num_stages,
        **extra,
    )


def _check_paged_q_decode(inp: dict, grid: tuple[int, int]) -> str:
    """K1 gate: hoisting q's LUT decode must be bit-identical.

    Same table, same bf16 operands into the same tl.dot, same fp32
    accumulate -- so this is an equality assert, not a tolerance.
    """
    outs = {}
    for q_bf16 in (False, True):
        inp["logits"].zero_()
        _launch_paged_logits(inp, grid, 0, 2, q_bf16)
        torch.accelerator.synchronize()
        outs[q_bf16] = inp["logits"].clone()
    same = torch.equal(outs[False], outs[True])
    assert same, "hoisting the q LUT decode changed the logits"
    return f"grid={grid}: in-kernel vs hoisted q decode bit-identical"


def bench_indexer_paged(
    batches: list[int],
    n_compresseds: list[int],
    maxnregs: list[int],
    stages: list[int],
    device: torch.device,
    next_n: int = 6,
    q_decodes: list[int] | None = None,
) -> None:
    q_decodes = [0, 1] if q_decodes is None else q_decodes
    rows = []
    checks = []
    for batch, n_compressed in itertools.product(batches, n_compresseds):
        inp = _paged_logits_inputs(batch, n_compressed, device, next_n)
        grid = (batch * next_n, inp["seq_blocks"])
        checks.append(_check_paged_q_decode(inp, grid))
        for maxnreg, num_stages, q_bf16 in itertools.product(
            maxnregs, stages, q_decodes
        ):
            fn = partial(
                _launch_paged_logits, inp, grid, maxnreg, num_stages, bool(q_bf16)
            )
            n_regs = n_spills = shared = 0
            try:
                compiled = fn()
                torch.accelerator.synchronize()
                n_regs = getattr(compiled, "n_regs", 0)
                n_spills = getattr(compiled, "n_spills", 0)
                shared = getattr(compiled, "metadata", None)
                shared = getattr(shared, "shared", 0) if shared else 0
            except Exception as exc:  # noqa: BLE001 - report and keep sweeping
                print(f"  skipped ({type(exc).__name__}: {exc})")
            ctas, binder, occ = _occupancy(n_regs, shared, 4)
            us = _time_us(fn)
            ns_per_cta = float("nan") if us != us else us * 1e3 / (grid[0] * grid[1])
            rows.append(
                [
                    str(batch),
                    str(n_compressed),
                    str(n_compressed * _PAGED_COMPRESS_RATIO),
                    str(grid[0] * grid[1]),
                    "host" if q_bf16 else "kernel",
                    str(maxnreg) if maxnreg else "-",
                    str(num_stages),
                    str(n_regs),
                    str(n_spills),
                    str(shared),
                    f"{ctas} ({binder})",
                    _fmt(occ),
                    _fmt(us, ".2f"),
                    _fmt(ns_per_cta, ".1f"),
                ]
            )
    _print_table(
        f"indexer MQA logits, paged decode (H={CFG_INDEX_N_HEADS}, "
        f"D={CFG_INDEX_HEAD_DIM}, block_size={_PAGED_BLOCK_SIZE}, "
        f"next_n={next_n}, num_warps=4)",
        [
            "B",
            "n_cmp",
            "ctx",
            "CTAs",
            "q decode",
            "maxnreg",
            "stages",
            "regs",
            "spill",
            "smem",
            "CTA/SM",
            "occ%",
            "us",
            "ns/CTA",
        ],
        rows,
    )
    for line in checks:
        print(f"    {line}")


# ---------------------------------------------------------------------------
# The decode tail: ~130 launches/step of small elementwise/reduction kernels
# running at 1-2 CTAs. Whether widening their grids can pay is decided by one
# number -- how much of each launch sits above the floor that any kernel node
# costs -- so this arm measures the floor first and reports every kernel as a
# multiple of it. Timed under cudagraph replay because that is how decode runs
# them; an eager loop measures host dispatch instead.
# ---------------------------------------------------------------------------


def _graph_time_us(fn: Callable[[], object], reps: int = 50) -> float:
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(20):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / (20 * reps) * 1000


def bench_tail_launch(ms: list[int], device: torch.device) -> None:
    import vllm._custom_ops as ops

    tiny = torch.zeros(1, device=device)
    floor = _graph_time_us(lambda: tiny.add_(1.0))

    rows = []
    for m in ms:
        for label, dim in (("hidden", CFG_HIDDEN_SIZE), ("q_lora", CFG_Q_LORA_RANK)):
            x = torch.randn(m, dim, dtype=torch.bfloat16, device=device)
            w = torch.randn(dim, dtype=torch.bfloat16, device=device)
            out = torch.empty_like(x)
            res = torch.randn_like(x)
            # Call the fused ops directly: an RMSNorm module built outside a
            # real engine silently falls back to the native PyTorch composite
            # ("Priority not set for op rms_norm"), which is several kernels
            # and ~10x slower than what production runs.
            for name, fn in (
                (
                    f"rms_norm {label}",
                    lambda out=out, x=x, w=w: ops.rms_norm(out, x, w, 1e-6),
                ),
                (
                    f"fused_add_rms_norm {label}",
                    lambda x=x, res=res, w=w: ops.fused_add_rms_norm(x, res, w, 1e-6),
                ),
            ):
                us = _graph_time_us(fn)
                rows.append(
                    [
                        str(m),
                        name,
                        _fmt(us, ".2f"),
                        _fmt(us / floor, ".1f"),
                        _fmt(us - floor, ".2f"),
                    ]
                )
        src = torch.randn(
            m,
            CFG_NUM_EXPERTS_PER_TOK,
            CFG_HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=device,
        )
        dst = torch.empty(m, CFG_HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        us = _graph_time_us(lambda src=src, dst=dst: ops.moe_sum(src, dst))
        rows.append(
            [
                str(m),
                "moe_sum",
                _fmt(us, ".2f"),
                _fmt(us / floor, ".1f"),
                _fmt(us - floor, ".2f"),
            ]
        )

    _print_table(
        f"decode tail vs launch floor ({floor:.2f} us for a 1-element add)",
        ["M", "kernel", "us", "x floor", "above floor"],
        rows,
    )
    print(
        "\n'above floor' bounds what any grid change can win per launch: the\n"
        "payload is 8 KB, so what sits above the floor is fixed kernel overhead\n"
        "rather than a parallelism shortfall, and extra CTAs have nothing to do."
    )


# ---------------------------------------------------------------------------
# Unquantized bf16 GEMVs at M=1. These are the two narrow-N projections that
# stay in bf16 while everything else is block-fp8: the indexer's weights_proj
# and the MoE router gate. Both are latency items rather than bandwidth ones --
# the weights are 512 KB and 2 MB -- so what matters is launch count and how
# many CTAs the N dimension can supply.
# ---------------------------------------------------------------------------

# (name, K, N, out dtype, launches per step). Counts are per *step*, not per
# layer, because these two do not run on the same layers: the indexer is built
# only where compress_ratio == 4 (attention.py:277), which config.json puts at
# 21 layers, while every layer's ffn is a DeepseekV4MoE, so the gate runs 43x.
_BF16_GEMV_SHAPES = [
    # ReplicatedLinear(hidden_size, index_n_heads), quant_config=None -> bf16
    ("indexer.weights_proj", CFG_HIDDEN_SIZE, CFG_INDEX_N_HEADS, torch.bfloat16, 21),
    # GateLinear(hidden_size, n_routed_experts, out_dtype=fp32). On SM80 every
    # specialized tier is gated behind SM90+, so this lands on tier 6: F.linear
    # in bf16 followed by a separate .to(fp32) cast -- two launches, and the
    # fp32 output dtype is nominal because the accumulation has already been
    # rounded through bf16.
    ("moe.gate", CFG_HIDDEN_SIZE, CFG_N_ROUTED_EXPERTS, torch.float32, 43),
]


def _launch_bf16_gemv(x: torch.Tensor, w: torch.Tensor, out_dtype):
    """Exercise the shipped kernel, not a copy of it."""
    from vllm.model_executor.kernels.linear.gemv_triton import bf16_gemv

    return bf16_gemv(x, w, out_dtype)


def bench_bf16_gemv(ms: list[int], device: torch.device) -> None:
    import torch.nn.functional as F

    rows = []
    step_us: dict[str, dict[str, float]] = {}
    for m_tokens, (name, k, n, out_dtype, count) in itertools.product(
        ms, _BF16_GEMV_SHAPES
    ):
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.02
        x = torch.randn(m_tokens, k, dtype=torch.bfloat16, device=device)

        def _baseline(x=x, w=w, out_dtype=out_dtype):
            y = F.linear(x, w)
            # GateLinear tier 6 casts after the GEMM; that cast is a real
            # launch and belongs in the baseline.
            return y.to(out_dtype) if y.dtype != out_dtype else y

        ref = _baseline()
        # fp32 reference computed the way a specialized tier would, to show
        # what the bf16 round-trip costs in accuracy rather than only in time.
        exact = (x.float() @ w.float().T).flatten()

        variants: list[tuple[str, Callable[[], None]]] = [("production", _baseline)]
        variants += [("triton gemv", partial(_launch_bf16_gemv, x, w, out_dtype))]
        for label, fn in variants:
            us = _time_us(fn)
            got = fn()
            torch.accelerator.synchronize()
            got = got.float().flatten()
            rows.append(
                [
                    str(m_tokens),
                    name,
                    f"{k}x{n}",
                    label,
                    str(n),
                    _fmt(us, ".2f"),
                    f"{(got - ref.float().flatten()).abs().max().item():.1e}",
                    f"{(got - exact).abs().max().item():.1e}",
                ]
            )
            step_us.setdefault(f"M={m_tokens} {label}", {})[name] = us * count
    _print_table(
        "unquantized bf16 GEMV at M=1 (per-step launch counts)",
        ["M", "layer", "KxN", "impl", "CTAs", "us", "vs prod", "vs fp32"],
        rows,
    )
    print("\nPer-step totals (21 indexer layers + 43 gate layers):")
    for label, per_shape in sorted(step_us.items()):
        if len(per_shape) != len(_BF16_GEMV_SHAPES):
            continue
        print(f"  {label:>22}: {sum(per_shape.values()) / 1e3:.3f} ms/step")


# ---------------------------------------------------------------------------
# The three unquantized attention input GEMMs, separate vs concatenated.
# `attn_gemm_parallel_execute` runs four GEMMs off the same hidden_states; the
# three bf16 ones read the same x and can be one GEMM over a concatenated
# weight. Row counts come from the checkpoint (layers.2.attn.*): compressor
# 2 x [1024, 4096], indexer compressor 2 x [256, 4096], weights_proj [64, 4096].
#
# Each timed iteration uses a DIFFERENT weight set. One 20.5 MB set replayed in
# a loop becomes L2-resident after the first pass, which is not what a decode
# step does -- 43 layers each stream their own weights once -- and that alone
# moves the answer (the L2-hot version reads 28.2 us for the separate arm
# against 31.6 cold).
# ---------------------------------------------------------------------------

_ATTN_INPUT_NS = (2048, 512, 64)  # compressor, indexer compressor, weights_proj
_ATTN_INPUT_ROTATE = 8


def bench_attn_input_gemm(ms: list[int], device: torch.device) -> None:
    from vllm.model_executor.kernels.linear.gemv_triton import bf16_gemv

    sets = []
    for _ in range(_ATTN_INPUT_ROTATE):
        parts = [
            torch.randn(n, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
            for n in _ATTN_INPUT_NS
        ]
        sets.append((parts, torch.cat(parts, dim=0).contiguous()))

    def separate(x: torch.Tensor) -> None:
        for parts, _ in sets:
            torch.mm(x, parts[0].T, out_dtype=torch.float32)
            # Production routes weights_proj through the M<=8 Triton GEMV.
            if x.shape[0] <= 8:
                bf16_gemv(x, parts[2])
            else:
                torch.mm(x, parts[2].T)
            torch.mm(x, parts[1].T, out_dtype=torch.float32)

    def merged_mm(x: torch.Tensor) -> None:
        for _, w in sets:
            torch.mm(x, w.T, out_dtype=torch.float32)

    def merged_gemv(x: torch.Tensor) -> None:
        for _, w in sets:
            bf16_gemv(x, w, out_dtype=torch.float32)

    rows = []
    for m_tokens in ms:
        x = torch.randn(m_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        per_iter = {}
        for label, fn in (
            ("separate x3", separate),
            ("merged cuBLAS", merged_mm),
            ("merged Triton GEMV", merged_gemv),
        ):
            us = _time_us(partial(fn, x)) / _ATTN_INPUT_ROTATE
            per_iter[label] = us
            rows.append(
                [
                    str(m_tokens),
                    label,
                    _fmt(us, ".2f"),
                    _fmt(
                        (per_iter["separate x3"] - us) * _ATTN_INPUT_LAYERS / 1e3, ".3f"
                    ),
                ]
            )
    ns = "+".join(map(str, _ATTN_INPUT_NS))
    _print_table(
        f"attention input GEMMs (K={HIDDEN_SIZE}, N={ns} = {sum(_ATTN_INPUT_NS)}, "
        f"{_ATTN_INPUT_ROTATE} rotated weight sets)",
        ["M", "impl", "us/layer", "ms/step saved"],
        rows,
    )
    print(
        f"\nms/step scales the per-layer delta by the {_ATTN_INPUT_LAYERS} ratio-4\n"
        "layers, which are the only ones carrying all three GEMMs."
    )


# ---------------------------------------------------------------------------
# Dense Marlin fp8 at M=1: the largest single decode component (~3.22 ms/step,
# 28%), streaming 26.2 MB/layer of weights at ~15% of DRAM peak. Marlin runs
# 200 threads with 145 KB smem/CTA = 1 CTA/SM, and at M=1 the smem staging buys
# nothing (weights stream once, zero reuse) -- the kernel's structure is wrong
# for GEMV-shaped work. Arms: production Marlin; dequant-to-bf16 at load +
# cuBLAS (Route C control, costs +1.13 GB VRAM/rank); fused LUT/ALU-decode
# bf16 GEMV in Triton drawing parallelism from N (Route B primary).
# ---------------------------------------------------------------------------

# The six per-layer per-rank M=1 GEMM shapes that go through dense Marlin fp8
# at TP=8 (wo_a is exempt: consumed as raw block-fp8 by the attention bmm).
# (name, K, N, launches per layer); every dimension traces to config.json.
_DENSE_GEMM_SHAPES = [
    # hidden -> q_lora_rank + head_dim, fused ReplicatedLinear
    ("fused_wqa_wkv", CFG_HIDDEN_SIZE, CFG_Q_LORA_RANK + CFG_HEAD_DIM, 1),
    # q_lora_rank -> (num_attention_heads / TP) * head_dim, and
    # (o_groups * o_lora_rank) / TP -> hidden: same (K, N) at TP=8
    (
        "wq_b+wo_b",
        CFG_Q_LORA_RANK,
        CFG_NUM_ATTENTION_HEADS // TP_SIZE * CFG_HEAD_DIM,
        2,
    ),
    # q_lora_rank -> index_n_heads * index_head_dim, ReplicatedLinear
    ("indexer.wq_b", CFG_Q_LORA_RANK, CFG_INDEX_N_HEADS * CFG_INDEX_HEAD_DIM, 1),
    # hidden -> 2 * moe_intermediate_size / TP (shared expert gate_up)
    ("shared_gate_up", CFG_HIDDEN_SIZE, 2 * CFG_MOE_INTERMEDIATE_SIZE // TP_SIZE, 1),
    # moe_intermediate_size / TP -> hidden (shared expert down)
    ("shared_down", CFG_MOE_INTERMEDIATE_SIZE // TP_SIZE, CFG_HIDDEN_SIZE, 1),
]


def _fp8_block_quant(
    w: torch.Tensor, qb: int = CFG_WEIGHT_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a (N, K) bf16 weight to block-fp8 exactly like the checkpoint:
    one fp32 scale per qb x qb block, e4m3 payload."""
    n, k = w.shape
    assert n % qb == 0 and k % qb == 0
    wv = w.float().view(n // qb, qb, k // qb, qb)
    amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / 448.0
    q = (wv / scales).clamp(-448.0, 448.0)
    w_fp8 = q.view(n, k).to(torch.float8_e4m3fn)
    return w_fp8, scales.view(n // qb, k // qb).contiguous()


def _fp8_block_dequant(w_fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    qb = CFG_WEIGHT_BLOCK_SIZE
    n, k = w_fp8.shape
    wv = w_fp8.float().view(n // qb, qb, k // qb, qb)
    return (wv * scales.view(n // qb, 1, k // qb, 1)).view(n, k).bfloat16()


def _make_marlin_layer(
    w_fp8: torch.Tensor, scales: torch.Tensor, device: torch.device
) -> torch.nn.Module:
    """Reproduce MarlinFP8ScaledMMLinearKernel.process_weights_after_loading
    for a block-quant layer (size_k_first=False)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_fp8_layer_for_marlin,
    )

    n, k = w_fp8.shape
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_fp8, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
    layer.orig_dtype = torch.bfloat16
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.weight_block_size = [CFG_WEIGHT_BLOCK_SIZE, CFG_WEIGHT_BLOCK_SIZE]
    weight, weight_scale_inv = process_fp8_weight_block_strategy(
        layer.weight, layer.weight_scale_inv
    )
    layer.weight = torch.nn.Parameter(weight.data, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(
        weight_scale_inv.data, requires_grad=False
    )
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    return layer


def _import_gemv_triton():
    from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32, _decode_fp8_lut

    @triton.jit
    def _fp8_block_gemv_kernel(
        x_ptr,  # [M, K] bf16
        w_ptr,  # [N, K] fp8-e4m3 bytes
        scale_ptr,  # [N // QB, K // QB] fp32
        out_ptr,  # [M, N] bf16
        lut_ptr,
        stride_x_m,
        stride_w_n,
        stride_s_n,
        stride_o_m,
        K,
        N,
        QB: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_LUT: tl.constexpr,
    ):
        # One CTA owns BLOCK_N output rows for one m: parallelism comes from N
        # (the anti-Marlin shape) and nothing is staged through smem -- at M=1
        # each weight byte is used exactly once, so staging cannot pay.
        pid_n = tl.program_id(0)
        m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_k = tl.arange(0, QB)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, QB):
            x = tl.load(x_ptr + m * stride_x_m + k0 + offs_k).to(tl.float32)
            w_u8 = tl.load(
                w_ptr + offs_n[:, None] * stride_w_n + (k0 + offs_k)[None, :],
                mask=mask_n[:, None],
                other=0,
            )
            if USE_LUT:
                w = _decode_fp8_lut(w_u8, False, lut_ptr).to(tl.float32)
            else:
                w = _decode_fp8_f32(w_u8, False)
            # The K quant-block boundary aligns with the loop step, so each
            # iteration touches exactly one scale per output row.
            s = tl.load(
                scale_ptr + (offs_n // QB) * stride_s_n + k0 // QB,
                mask=mask_n,
                other=0.0,
            )
            acc += tl.sum(w * x[None, :], axis=1) * s
        tl.store(out_ptr + m * stride_o_m + offs_n, acc.to(tl.bfloat16), mask=mask_n)

    @triton.jit
    def _fp8_block_gemv_dot_kernel(
        x_ptr,  # [M, K] bf16, M <= 16
        w_ptr,  # [N, K] fp8-e4m3 bytes
        scale_ptr,  # [N // QB, K // QB] fp32
        out_ptr,  # [M, N] bf16
        stride_x_m,
        stride_w_n,
        stride_s_n,
        stride_o_m,
        M,
        K,
        N,
        QB: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # v2: bit-shift fp8->fp16 decode (3 int ops/elem; exact incl. e4m3
        # denormals since every e4m3 value lands normal in fp16 after the 2^8
        # rebias, folded into the block scale) and tl.dot so the MAC rides the
        # idle tensor pipe. NaN weight bytes decode to finite garbage; block-
        # quantized checkpoints contain no NaN weights. M<=16 rides the MMA
        # padding for free, so one CTA covers the whole M<=8 dispatch range.
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_m = tl.arange(0, 16)
        mask_m = offs_m < M
        offs_k = tl.arange(0, QB)
        acc = tl.zeros([16, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, QB):
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x_m + (k0 + offs_k)[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            u = tl.load(
                w_ptr + offs_n[:, None] * stride_w_n + (k0 + offs_k)[None, :],
                mask=mask_n[:, None],
                other=0,
            ).to(tl.uint16)
            w = (
                (((u & 0x80) << 8) | ((u & 0x7F) << 7))
                .to(tl.float16, bitcast=True)
                .to(tl.bfloat16)
            )
            s = tl.load(
                scale_ptr + (offs_n // QB) * stride_s_n + k0 // QB,
                mask=mask_n,
                other=0.0,
            )
            acc += tl.dot(x, tl.trans(w)) * (s * 256.0)[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * stride_o_m + offs_n[None, :],
            acc.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _fp8_block_gemv_dot_split_kernel(
        x_ptr,  # [M, K] bf16, M <= 16
        w_ptr,  # [N, K] fp8-e4m3 bytes
        scale_ptr,  # [N // QB, K // QB] fp32
        part_ptr,  # [SPLIT_K, 16, N] fp32
        stride_x_m,
        stride_w_n,
        stride_s_n,
        stride_p_s,
        stride_p_m,
        M,
        K,
        N,
        QB: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        # Split-K flavour for the wide-K narrow-N shapes whose unsplit grid is
        # 16-48 CTAs: same decode/dot body, each split owns K/SPLIT_K, partials
        # reduced by _gemv_reduce_kernel (batch-1 grid starvation; split-K is
        # the in-tree existence proof that splitting a reduction fixes it).
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_m = tl.arange(0, 16)
        mask_m = offs_m < M
        offs_k = tl.arange(0, QB)
        acc = tl.zeros([16, BLOCK_N], dtype=tl.float32)
        k_per_split = K // SPLIT_K
        k_lo = pid_s * k_per_split
        for k0 in range(k_lo, k_lo + k_per_split, QB):
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x_m + (k0 + offs_k)[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            u = tl.load(
                w_ptr + offs_n[:, None] * stride_w_n + (k0 + offs_k)[None, :],
                mask=mask_n[:, None],
                other=0,
            ).to(tl.uint16)
            w = (
                (((u & 0x80) << 8) | ((u & 0x7F) << 7))
                .to(tl.float16, bitcast=True)
                .to(tl.bfloat16)
            )
            s = tl.load(
                scale_ptr + (offs_n // QB) * stride_s_n + k0 // QB,
                mask=mask_n,
                other=0.0,
            )
            acc += tl.dot(x, tl.trans(w)) * (s * 256.0)[None, :]
        tl.store(
            part_ptr
            + pid_s * stride_p_s
            + offs_m[:, None] * stride_p_m
            + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _gemv_reduce_kernel(
        part_ptr,  # [SPLIT_K, 16, N] fp32
        out_ptr,  # [M, N] bf16
        stride_p_s,
        stride_p_m,
        stride_o_m,
        M,
        N,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for s in tl.static_range(SPLIT_K):
            acc += tl.load(
                part_ptr + s * stride_p_s + m * stride_p_m + offs_n,
                mask=mask_n,
                other=0.0,
            )
        tl.store(out_ptr + m * stride_o_m + offs_n, acc.to(tl.bfloat16), mask=mask_n)

    return (
        _fp8_block_gemv_kernel,
        _fp8_block_gemv_dot_kernel,
        _fp8_block_gemv_dot_split_kernel,
        _gemv_reduce_kernel,
    )


def _launch_gemv(
    kernel,
    x: torch.Tensor,
    w_u8: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    lut: torch.Tensor,
    block_n: int,
    num_warps: int,
    use_lut: bool,
) -> None:
    m, k = x.shape
    n = w_u8.shape[0]
    kernel[(triton.cdiv(n, block_n), m)](
        x,
        w_u8,
        scales,
        out,
        lut,
        x.stride(0),
        w_u8.stride(0),
        scales.stride(0),
        out.stride(0),
        k,
        n,
        QB=CFG_WEIGHT_BLOCK_SIZE,
        BLOCK_N=block_n,
        USE_LUT=use_lut,
        num_warps=num_warps,
    )


def _launch_gemv_dot(
    kernel,
    x: torch.Tensor,
    w_u8: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> None:
    m, k = x.shape
    n = w_u8.shape[0]
    kernel[(triton.cdiv(n, block_n),)](
        x,
        w_u8,
        scales,
        out,
        x.stride(0),
        w_u8.stride(0),
        scales.stride(0),
        out.stride(0),
        m,
        k,
        n,
        QB=CFG_WEIGHT_BLOCK_SIZE,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


def _launch_gemv_split(
    split_kernel,
    reduce_kernel,
    x: torch.Tensor,
    w_u8: torch.Tensor,
    scales: torch.Tensor,
    part: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    split_k: int,
) -> None:
    m, k = x.shape
    n = w_u8.shape[0]
    split_kernel[(triton.cdiv(n, block_n), split_k)](
        x,
        w_u8,
        scales,
        part,
        x.stride(0),
        w_u8.stride(0),
        scales.stride(0),
        part.stride(0),
        part.stride(1),
        m,
        k,
        n,
        QB=CFG_WEIGHT_BLOCK_SIZE,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        num_warps=4,
    )
    reduce_kernel[(triton.cdiv(n, 256), m)](
        part,
        out,
        part.stride(0),
        part.stride(1),
        out.stride(0),
        m,
        n,
        BLOCK_N=256,
        SPLIT_K=split_k,
        num_warps=4,
    )


def bench_dense_gemv(ms: list[int], block_ns: list[int], device: torch.device) -> None:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
    )
    from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut

    kernel, dot_kernel, split_kernel, reduce_kernel = _import_gemv_triton()
    lut = get_e4m3fn_bf16_lut(device)
    rows = []
    # Best us per (arm, shape) for the per-step summary.
    step_us: dict[str, dict[str, float]] = {}
    for name, k, n, _count in _DENSE_GEMM_SHAPES:
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.02
        w_fp8, scales = _fp8_block_quant(w)
        w_dq = _fp8_block_dequant(w_fp8, scales)
        marlin_layer = _make_marlin_layer(w_fp8.clone(), scales.clone(), device)
        w_u8 = w_fp8.view(torch.uint8)
        for m_tokens in ms:
            x = torch.randn(m_tokens, k, dtype=torch.bfloat16, device=device)
            out = torch.empty(m_tokens, n, dtype=torch.bfloat16, device=device)
            ref = (x.float() @ w_dq.float().t()).bfloat16()

            def run_marlin(x=x, layer=marlin_layer, n=n, k=k) -> torch.Tensor:
                return apply_fp8_marlin_linear(
                    x,
                    layer.weight,
                    layer.weight_scale_inv,
                    layer.workspace,
                    size_n=n,
                    size_k=k,
                    bias=None,
                )

            def run_cublas(x=x, w_dq=w_dq) -> torch.Tensor:
                return torch.nn.functional.linear(x, w_dq, None)

            variants: list[tuple[str, Callable[[], None], torch.Tensor | None]] = [
                ("marlin", run_marlin, run_marlin()),
                ("cublas-bf16", run_cublas, run_cublas()),
            ]
            # The Triton GEMV arms are M<=16 kernels by construction (the dot
            # tile and the split partial buffer are 16 rows); past that they
            # write garbage or fault, so only the library arms run.
            gemv_block_ns = block_ns if m_tokens <= 16 else []
            for block_n, use_lut in itertools.product(gemv_block_ns, (True, False)):
                label = f"gemv-{'lut' if use_lut else 'alu'}-bn{block_n}"
                fn = partial(
                    _launch_gemv, kernel, x, w_u8, scales, out, lut, block_n, 4, use_lut
                )
                fn()
                variants.append((label, fn, out.clone()))
            for block_n in gemv_block_ns:
                label = f"gemv-dot-bn{block_n}"
                fn = partial(
                    _launch_gemv_dot, dot_kernel, x, w_u8, scales, out, block_n, 4
                )
                fn()
                variants.append((label, fn, out.clone()))
            for block_n, split_k in itertools.product(gemv_block_ns, (2, 4, 8)):
                if k // split_k % CFG_WEIGHT_BLOCK_SIZE != 0:
                    continue
                part = torch.empty(split_k, 16, n, dtype=torch.float32, device=device)
                label = f"gemv-split{split_k}-bn{block_n}"
                fn = partial(
                    _launch_gemv_split,
                    split_kernel,
                    reduce_kernel,
                    x,
                    w_u8,
                    scales,
                    part,
                    out,
                    block_n,
                    split_k,
                )
                fn()
                variants.append((label, fn, out.clone()))

            weight_bytes = n * k  # fp8 payload; what an ideal GEMV must stream
            for label, fn, got in variants:
                err = (
                    float("nan")
                    if got is None
                    else (got.float() - ref.float()).abs().max().item()
                )
                us = _time_us(fn)
                gbs = float("nan") if us != us else weight_bytes / (us * 1e-6) / 1e9
                if m_tokens == 1 and us == us:
                    arm = label.split("-bn")[0]
                    if arm.startswith("gemv-split"):
                        arm = "gemv-split"
                    for key in (arm, "best-hybrid"):
                        best = step_us.setdefault(key, {})
                        prev = best.get(name)
                        best[name] = us if prev is None or us < prev else prev
                rows.append(
                    [
                        name,
                        str(m_tokens),
                        f"{k}x{n}",
                        label,
                        _fmt(us),
                        _fmt(gbs, ".0f"),
                        f"{err:.1e}",
                    ]
                )
    _print_table(
        "dense Marlin fp8 vs GEMV routes (per-rank TP=8 shapes)",
        ["layer", "M", "KxN", "impl", "us", "GB/s", "max|err|"],
        rows,
    )
    shape_counts = {name: count for name, _, _, count in _DENSE_GEMM_SHAPES}
    print(
        f"\nPer-step M=1 sum over {CFG_NUM_HIDDEN_LAYERS} layers "
        "(best config per arm; acceptance is beating marlin by >= 1.5 ms):"
    )
    for label, per_shape in sorted(step_us.items()):
        if set(per_shape) != set(shape_counts):
            print(f"  {label:>12}: incomplete ({sorted(per_shape)})")
            continue
        total_ms = (
            sum(per_shape[s] * shape_counts[s] for s in per_shape)
            * CFG_NUM_HIDDEN_LAYERS
            / 1e3
        )
        print(f"  {label:>12}: {total_ms:.2f} ms/step")


# ---------------------------------------------------------------------------
# Routed MoE experts (MXFP4 Marlin).
#
# What production dispatches, taken from the p6 server log rather than inferred:
#   [mxfp4.py:626] Using 'MARLIN' Mxfp4 MoE backend.
#   [mxfp4.py:1727] Using MarlinExperts
# so the kernel family is `moe_wna16_marlin_gemm`, two calls per layer (w13 then
# w2) sharing one workspace. MegaMoE is NOT live: it needs
# --enable-expert-parallel plus moe_backend=deep_gemm_mega_moe, and the serve
# command sets neither, so experts are TP-sharded over the intermediate
# dimension (N = moe_intermediate_size / TP) with all E experts on every rank.
#
# Weights are built through the production prep function
# (`prepare_moe_mxfp4_layer_for_marlin`), not a hand-rolled repack, so the
# layouts, the group size (32) and the e8m0 scale decode are the shipped ones.
#
# Measured shape of the cost curve (A100, TP=8 shapes above), because "the MoE
# is bandwidth-bound" is true at exactly one place on it:
#
#     M       weight-streaming rate    what limits it
#     1       245 GB/s                 latency: 6 blocks of work, DRAM 11-17%
#                                      of peak, ~40 us of the 43 us call is
#                                      fixed cost
#     64      1310-1375 GB/s           bandwidth, against 1457 GB/s measured
#                                      for a large streaming read on this box
#     2048    ~450 GB/s                compute: gemm1 reaches 53% of the bf16
#                                      tensor pipe, so weights amortize and
#                                      the rate stops being the metric
#
# So a projection that carries one regime's achieved bandwidth into another is
# wrong by up to 5.6x here, in both directions. The size sweep is what
# separates the three, and it is cheap -- run it before believing any single
# operating point. The same effect shows up whenever per-CTA work shrinks: the
# DSpark Markov GEMM drops 1457 -> 807 GB/s purely from being sharded 8 ways.
# ---------------------------------------------------------------------------

CFG_SWIGLU_LIMIT = 10.0  # swiglu_limit
MOE_N_PER_RANK = CFG_MOE_INTERMEDIATE_SIZE // TP_SIZE  # 256


def _make_mxfp4_marlin_experts(
    num_experts: int, n: int, k: int, device: torch.device, dtype: torch.dtype
):
    """Production-shaped MXFP4 expert weights, repacked the production way."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )

    w13 = torch.randint(
        0, 255, (num_experts, 2 * n, k // 2), dtype=torch.uint8, device=device
    )
    w2 = torch.randint(
        0, 255, (num_experts, k, n // 2), dtype=torch.uint8, device=device
    )
    # e8m0 scales around 1.0 (127) so dequantized weights stay in a sane range.
    w13_scale = torch.randint(
        120, 134, (num_experts, 2 * n, k // 32), dtype=torch.uint8, device=device
    )
    w2_scale = torch.randint(
        120, 134, (num_experts, k, n // 32), dtype=torch.uint8, device=device
    )

    class _Layer:
        params_dtype = dtype

    return prepare_moe_mxfp4_layer_for_marlin(
        _Layer(), w13, w2, w13_scale, w2_scale, None, None
    )


def _moe_routing(
    m: int,
    num_experts: int,
    topk: int,
    skew: str,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """topk_ids/topk_weights with a controllable expert-load distribution.

    The real distribution is not knowable from a microbench -- DSv4 routes
    partly by a token-id hash table (`gate.tid2eid`, num_hash_layers=3) -- so
    this parameterises it and the sweep reports the sensitivity instead of
    asserting one shape. What the kernel actually sees is the block count from
    `moe_align_block_size`: sum(ceil(tokens_e / block_m)), which grows with
    skew because a lightly-loaded expert still costs a whole block.
    """
    if skew == "uniform":
        # Every expert equally loaded (the best case for block packing).
        flat = torch.arange(m * topk, device=device) % num_experts
        topk_ids = flat.view(m, topk).to(torch.int32)
    elif skew == "random":
        topk_ids = torch.randint(
            0,
            num_experts,
            (m, topk),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
    elif skew == "zipf":
        # Rank-frequency ~ 1/rank over experts, then a random permutation so no
        # expert index is privileged.
        ranks = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
        probs = 1.0 / ranks
        probs = probs / probs.sum()
        perm = torch.randperm(num_experts, device=device, generator=generator)
        draw = torch.multinomial(probs, m * topk, replacement=True, generator=generator)
        topk_ids = perm[draw].view(m, topk).to(torch.int32)
    elif skew == "hot":
        # Degenerate control: every token to the same expert.
        topk_ids = torch.zeros((m, topk), dtype=torch.int32, device=device)
    else:
        raise ValueError(f"unknown skew {skew!r}")
    topk_weights = torch.rand(
        (m, topk), dtype=torch.float32, device=device, generator=generator
    )
    return topk_ids, topk_weights


def _moe_block_stats(
    topk_ids: torch.Tensor, num_experts: int, block_m: int
) -> tuple[int, int]:
    """(blocks the kernel will run, distinct experts touched)."""
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)
    blocks = int(torch.ceil(counts.float() / block_m).sum().item())
    return blocks, int((counts > 0).sum().item())


def _moe_selected_block_m(m: int, topk: int, num_experts: int) -> int:
    """Mirror of the ladder inside fused_marlin_moe (marlin_moe.py:330-333).

    Copied rather than imported because the production function computes it
    inline and does not expose it; the block_size_m sweep below prints which
    rung the heuristic lands on so a drift shows up as a mislabelled row.
    """
    for block_size_m in [8, 16, 32, 48, 64]:
        if m * topk / num_experts / block_size_m < 0.9:
            break
    return block_size_m


def _bench_moe_decompose(
    tokens,
    skew,
    num_experts,
    topk,
    n,
    k,
    w13,
    w2,
    w13_scale,
    w2_scale,
    workspace,
    generator,
    device,
    dtype,
) -> None:
    """Split one expert call into its five ops.

    At M=1 the whole call is nearly fixed cost (doubling the tokens adds a few
    percent), so the interesting question is not the GEMM's efficiency but
    which op holds the time. Everything here is timed on the same tensors the
    fused path builds, so the parts are comparable to the whole.
    """
    rows = []
    for m in tokens:
        block_m = _moe_selected_block_m(m, topk, num_experts)
        for name, fn in _moe_call_parts(
            m,
            block_m,
            num_experts,
            topk,
            n,
            k,
            skew,
            w13,
            w2,
            w13_scale,
            w2_scale,
            workspace,
            generator,
            device,
            dtype,
        ).items():
            rows.append([str(m), str(block_m), name, _fmt(_time_us(fn))])
    _print_table(
        f"one expert call decomposed (skew={skew})",
        ["M", "blk_m", "op", "us"],
        rows,
    )


def _moe_call_parts(
    m,
    block_m,
    num_experts,
    topk,
    n,
    k,
    skew,
    w13,
    w2,
    w13_scale,
    w2_scale,
    workspace,
    generator,
    device,
    dtype,
):
    """The five ops of one expert call, as separately timeable callables."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        apply_moe_activation,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
    from vllm.scalar_type import scalar_types

    quant_type = scalar_types.float4_e2m1f
    ids, weights = _moe_routing(m, num_experts, topk, skew, device, generator)
    x = torch.randn((m, k), dtype=dtype, device=device) / 10
    sorted_ids, expert_ids, num_padded = moe_align_block_size(
        ids, block_m, num_experts, None, ignore_invalid_experts=True
    )
    c1 = torch.empty((m * topk, 2 * n), device=device, dtype=dtype)
    c2 = torch.empty((m * topk, n), device=device, dtype=dtype)
    c3 = torch.empty((m * topk, k), device=device, dtype=dtype)
    out = torch.empty((m, k), device=device, dtype=dtype)

    def gemm1():
        ops.moe_wna16_marlin_gemm(
            x,
            c1,
            w13,
            None,
            w13_scale,
            None,
            None,
            None,
            None,
            None,
            workspace,
            sorted_ids,
            expert_ids,
            num_padded,
            weights,
            moe_block_size=block_m,
            top_k=topk,
            mul_topk_weights=False,
            b_q_type=quant_type,
            size_m=m,
            size_n=2 * n,
            size_k=k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    def gemm2():
        ops.moe_wna16_marlin_gemm(
            c2,
            c3,
            w2,
            None,
            w2_scale,
            None,
            None,
            None,
            None,
            None,
            workspace,
            sorted_ids,
            expert_ids,
            num_padded,
            weights,
            moe_block_size=block_m,
            top_k=1,
            mul_topk_weights=True,
            b_q_type=quant_type,
            size_m=m * topk,
            size_n=k,
            size_k=n,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    return {
        "moe_align_block_size": lambda: moe_align_block_size(
            ids, block_m, num_experts, None, ignore_invalid_experts=True
        ),
        "gemm1 (w13)": gemm1,
        "activation": lambda: apply_moe_activation(
            MoEActivation.SILU, c2, c1, clamp_limit=CFG_SWIGLU_LIMIT
        ),
        "gemm2 (w2)": gemm2,
        "moe_sum": lambda: torch.sum(c3.view(-1, topk, k), dim=1, out=out),
    }


def _moe_block_m_runner(
    fused,
    align,
    scalar_types,
    x,
    ids,
    weights,
    block_m,
    num_experts,
    topk,
    n,
    k,
    w13,
    w2,
    w13_scale,
    w2_scale,
    workspace,
    cache13,
    cache2,
):
    """One block_size_m arm, bound outside the sweep loop."""
    sorted_ids, expert_ids, num_padded = align(
        ids, block_m, num_experts, None, ignore_invalid_experts=True
    )

    def run():
        fused(
            hidden_states=x,
            w1=w13,
            w2=w2,
            bias1=None,
            bias2=None,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            topk_weights=weights,
            num_topk=topk,
            quant_type=scalar_types.float4_e2m1f,
            apply_router_weight_on_input=False,
            expert_map=None,
            block_size_m=block_m,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_padded,
            topk_ids=ids,
            workspace=workspace,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
            clamp_limit=CFG_SWIGLU_LIMIT,
        )

    return run


def _bench_moe_block_m(
    tokens,
    skews,
    num_experts,
    topk,
    n,
    k,
    w13,
    w2,
    w13_scale,
    w2_scale,
    workspace,
    generator,
    device,
    dtype,
) -> None:
    """Sweep the block size the heuristic picks for us.

    `fused_marlin_moe` chooses block_size_m from a fixed ladder with a
    "TODO: tune this further for specific models" next to it, and the choice
    sets the block count -- which the sweep above shows is what prefill time is
    proportional to. Bigger blocks waste padding on a lightly loaded expert;
    smaller blocks re-read that expert's weights once per block. Both sides are
    real, so the crossover is a measurement.
    """
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        _fused_marlin_moe,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
    from vllm.scalar_type import scalar_types

    rows = []
    for m in tokens:
        for skew in skews:
            chosen = _moe_selected_block_m(m, topk, num_experts)
            ids, weights = _moe_routing(m, num_experts, topk, skew, device, generator)
            x = torch.randn((m, k), dtype=dtype, device=device) / 10
            cache13 = torch.empty(
                (m * topk * max(2 * n, k),), dtype=dtype, device=device
            )
            cache2 = torch.empty((m * topk, n), dtype=dtype, device=device)
            for block_m in (8, 16, 32, 48, 64):
                blocks, _ = _moe_block_stats(ids, num_experts, block_m)
                run = _moe_block_m_runner(
                    _fused_marlin_moe,
                    moe_align_block_size,
                    scalar_types,
                    x,
                    ids,
                    weights,
                    block_m,
                    num_experts,
                    topk,
                    n,
                    k,
                    w13,
                    w2,
                    w13_scale,
                    w2_scale,
                    workspace,
                    cache13,
                    cache2,
                )
                rows.append(
                    [
                        str(m),
                        skew,
                        str(block_m) + ("*" if block_m == chosen else ""),
                        str(blocks),
                        _fmt(_time_us(run)),
                    ]
                )
    _print_table(
        "block_size_m sweep (* = what the in-tree heuristic picks)",
        ["M", "skew", "blk_m", "blocks", "us"],
        rows,
    )


def _moe_rotation_runner(
    fused,
    scalar_types,
    routings,
    hidden,
    num_experts,
    w13,
    w2,
    w13_scale,
    w2_scale,
    workspace,
    cache13,
    cache2,
    out,
):
    """One timing arm over a rotation of routings, bound outside the loop."""

    def run():
        for (ids, weights), x in zip(routings, hidden):
            fused(
                hidden_states=x,
                w1=w13,
                w2=w2,
                bias1=None,
                bias2=None,
                w1_scale=w13_scale,
                w2_scale=w2_scale,
                topk_weights=weights,
                topk_ids=ids,
                quant_type_id=scalar_types.float4_e2m1f.id,
                global_num_experts=num_experts,
                workspace=workspace,
                intermediate_cache13=cache13,
                intermediate_cache2=cache2,
                output=out,
                clamp_limit=CFG_SWIGLU_LIMIT,
            )

    return run


def bench_moe_experts(
    tokens: list[int],
    skews: list[str],
    num_experts: int,
    topk: int,
    n: int,
    k: int,
    rotations: int,
    device: torch.device,
) -> None:
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.scalar_type import scalar_types

    dtype = torch.bfloat16
    w13, w2, w13_scale, w2_scale, _, _ = _make_mxfp4_marlin_experts(
        num_experts, n, k, device, dtype
    )
    workspace = marlin_make_workspace_new(device, 4)
    generator = torch.Generator(device=device).manual_seed(0)
    # Bytes of packed weight per expert, both GEMMs: this is the quantity the
    # weight-streaming hypothesis says the kernel is paying for.
    bytes_per_expert = (2 * n * k + k * n) // 2 + (2 * n * k // 32 + k * n // 32) * 2

    rows = []
    for m in tokens:
        for skew in skews:
            # Rotate routings (trap #7): at M=1 a single routing touches ~6
            # experts, a few MB, which sits in L2 across replays and reads
            # far faster than the first touch ever does.
            routings = [
                _moe_routing(m, num_experts, topk, skew, device, generator)
                for _ in range(rotations)
            ]
            hidden = [
                torch.randn((m, k), dtype=dtype, device=device) / 10
                for _ in range(rotations)
            ]
            block_m = _moe_selected_block_m(m, topk, num_experts)
            stats = [_moe_block_stats(ids, num_experts, block_m) for ids, _ in routings]
            blocks = sum(s[0] for s in stats) / len(stats)
            touched = sum(s[1] for s in stats) / len(stats)

            # Production hoists these: MarlinExperts sizes them in
            # workspace_shapes() and the modular kernel hands them in, so a
            # bench that leaves them None measures allocator work the server
            # never does (8.8 of 44.1 us at M=1, measured).
            cache13 = torch.empty(
                (m * topk * max(2 * n, k),), dtype=dtype, device=device
            )
            cache2 = torch.empty((m * topk, n), dtype=dtype, device=device)
            out = torch.empty((m, k), dtype=dtype, device=device)

            run = _moe_rotation_runner(
                fused_marlin_moe,
                scalar_types,
                routings,
                hidden,
                num_experts,
                w13,
                w2,
                w13_scale,
                w2_scale,
                workspace,
                cache13,
                cache2,
                out,
            )

            us = _time_us(run) / rotations
            # Weights the kernel must read at least once per call.
            gbs = touched * bytes_per_expert / (us * 1e-6) / 1e9 if us == us else 0.0
            rows.append(
                [
                    str(m),
                    skew,
                    str(block_m),
                    f"{touched:.0f}",
                    f"{blocks:.0f}",
                    _fmt(us),
                    _fmt(gbs, ".0f"),
                    _fmt(us * CFG_NUM_HIDDEN_LAYERS / 1e3, ".2f"),
                ]
            )
    _print_table(
        f"MXFP4 Marlin MoE experts (E={num_experts}, topk={topk}, "
        f"N={n}, K={k}, per-rank TP={TP_SIZE})",
        ["M", "skew", "blk_m", "experts", "blocks", "us", "wt GB/s", "ms/step x43"],
        rows,
    )
    _bench_moe_decompose(
        tokens,
        skews[0],
        num_experts,
        topk,
        n,
        k,
        w13,
        w2,
        w13_scale,
        w2_scale,
        workspace,
        generator,
        device,
        dtype,
    )
    _bench_moe_block_m(
        tokens,
        skews,
        num_experts,
        topk,
        n,
        k,
        w13,
        w2,
        w13_scale,
        w2_scale,
        workspace,
        generator,
        device,
        dtype,
    )
    print(
        "\n'experts' is how many distinct experts a call touches and 'wt GB/s'\n"
        "prices only their packed weights, so it is the weight-streaming rate --\n"
        "compare it against a large streaming read on this box, not against the\n"
        "HBM datasheet number."
    )


KERNELS = (
    "moe-experts",
    "sparse-decode",
    "sparse-prefill",
    "sparse-decode-c128",
    "sparse-prefill-c128",
    "bf16-gemv",
    "tail-launch",
    "prenorm-gemm",
    "mhc-pre",
    "dequant-gather",
    "indexer-logits",
    "indexer-paged",
    "attn-input-gemm",
    "dense-gemv",
)

# What a batch-1 decode step is made of, and where each part can be timed. This
# exists so that summing the sweeps above is never mistaken for a decomposition
# of the whole step: the covered rows are a minority of decode time, and a total
# that silently omits Marlin and MoE would understate the step by roughly half.
_DECODE_COVERAGE = [
    ("sparse-MLA decode", "~18%", "here: --kernel sparse-decode"),
    ("mHC pre big-fuse", "~6%", "here: --kernel mhc-pre"),
    ("dense Marlin fp8", "~26%", "here: --kernel dense-gemv (vs GEMV routes)"),
    ("MoE experts (Marlin)", "~10%", "here: --kernel moe-experts"),
    ("cuBLAS GEMV", "~13%", "not covered: plain torch matmul at M=1"),
    ("indexer MQA logits", "~11%", "here: --kernel indexer-logits (prefill path)"),
    ("mHC post / fused-post-pre", "~3%", "NOT COVERED"),
    ("norms, RoPE, elementwise", "rest", "NOT COVERED - ~235 small launches"),
]


def print_decode_coverage() -> None:
    _print_table(
        "batch-1 decode coverage (shares are relative, from the c1 decode trace)",
        ["component", "share", "where to measure"],
        [list(row) for row in _DECODE_COVERAGE],
    )
    print(
        "\nDo not sum only the covered rows and call it a step budget -- see the\n"
        "NOT COVERED entries above."
    )


def _int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(",")]


def _pair_list(value: str) -> list[tuple[int, int]]:
    """Parse "8x4,8x6" into [(8, 4), (8, 6)]."""
    pairs = []
    for item in value.split(","):
        a, _, b = item.partition("x")
        pairs.append((int(a), int(b)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=(*KERNELS, "all"), default="all")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="print what a batch-1 decode step is made of and exit",
    )
    parser.add_argument("--batches", type=_int_list, default=[1, 8, 32])
    parser.add_argument("--block-h", type=_int_list, default=[16, 8, 4, 2])
    parser.add_argument("--splits", type=_int_list, default=[1, 4, 8, 16, 32, 64])
    parser.add_argument(
        "--c128-ms",
        type=_int_list,
        default=[15360],
        help="chunk tokens for the ratio-128 prefill arm; 15360 is the served "
        "PREFILL_CHUNK_SIZE at 256k",
    )
    parser.add_argument(
        "--c128-depths",
        type=_int_list,
        default=[0, 100_000, 200_000],
        help="context already in the KV cache when the chunk starts; the "
        "compressed prefix is depth/128 rows, so this is the axis the query "
        "blocking is priced against",
    )
    parser.add_argument(
        "--c128-block-ms",
        type=_int_list,
        default=[1, 2, 4, 8, 16],
        help="BLOCK_M sweep for the query-blocked kernels. BLOCK_M=1 is the "
        "control rung: same tile as today, but rows derived from positions "
        "instead of read from an index list",
    )
    parser.add_argument(
        "--c128-batches",
        type=_int_list,
        default=[4, 16, 27],
        help="resident requests for the ratio-128 decode arm; 27 is the "
        "measured c64@256k residency",
    )
    parser.add_argument(
        "--c128-next-n",
        type=int,
        default=6,
        help="query tokens per request (1 + num_speculative_tokens). The "
        "per-query arms above hardcode 1, which is a harness/production "
        "mismatch this arm exists to close",
    )
    parser.add_argument(
        "--c128-decode-depths",
        type=_int_list,
        default=[200_000],
        help="context depth for the ratio-128 decode arm",
    )
    parser.add_argument(
        "--c128-no-check",
        action="store_true",
        help="skip the fp32 reference check on the prefill arm (it costs a "
        "few seconds per operating point)",
    )
    parser.add_argument("--warps", type=_int_list, default=[4, 8])
    parser.add_argument("--tokens", type=_int_list, default=[1, 8, 32, 128, 512, 2048])
    parser.add_argument(
        "--prenorm-configs",
        type=_pair_list,
        default=[(2, 12), (4, 8), (8, 4), (8, 6), (8, 12)],
        help="block_m x tile_n pairs, e.g. 8x4,8x6",
    )
    parser.add_argument(
        "--sinkhorn-iters", type=_int_list, default=[1, 5, 10, SINKHORN_ITERS]
    )
    parser.add_argument("--gather-lens", type=_int_list, default=[2176, 8192, 32768])
    parser.add_argument("--gather-reqs", type=_int_list, default=[1, 4])
    parser.add_argument(
        "--gather-workers", type=_int_list, default=[128, 256, 512, 1024, 2048]
    )
    parser.add_argument(
        "--logits-ms",
        type=_int_list,
        default=[240, 2048],
        help="query rows per call. 240 is the production prefill shape at "
        "256k: the logits budget splits a 15,360-token chunk into 8 "
        "sub-chunks and the query shard splits each across 8 ranks. 2048 is "
        "the point the KV_GROUP gate was originally tuned at",
    )
    parser.add_argument(
        "--logits-ns",
        type=_int_list,
        default=[28672, 61440],
        help="compressed context lengths (= context/compress_ratio); 61440 "
        "is the 256k-deep chunk",
    )
    parser.add_argument(
        "--prefill-index-mode",
        type=lambda v: [x.strip() for x in v.split(",")],
        default=["topk"],
        help="sparse-prefill index construction: 'topk' builds the ratio-4 "
        "layers' scattered per-query selections, 'prefix' builds the "
        "ratio-128 layers' real dense prefix (pos+1)//128, which is the only "
        "mode that preserves row-sharing between adjacent queries",
    )
    parser.add_argument(
        "--logits-factor-scale",
        type=_int_list,
        default=[0, 1],
        help="K7 arm: 0 scales every (head, kv) element before the relu, "
        "1 factors k_scale out to a per-kv multiply on the head sum",
    )
    parser.add_argument(
        "--prefill-index-mode",
        type=lambda v: [x.strip() for x in v.split(",")],
        default=["topk"],
        help="sparse-prefill index construction: 'topk' builds the ratio-4 "
        "layers' scattered per-query selections, 'prefix' builds the "
        "ratio-128 layers' real dense prefix (pos+1)//128, which is the only "
        "mode that preserves row-sharing between adjacent queries",
    )
    parser.add_argument(
        "--maxnreg",
        type=_int_list,
        default=[0, 128],
        help="maxnreg values; 0 means unconstrained (today's 132 regs)",
    )
    parser.add_argument(
        "--logits-stages",
        type=_int_list,
        default=[2],
        help="num_stages values (production autotunes over 2 and 4)",
    )
    parser.add_argument(
        "--logits-groups",
        type=_int_list,
        default=[1, 4],
        help="KV_GROUP values: BLOCK_N tiles per CTA sharing one q-tile load",
    )
    parser.add_argument(
        "--paged-batches",
        type=_int_list,
        default=[4, 12, 27],
        help="decode batch for the paged indexer sweep; the grid is "
        "(B x next_n) x blocks, and 27 is the c64@256k residency",
    )
    parser.add_argument(
        "--paged-next-n",
        type=int,
        default=6,
        help="DSpark draft length: 5 speculative tokens plus the bonus. The "
        "grid's first dimension is B x next_n",
    )
    parser.add_argument(
        "--paged-q-bf16",
        type=_int_list,
        default=[0, 1],
        help="K1 arm: 0 LUT-decodes q inside every CTA, 1 takes q already "
        "decoded to bf16 by the wrapper",
    )
    parser.add_argument(
        "--paged-ns",
        type=_int_list,
        default=[940, 26750],
        help="compressed context lengths for the paged indexer sweep; the "
        "defaults are the two operating points the node trace measured "
        "(short-context and 107k at compress_ratio 4)",
    )
    parser.add_argument(
        "--decode-ctx",
        type=_int_list,
        default=None,
        help="context lengths (raw tokens) for the sparse-decode sweep; each "
        "derives its own top-k segment, SWA segment and gather-pool size, "
        "which is the only way to place the bench at a live operating point "
        "instead of between two of them. Overrides --decode-topk-len and "
        "--decode-pool-rows.",
    )
    parser.add_argument(
        "--decode-cache-rows",
        type=int,
        default=0,
        help="total rows in the compressed KV pool the top-k gather scatters "
        "through; the sequence still owns only ctx/compress_ratio of them, so "
        "this separates address spread from touched bytes. 0 = the pool is the "
        "sequence (compact). Serving's pool is ~48k blocks x 64 rows.",
    )
    parser.add_argument(
        "--decode-topk-len",
        type=int,
        default=TOPK_LEN,
        help="compressed top-k tokens per query (min(index_topk, ctx/ratio))",
    )
    parser.add_argument(
        "--decode-pool-rows",
        type=int,
        default=26875,
        help="rows the scattered top-k gather indexes into; keep this at the "
        "live ctx/compress_ratio (L2-resident), not the cache capacity",
    )
    parser.add_argument(
        "--prefill-ms",
        type=_int_list,
        default=[2048],
        help="chunk tokens; 2048 is the serving max_num_batched_tokens",
    )
    parser.add_argument(
        "--prefill-ctxs",
        type=_int_list,
        default=[8192, 32768],
        help="KV pool rows the top-k gather scatters across",
    )
    parser.add_argument(
        "--prefill-block-ks",
        type=_int_list,
        default=[16],
        help="BLOCK_K values; production picks 16 for head_dim >= 256",
    )
    parser.add_argument(
        "--prefill-stages",
        type=_int_list,
        default=[0],
        help="num_stages; 0 = omit the argument, which is what serving does",
    )
    parser.add_argument(
        "--moe-tokens",
        type=_int_list,
        default=[1, 2048],
        help="M for --kernel moe-experts: 1 is decode, 2048 a prefill chunk.",
    )
    parser.add_argument(
        "--moe-skew",
        type=lambda v: v.split(","),
        default=["uniform", "random", "zipf"],
        help="expert-load distribution(s): uniform, random, zipf, hot.",
    )
    parser.add_argument("--moe-experts", type=int, default=CFG_N_ROUTED_EXPERTS)
    parser.add_argument("--moe-topk", type=int, default=CFG_NUM_EXPERTS_PER_TOK)
    parser.add_argument(
        "--moe-n",
        type=int,
        default=MOE_N_PER_RANK,
        help="intermediate size per rank (moe_intermediate_size / TP).",
    )
    parser.add_argument(
        "--moe-rotations",
        type=int,
        default=8,
        help="distinct routing+input sets per timing (weight-set rotation).",
    )
    parser.add_argument("--gemv-ms", type=_int_list, default=[1])
    parser.add_argument("--gemv-block-ns", type=_int_list, default=[16, 32, 64])
    parser.add_argument(
        "--gemv-block-ks", type=_int_list, default=[256, 512, 1024, 2048]
    )
    args = parser.parse_args()

    if args.coverage:
        print_decode_coverage()
        return

    torch.manual_seed(0)
    device = torch.device("cuda")
    # The two c128 arms build a 15,360-query chunk and an fp32 reference per
    # operating point, so they are opt-in rather than part of "all".
    selected = (
        tuple(k for k in KERNELS if not k.endswith("-c128"))
        if args.kernel == "all"
        else (args.kernel,)
    )

    if "moe-experts" in selected:
        bench_moe_experts(
            args.moe_tokens,
            args.moe_skew,
            args.moe_experts,
            args.moe_topk,
            args.moe_n,
            CFG_HIDDEN_SIZE,
            args.moe_rotations,
            device,
        )

    if "sparse-decode" in selected:
        points = (
            [_decode_operating_point(ctx) for ctx in args.decode_ctx]
            if args.decode_ctx
            else [
                dict(
                    topk_len=args.decode_topk_len,
                    topk_rows=args.decode_pool_rows,
                    swa_len=SWA_LEN,
                )
            ]
        )
        for point in points:
            if "ctx" in point:
                print(f"\n### context {point['ctx']} tokens")
            bench_sparse_decode(
                args.batches,
                args.block_h,
                args.splits,
                args.warps,
                device,
                point["topk_len"],
                point["topk_rows"],
                point["swa_len"],
                args.decode_cache_rows,
            )
    if "sparse-prefill-c128" in selected:
        bench_sparse_prefill_c128(
            args.c128_ms,
            args.c128_depths,
            args.c128_block_ms,
            args.prefill_block_ks,
            args.warps,
            device,
            check=not args.c128_no_check,
        )
    if "sparse-decode-c128" in selected:
        bench_sparse_decode_c128(
            args.c128_batches,
            args.c128_next_n,
            args.c128_decode_depths,
            args.c128_block_ms,
            args.splits,
            args.warps,
            device,
            args.decode_cache_rows,
        )
    if "sparse-prefill" in selected:
        bench_sparse_prefill(
            args.prefill_ms,
            args.prefill_ctxs,
            args.block_h,
            args.prefill_block_ks,
            args.warps,
            args.maxnreg,
            args.prefill_stages,
            device,
            args.prefill_index_mode,
        )
    if "prenorm-gemm" in selected:
        bench_prenorm_gemm(args.tokens, args.prenorm_configs, device)
    if "mhc-pre" in selected:
        bench_mhc_pre(args.tokens, args.sinkhorn_iters, device)
    if "dequant-gather" in selected:
        bench_dequant_gather(
            args.gather_lens, args.gather_reqs, args.gather_workers, device
        )
    if "indexer-logits" in selected:
        bench_indexer_logits(
            args.logits_ms,
            args.logits_ns,
            args.maxnreg,
            args.logits_stages,
            args.logits_groups,
            device,
            args.logits_factor_scale,
        )
    if "indexer-paged" in selected:
        bench_indexer_paged(
            args.paged_batches,
            args.paged_ns,
            args.maxnreg,
            args.logits_stages,
            device,
            args.paged_next_n,
            args.paged_q_bf16,
        )
    if "tail-launch" in selected:
        bench_tail_launch(args.gemv_ms, device)
    if "bf16-gemv" in selected:
        bench_bf16_gemv(args.gemv_ms, device)
    if "attn-input-gemm" in selected:
        bench_attn_input_gemm(args.gemv_ms, device)
    if "dense-gemv" in selected:
        bench_dense_gemv(args.gemv_ms, args.gemv_block_ns, device)


if __name__ == "__main__":
    main()
