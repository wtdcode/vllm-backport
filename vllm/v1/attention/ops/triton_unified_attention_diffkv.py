# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton unified attention with different K/V head dimensions (DiffKV).

This is a slimmed fork of ``triton_unified_attention.py`` for models like
MiMo-V2.5 where the V tensor's head dimension differs from K's.  The KV cache
is the same packed layout used by ``FlashAttentionDiffKVBackend``:

    kv_cache: [num_blocks, block_size, num_kv_heads, head_size_qk + head_size_v]

We slice ``key_cache = kv_cache[..., :head_size_qk]`` and
``value_cache = kv_cache[..., head_size_qk:]`` on the host, so the kernel
takes two cache pointers but with two distinct head sizes.

Both 2D and 3D launches are supported:
  - 2D: one program per (q-block, kv-head); tile-loop walks the full KV
    sequence; final output written directly.  Used for prefill and large
    decode batches.
  - 3D: one program per (q-block, kv-head, segm); each program covers a
    KV slice and writes per-segment partials (max/expsum/output).  A
    follow-up ``kernel_reduce_segments_diffkv`` combines them.  Selected
    for decode-only batches whose 2D grid would under-fill the GPU.
"""

import os
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    find_seq_idx,
    init_softmax_M,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)

logger = init_logger(__name__)

is_batch_invariant = envs.VLLM_BATCH_INVARIANT

# diffkv split-KV + wide-prefill knobs (thor patch, ported from the diffbot recipe).
# Stock-off defaults: the stock kernel behavior is unchanged unless explicitly enabled
# on the server (validated values from the diffbot recipe in comments).
# VLLM_DIFFKV_SPEC_3D_MAX_Q: max query tokens per sequence for the split-KV verify path
# (0 = off; diffbot validated 16: MTP verify batches take the 3D split-KV launch).
# Upstream default 0 (path off). 16 enables the split-KV verify launch:
# measured 3x decode at 33K ctx with MTP-3 on sm86 TP8 (50 -> 149 TG/s).
_SPEC_3D_MAX_Q = int(os.environ.get("VLLM_DIFFKV_SPEC_3D_MAX_Q", "16"))
# Wide-prefill knobs (diffbot recipe): prefill-shaped 2D launches
# (max_seqlen_q >= _PREFILL_MIN_Q) use a larger BLOCK_M (several query
# tokens per program) so K/V tiles are reused across query rows. The stock
# BLOCK_M=16 with a GQA group of 16 means ONE query token per program.
# Validated on sm120: 128 / 8 warps / tile 32 = 3.0x on a 4096-token chunk
# at 30K ctx (global layers), 2.5x SWA; tile 64 exceeds sm120's 99 KB smem
# for BLOCK_M 128. Stock-off default: 16 keeps the stock launch.
# Upstream default 16. 128 = wide prefill tiles (recipe-claimed up to 3x on
# 4096-token chunks; kept by the sm86 prefill ladder). Automatically clamped
# to 64 under the fp8-KV LUT path on sm<89 (shared-memory ceiling).
_PREFILL_BLOCK_M = int(os.environ.get("VLLM_DIFFKV_PREFILL_BLOCK_M", "128"))
_PREFILL_NUM_WARPS = int(os.environ.get("VLLM_DIFFKV_PREFILL_NUM_WARPS", "8"))
# 2 stages: same speed as the default 3 for bf16 and keeps the fp8-KV dequant
# under smem limits.
_PREFILL_NUM_STAGES = int(os.environ.get("VLLM_DIFFKV_PREFILL_NUM_STAGES", "2"))
_PREFILL_TILE = int(os.environ.get("VLLM_DIFFKV_PREFILL_TILE", "32"))
# fp8 KV on full-attention layers: tile 64 fits (fp8 K/V tiles halve smem).
# SWA layers keep tile 32 (tile 64 + sinks exceeds smem).
_PREFILL_TILE_FP8_GLOBAL = int(
    os.environ.get("VLLM_DIFFKV_PREFILL_TILE_FP8_GLOBAL", "64")
)
_PREFILL_MIN_Q = 64
# Split-KV spec verify (diffbot recipe): cover ALL q tokens of a request in
# one program (BLOCK_M = q_len x GQA group, capped) so each K/V tile is read
# once instead of once per verify token (8x for q=8).
_SPEC_3D_BLOCK_M = int(os.environ.get("VLLM_DIFFKV_SPEC_3D_BLOCK_M", "128"))
_SPEC_3D_NUM_WARPS = int(os.environ.get("VLLM_DIFFKV_SPEC_3D_NUM_WARPS", "8"))
_SPEC_3D_TILE = int(os.environ.get("VLLM_DIFFKV_SPEC_3D_TILE", "16"))
# fp8 E4M3 KV dequant strategy: "auto" picks the LUT gather below sm89 and
# the native in-kernel conversion on sm89+ (Triton cannot lower the native
# E4M3->fp16/bf16 element conversion on sm80/sm86, see vllm PR #55184).
# "lut" / "native" force a mode for testing.
_FP8_DEQUANT_MODE = os.environ.get("VLLM_DIFFKV_FP8_DEQUANT", "auto").lower()

_fp8_e4m3_lut_cache: dict[tuple[int, torch.device], torch.Tensor] = {}


def _get_fp8_e4m3_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """256-entry dequant table for E4M3 bytes (sm86-safe gather path)."""
    key = (str(dtype), device)
    lut = _fp8_e4m3_lut_cache.get(key)
    if lut is None:
        codes = torch.arange(256, dtype=torch.uint8)
        lut = codes.view(torch.float8_e4m3fn).to(dtype).contiguous().to(device)
        _fp8_e4m3_lut_cache[key] = lut
    return lut


@triton.jit
def kernel_unified_attention_diffkv(
    # Output destinations.  In 2D mode we write the final result into
    # ``output_ptr``; in 3D mode we write per-segment partials into
    # ``segm_*`` and ``output_ptr`` is unused (callers may pass any
    # non-null pointer).
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,  # view of packed cache: [..., :head_size_qk]
    value_cache_ptr,  # view of packed cache: [..., head_size_qk:hqk+hv]
    sink_ptr,
    k_descale_ptr,
    v_descale_ptr,
    FP8_KV_CACHE: tl.constexpr,
    lut_ptr,
    FP8_LUT: tl.constexpr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    scale,
    softcap,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,  # == HEAD_SIZE_QK
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,  # == HEAD_SIZE_V
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_QK: tl.constexpr,
    HEAD_SIZE_QK_PADDED: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    # Strides for both cache views (they share the same packed buffer, so
    # dims 0/1/2 strides match; only the per-head extent differs).
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    # ``IS_3D`` toggles between 2D layout (one program walks the full KV
    # sequence) and 3D layout (split-KV / FlashDecoding-style: per-segm
    # programs write partials, finalized by ``kernel_reduce_segments_diffkv``).
    IS_3D: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d_qk = tl.arange(0, HEAD_SIZE_QK_PADDED)
    offs_d_v = tl.arange(0, HEAD_SIZE_V_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d_qk[None, :]
    )

    dim_mask_qk = tl.where(offs_d_qk < HEAD_SIZE_QK, 1, 0).to(tl.int1)
    dim_mask_v = tl.where(offs_d_v < HEAD_SIZE_V, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_QK_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask_qk[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    if FP8_KV_CACHE:
        k_descale = tl.load(k_descale_ptr)
        v_descale = tl.load(v_descale_ptr)
        # fp8 KV: run the dots in fp16 -- sm120 converts E4M3->fp16 natively,
        # E4M3->bf16 is a multi-step path that made BLOCK_M 128 prefill 1.74x
        # slower. fp16 also has more mantissa than bf16.
        Q = Q.to(tl.float16)

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_V_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_V_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        False,  # USE_MM_PREFIX
        IS_3D,
    )

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d_v[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d_qk[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        # K : (HEAD_SIZE_QK_PADDED, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask_qk[:, None] & tile_mask[None, :],
            other=0.0,
        )
        # E4M3 -> fp16/bf16 is exact; the per-tensor K/V descales are applied
        # in fp32 on S and acc below (no fp32 staging tile: that pushed
        # BLOCK_M 128 prefill past smem limits).
        if FP8_LUT:
            # sm80/sm86: dequantize through a 256-entry fp16 LUT gather
            # instead of the (unavailable) native element conversion.
            K = tl.load(
                lut_ptr + K_load.to(tl.int32),
                mask=dim_mask_qk[:, None] & tile_mask[None, :],
                other=0.0,
            )
        else:
            K = K_load.to(Q.dtype)
        # V : (TILE_SIZE, HEAD_SIZE_V_PADDED)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask_v[None, :] & tile_mask[:, None],
            other=0.0,
        )
        if FP8_LUT:
            V = tl.load(
                lut_ptr + V_load.to(tl.int32),
                mask=dim_mask_v[None, :] & tile_mask[:, None],
                other=0.0,
            )
        else:
            V = V_load.to(Q.dtype)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            None,  # mm_prefix_range_ptr
            SLIDING_WINDOW,
            False,  # USE_MM_PREFIX
            0,  # MAX_MM_RANGES
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if FP8_KV_CACHE:
            S += (scale * k_descale) * tl.dot(Q, K)
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )
        if FP8_KV_CACHE:
            acc += tl.dot(P.to(V.dtype), V) * v_descale
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue --------------------------------------------------------
    if IS_3D:
        # Store per-segment partials; finalized by reduce_segments_diffkv.
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
            + segm_idx * HEAD_SIZE_V_PADDED
            + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
        )
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        # With several query tokens per program, a row can see only masked
        # keys in this segment (softmax_step then leaves M=0, L=0); report
        # -inf so reduce_segments ignores it.
        M = tl.where(L > 0.0, M, float("-inf"))
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d_v[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


@triton.jit
def kernel_reduce_segments_diffkv(
    output_ptr,  # [num_tokens, num_query_heads, head_size_v]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size_v]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,  # == HEAD_SIZE_V
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Combine per-segment partials into the final softmax output.

    Mirrors ``reduce_segments`` from triton_unified_attention.py but
    indexes V's head size (``HEAD_SIZE_V``) instead of the shared one.
    """
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_V_PADDED) < HEAD_SIZE_V, 1, 0).to(
        tl.int1
    )

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_V_PADDED
        + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_V_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def unified_attention_diffkv(
    q,  # [num_tokens, num_query_heads, head_size_qk]
    k,  # view: [num_blocks, block_size, num_kv_heads, head_size_qk]
    v,  # view: [num_blocks, block_size, num_kv_heads, head_size_v]
    out,  # [num_tokens, num_query_heads, head_size_v]
    cu_seqlens_q,
    seqused_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    max_seqlen_q: int = 1,
    alibi_slopes=None,
    sinks=None,
    use_alibi_sqrt=False,
    # 3D / split-KV softmax buffers.  When all four are provided and the
    # batch is decode-only with few sequences, the 3D path is taken.
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
):
    assert causal, "Only causal attention is supported"
    # thor port of local-inference-lab/vllm #830: per-tensor E4M3 K/V cache.
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    fp8_kv_cache = k.dtype in fp8_dtypes
    if fp8_kv_cache:
        if v.dtype != k.dtype or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FP8 DiffKV requires matching E4M3 K/V and FP16/BF16 Q")
        if k_descale is None or v_descale is None:
            raise ValueError("FP8 DiffKV requires separate K and V descale tensors")
        if k_descale.numel() != 1 or v_descale.numel() != 1:
            raise ValueError("FP8 DiffKV only supports per-tensor K/V scales")

    # sm80/sm86 cannot lower the native E4M3 conversion; dequant through a
    # 256-entry LUT gather instead (cache reinterpreted as raw bytes).
    fp8_lut = False
    lut = None
    if fp8_kv_cache:
        mode = _FP8_DEQUANT_MODE
        if mode == "auto":
            if q.is_cuda:
                major, minor = torch.cuda.get_device_capability(q.device)
                mode = "native" if (major, minor) >= (8, 9) else "lut"
            else:
                mode = "native"
        if mode == "lut":
            if k.dtype != torch.float8_e4m3fn:
                raise ValueError(f"LUT dequant requires E4M3 (fn) cache, got {k.dtype}")
            fp8_lut = True
            lut = _get_fp8_e4m3_lut(q.device, torch.float16)
            k = k.view(torch.uint8)
            v = v.view(torch.uint8)
        elif mode != "native":
            raise ValueError(
                f"VLLM_DIFFKV_FP8_DEQUANT must be auto|lut|native, got {mode!r}"
            )

    # sm80/sm86 have a ~99KB shared-memory ceiling. The LUT gather stages
    # the raw uint8 tile alongside the gathered fp16 tile, so a BLOCK_M=128
    # prefill/verify kernel needs ~112KB and fails to launch
    # (OutOfResources: 114688 vs 101376). bf16 direct loads fit at 128 on
    # these parts, so only clamp the wide-tile knobs when the LUT path is
    # actually taken below sm89.
    lut_low_smem = fp8_lut and q.is_cuda and (
        torch.cuda.get_device_capability(q.device) < (8, 9)
    )
    prefill_block_m = _PREFILL_BLOCK_M
    spec_3d_block_m = _SPEC_3D_BLOCK_M
    if lut_low_smem:
        if prefill_block_m > 64:
            logger.warning_once(
                "fp8-KV LUT dequant: clamping VLLM_DIFFKV_PREFILL_BLOCK_M "
                "%d -> 64 to fit the sm86 shared-memory ceiling",
                prefill_block_m,
            )
            prefill_block_m = 64
        if spec_3d_block_m > 64:
            logger.warning_once(
                "fp8-KV LUT dequant: clamping VLLM_DIFFKV_SPEC_3D_BLOCK_M "
                "%d -> 64 to fit the sm86 shared-memory ceiling",
                spec_3d_block_m,
            )
            spec_3d_block_m = 64

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size_qk = q.shape[2]
    head_size_v = v.shape[3]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    launch_kw: dict[str, int] = {}
    prefill_tile = None
    if (
        max_seqlen_q >= _PREFILL_MIN_Q
        and prefill_block_m > BLOCK_M
        and prefill_block_m % num_queries_per_kv == 0
    ):
        BLOCK_M = prefill_block_m
        launch_kw["num_warps"] = _PREFILL_NUM_WARPS
        if _PREFILL_NUM_STAGES > 0:
            launch_kw["num_stages"] = _PREFILL_NUM_STAGES
        prefill_tile = _PREFILL_TILE
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Decide between 2D and 3D launch.  Mirrors the standard launcher:
    # 3D requires preallocated softmax buffers, decode-only batches, and
    # a small number of sequences (otherwise 2D already saturates the SM).
    # thor patch: short multi-token decode batches (spec-decode verify,
    # q_len <= _SPEC_3D_MAX_Q) on full-attention layers also take the split-KV
    # path.  Their 2D grid is only (q_blocks x kv_heads) programs -- 18 for a
    # q_len=8 verify with 2 KV heads/rank -- each walking the whole context.
    # Sliding-window layers keep 2D: their loop is already window-bounded.
    spec_3d = (
        1 < max_seqlen_q <= _SPEC_3D_MAX_Q
        and sliding_window_val == 0
        and softmax_segm_output is not None
        and q.shape[0] <= softmax_segm_output.shape[0]
    )
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or (max_seqlen_q > 1 and not spec_3d)
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    spec_tile = None
    if use_3d and spec_3d and spec_3d_block_m > BLOCK_M:
        spec_bm = min(
            spec_3d_block_m,
            triton.next_power_of_2(max_seqlen_q * num_queries_per_kv),
        )
        if spec_bm > BLOCK_M and spec_bm % num_queries_per_kv == 0:
            BLOCK_M = spec_bm
            launch_kw["num_warps"] = _SPEC_3D_NUM_WARPS if BLOCK_M >= 128 else 4
            spec_tile = _SPEC_3D_TILE
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Tile size: 32 for prefill-class kernels.  Decode (small Q) prefers
    # smaller tiles to expose more parallelism along the KV dim.
    tile_size = 32 if not use_3d else (16 if q.element_size() >= 2 else 32)
    if spec_tile is not None:
        tile_size = spec_tile
    if prefill_tile is not None and not use_3d:
        tile_size = prefill_tile
        if fp8_kv_cache and sliding_window_val == 0:
            tile_size = _PREFILL_TILE_FP8_GLOBAL

    grid: tuple[Any, ...]
    if use_3d:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        segm_output_ptr = softmax_segm_output
        segm_max_ptr = softmax_segm_max
        segm_expsum_ptr = softmax_segm_expsum
        num_segments = num_par_softmax_segments
    else:
        grid = (total_num_q_blocks, num_kv_heads)
        # 2D never touches the segm tensors but Triton wants a non-null
        # pointer; reuse ``out``.
        segm_output_ptr = out
        segm_max_ptr = out
        segm_expsum_ptr = out
        num_segments = 1

    kernel_unified_attention_diffkv[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        k_descale_ptr=k_descale,
        v_descale_ptr=v_descale,
        FP8_KV_CACHE=fp8_kv_cache,
        lut_ptr=lut if fp8_lut else q,
        FP8_LUT=fp8_lut,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        scale=softmax_scale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE_QK=head_size_qk,
        HEAD_SIZE_QK_PADDED=triton.next_power_of_2(head_size_qk),
        HEAD_SIZE_V=head_size_v,
        HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=sliding_window_val,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        IS_3D=use_3d,
        **launch_kw,
    )

    if use_3d:
        kernel_reduce_segments_diffkv[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            TILE_SIZE=tile_size,
            HEAD_SIZE_V=head_size_v,
            HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
        )
