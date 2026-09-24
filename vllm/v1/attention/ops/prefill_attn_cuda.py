# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT loader + launcher for the sm86 CUDA chunked-prefill attention kernel.

Purpose-built for MiMo-V2.6 global (full-attention) layers' prefill at TP8:
Q [nq, 8, 192] bf16, packed paged KV [pages, 1, page, 320] fp8-e4m3/bf16,
causal with prefix. Selected per layer by the Triton DiffKV backend behind
VLLM_DIFFKV_PREFILL_CUDA=1 (full-attention layers only; the 39 SWA layers
and all decode/verify paths keep the Triton kernels).

The extension JIT-compiles on first use (~1 min, cached under
~/.cache/vllm/prefill_attn_sm86) so no vLLM rebuild is needed.
"""

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Upstream default off. Default-on after the chained-prefix long-ctx A/B on
# sm86 TP8: incremental prefill +7.1% @20K, +4.7% @50K, +13.0% @100K,
# TTFT@100K 37.4 -> 33.1 s (short/mid mixed-bench ~-5% decode; opt out
# with VLLM_DIFFKV_PREFILL_CUDA=0 for short-ctx latency profiles).
_PREFILL_CUDA_ENV = os.environ.get("VLLM_DIFFKV_PREFILL_CUDA", "1") != "0"

if _PREFILL_CUDA_ENV:
    # Import-time marker: distinguishes "env set but hook never engaged"
    # (shape/layer conditions failed) from "env unset or code absent".
    logger.info(
        "VLLM_DIFFKV_PREFILL_CUDA=1: CUDA prefill kernel enabled "
        "(JIT-compiles on first qualifying prefill; look for "
        "'JIT-compiled prefill_attn_sm86')"
    )


def prefill_cuda_enabled() -> bool:
    return _PREFILL_CUDA_ENV


@lru_cache(maxsize=1)
def _load_ext():
    """JIT-compile the extension (cached on disk after the first build)."""
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "csrc", "prefill_attn_sm86.cu")
    cache = os.path.expanduser(
        os.path.join("~", ".cache", "vllm", "prefill_attn_sm86")
    )
    os.makedirs(cache, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    ext = load(
        name=f"prefill_attn_sm86_{arch}",
        sources=[src],
        extra_cuda_cflags=[
            "-O3",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            "--use_fast_math",
        ],
        extra_cflags=["-O3"],
        build_directory=cache,
        verbose=False,
    )
    logger.info_once("JIT-compiled prefill_attn_sm86 (sm%s)", arch)
    return ext


def _qblock_metadata(
    cu_seqlens_q: torch.Tensor, block_tokens: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per q-block (seq idx, first global q row) for 16-token blocks per seq."""
    device = cu_seqlens_q.device
    lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    starts = cu_seqlens_q[:-1]
    blocks_per = (lens + block_tokens - 1) // block_tokens
    total = int(blocks_per.sum().item())
    seq_of_block = torch.repeat_interleave(
        torch.arange(lens.numel(), device=device), blocks_per
    )
    block_start_offset = torch.cumsum(blocks_per, 0) - blocks_per
    local_block = (
        torch.arange(total, device=device) - block_start_offset[seq_of_block]
    )
    qblk_seq = seq_of_block.to(torch.int32)
    qblk_start = (starts[seq_of_block] + local_block * block_tokens).to(torch.int32)
    return qblk_seq, qblk_start


def prefill_attn_cuda(
    q: torch.Tensor,
    kv_cache: torch.Tensor,  # [pages, 1, page, 320] fp8e4m3 or bf16
    block_table: torch.Tensor,  # [S, max_pages] int32
    cu_seqlens_q: torch.Tensor,  # [S+1] int32
    seq_lens: torch.Tensor,  # [S] int32 (context + new)
    softmax_scale: float,
    k_descale: float,
    v_descale: float,
    out: torch.Tensor,  # [nq, 8, 128] bf16
) -> None:
    ext = _load_ext()
    qblk_seq, qblk_start = _qblock_metadata(cu_seqlens_q)
    ext.prefill_attn_sm86(
        q,
        kv_cache,
        block_table,
        cu_seqlens_q.to(torch.int32),
        seq_lens.to(torch.int32),
        qblk_seq,
        qblk_start,
        softmax_scale,
        k_descale,
        v_descale,
        out,
    )


__all__ = [
    "prefill_attn_cuda",
    "prefill_cuda_enabled",
    "_load_ext",
    "_qblock_metadata",
]
