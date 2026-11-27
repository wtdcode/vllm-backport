# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch bf16 GEMV for narrow-N unquantized projections.

cuBLAS handles skinny GEMMs by splitting K and reducing, which for
`N = 64` costs two launches (a `dot_kernel` plus a `reduce_1Block_kernel`)
where the whole operation moves only a few hundred KB. Assigning one CTA per
output row removes the split entirely: N supplies the parallelism, each CTA
owns its output, and nothing needs reducing across CTAs.

This only pays at GEMV-ish shapes -- small M, N large enough to fill the
device, K large enough that a CTA has work to do. Prefill-sized M belongs on
cuBLAS, so callers gate on `should_use_triton_gemv`.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Measured boundary, not a guess. Summed over the DSv4 call sites (21 indexer
# + 43 gate layers per step) on A100:
#
#     M     production   triton
#      1     0.435 ms    0.195 ms
#      2     0.552       0.350
#      4     0.573       0.465
#      6     0.565       0.496      (DSpark: 5 speculative tokens + 1)
#      8     0.555       0.512
#     16     0.569       0.817      <- cuBLAS takes over
#
# Past ~8 rows the CTA-per-row shape re-reads x for every row it accumulates
# and a real GEMM stops being latency-bound. M is the whole batch's token
# count, so this lapses on its own at high concurrency.
MAX_GEMV_TOKENS = 8


@triton.jit
def _bf16_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offs = tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        # One weight row, reused across every token in the batch.
        wv = tl.load(w_ptr + pid * stride_wn + k_offs, mask=k_mask, other=0.0).to(
            tl.float32
        )
        xv = tl.load(
            x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += xv * wv[None, :]
    out = tl.sum(acc, axis=1)
    tl.store(
        out_ptr + m_offs * stride_om + pid,
        out.to(out_ptr.dtype.element_ty),
        mask=m_mask,
    )


def should_use_triton_gemv(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether `bf16_gemv` applies to this call."""
    return (
        current_platform.is_cuda()
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.ndim == 2
        and weight.ndim == 2
        and x.shape[0] <= MAX_GEMV_TOKENS
        and x.shape[1] == weight.shape[1]
        and x.is_contiguous()
        and weight.is_contiguous()
    )


def bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`x @ weight.T` for small M, in one launch.

    Accumulates in fp32 and stores directly in `out_dtype`, so an fp32
    output is genuinely fp32 rather than a bf16 result cast upward.
    """
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty(m, n, dtype=out_dtype or x.dtype, device=x.device)
    _bf16_gemv_kernel[(n,)](
        x,
        weight,
        out,
        m,
        k,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_M=triton.next_power_of_2(max(m, 1)),
        # The accumulator is [BLOCK_M, BLOCK_K], so the K tile has to shrink as
        # M grows. At M=1 a wide tile cuts the loop to two trips and measured
        # 2.30 us against 3.05 for a 256-wide one.
        BLOCK_K=2048 if m == 1 else 512,
        num_warps=8,
    )
    return out
