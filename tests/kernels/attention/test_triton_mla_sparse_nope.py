# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NoPE (head_dim 512) sparse MLA on the Triton kernel (GLM-5.3-Flash sm8x)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse_kernel import triton_mla_sparse_attention

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-alike GPU"
)


def _ref(q, kv, idx, sm_scale, dv=512):
    # q [T,H,D], kv [N,1,D], idx [T,1,K]
    T, H, _ = q.shape
    out = torch.zeros(T, H, dv, dtype=torch.float32, device=q.device)
    for t in range(T):
        sel = idx[t, 0]
        valid = (sel >= 0) & (sel < kv.shape[0])
        s = sel[valid].long()
        if s.numel() == 0:
            continue
        k = kv[s, 0].float()  # [k, D]
        qk = (q[t].float() @ k.T) * sm_scale  # [H, k]
        p = torch.softmax(qk, dim=-1)
        out[t] = p @ k[:, :dv]
    return out


@pytest.mark.parametrize("dim_qk", [576, 512])
@pytest.mark.parametrize("num_kv_splits", [1, 4])
def test_triton_mla_sparse_nope_and_rope(dim_qk: int, num_kv_splits: int):
    torch.manual_seed(0)
    dev = "cuda"
    T, H, N, K = 4, 16, 2048, 128
    q = torch.randn(T, H, dim_qk, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(N, 1, dim_qk, dtype=torch.bfloat16, device=dev)
    idx = torch.stack([torch.randperm(N, device=dev)[:K].int() for _ in range(T)])
    idx = idx.view(T, 1, K)
    sm = dim_qk**-0.5
    out = triton_mla_sparse_attention(
        q, kv, idx, sm_scale=sm, num_kv_splits=num_kv_splits
    )
    ref = _ref(q, kv, idx, sm)
    err = (out.float() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    assert rel < 2e-2, f"dim_qk={dim_qk} splits={num_kv_splits} rel={rel:.2e}"


@pytest.mark.parametrize("dim_qk", [576, 512])
@pytest.mark.parametrize("num_kv_splits", [1, 4])
def test_triton_mla_sparse_fp8_kv(dim_qk: int, num_kv_splits: int):
    """An e4m3 cache (`kv / k_scale`, as concat_and_cache_mla writes it) is
    decoded exactly: same output as the bf16 kernel on the dequantized rows,
    and within fp8 quantization error of the unquantized ones."""
    torch.manual_seed(0)
    dev = "cuda"
    T, H, N, K = 4, 16, 2048, 128
    k_scale = 0.5
    q = torch.randn(T, H, dim_qk, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(N, 1, dim_qk, dtype=torch.bfloat16, device=dev)
    kv_fp8 = (kv.float() / k_scale).to(torch.float8_e4m3fn)
    idx = torch.stack([torch.randperm(N, device=dev)[:K].int() for _ in range(T)])
    idx = idx.view(T, 1, K)
    idx[0, 0, K // 2 :] = -1
    sm = dim_qk**-0.5

    def run(cache, **kwargs):
        return triton_mla_sparse_attention(
            q, cache, idx, sm_scale=sm, num_kv_splits=num_kv_splits, **kwargs
        ).float()

    out = run(kv_fp8, k_scale=k_scale)
    dequant = run((kv_fp8.float() * k_scale).to(torch.bfloat16))
    torch.testing.assert_close(out, dequant, atol=2e-3, rtol=0)
    ref = run(kv)
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 5e-2, f"dim_qk={dim_qk} splits={num_kv_splits} rel={rel:.2e}"
