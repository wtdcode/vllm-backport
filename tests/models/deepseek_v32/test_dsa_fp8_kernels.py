# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 outputs of the DSA fused kernels against a torch reference.

The kernels write every e4m3 value as bytes through a uint8 pointer so they
compile below SM89 (Triton has no fp8e4nv convert there); this checks the
bytes match ``Tensor.to(torch.float8_e4m3fn)`` on whichever path the device
takes, for the indexer query/key, the per-tensor fp8 MLA cache and the
fp8_ds_mla cache.
"""

import pytest
import torch

from vllm.models.deepseek_v32.common.kernels import fused_norm_rope, fused_q

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEVICE = "cuda"
BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn


def _rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


def _rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def _ue8m0_quant(x: torch.Tensor):
    """Mirror of ``_fp8_ue8m0_quantize``: (fp8 bytes, scale) per row."""
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) / 448.0)))
    return (x / scale).to(FP8).view(torch.uint8), scale.squeeze(-1)


def _assert_fp8_bytes_close(got: torch.Tensor, ref: torch.Tensor, name: str):
    """Byte-exact except for rare 1-ulp flips from f32 op ordering."""
    got_f = got.view(FP8).float()
    ref_f = ref.view(FP8).float()
    mismatch = (got != ref).float().mean().item()
    assert mismatch < 2e-3, f"{name}: {mismatch:.2%} bytes differ"
    ulp = torch.maximum(ref_f.abs() * 2**-3, torch.tensor(2**-9, device=DEVICE))
    assert torch.all((got_f - ref_f).abs() <= ulp), f"{name}: >1 ulp difference"


def _cos_sin(rows: int, half: int):
    ang = torch.rand(rows, half, device=DEVICE) * 6.28
    return torch.cat((ang.cos(), ang.sin()), dim=-1).to(BF16)


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("quantize_mqa", [False, True])
def test_fused_q_fp8_outputs(interleave: bool, quantize_mqa: bool):
    torch.manual_seed(0)
    T, H, IH, MAXPOS = 33, 8, 4, 256
    positions = torch.randint(0, MAXPOS, (T,), device=DEVICE, dtype=torch.int64)
    q_pe = torch.randn(T, H, 64, device=DEVICE, dtype=BF16)
    ql_nope = torch.randn(T, H, 512, device=DEVICE, dtype=BF16) * 3
    q_cs = _cos_sin(MAXPOS, 32)
    index_q = torch.randn(T, IH, 128, device=DEVICE, dtype=BF16) * 2
    iq_cs = _cos_sin(MAXPOS, 32)
    q_scale = torch.tensor([0.37], device=DEVICE)
    index_weights = torch.randn(T, IH, device=DEVICE, dtype=BF16)

    index_q_fp8, index_weights_out, mqa_q = fused_q(
        positions,
        q_pe,
        q_cs,
        index_q,
        iq_cs,
        ql_nope,
        q_scale,
        index_weights,
        0.5,
        0.25,
        has_indexer=True,
        index_rope_interleave=interleave,
        quantize_mqa=quantize_mqa,
    )
    torch.cuda.synchronize()

    cos, sin = iq_cs[positions].float().split(32, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    iq = index_q.float()
    rope = _rope_interleaved if interleave else _rope_neox
    iq = torch.cat((rope(iq[..., :64], cos, sin), iq[..., 64:]), dim=-1)
    ref_bytes, ref_scale = _ue8m0_quant(iq)
    _assert_fp8_bytes_close(index_q_fp8.view(torch.uint8), ref_bytes, "index_q")
    ref_w = index_weights.float() * ref_scale * 0.5 * 0.25
    torch.testing.assert_close(index_weights_out, ref_w, rtol=1e-3, atol=1e-6)

    qcos, qsin = q_cs[positions].float().split(32, dim=-1)
    q_pe_ref = _rope_interleaved(q_pe.float(), qcos[:, None], qsin[:, None])
    if quantize_mqa:
        assert mqa_q.dtype == FP8 and mqa_q.shape == (T, H, 576)
        packed = torch.cat((ql_nope.float(), q_pe_ref), dim=-1) / q_scale
        ref = packed.to(FP8).view(torch.uint8)
        _assert_fp8_bytes_close(mqa_q.view(torch.uint8), ref, "mqa_q")
    else:
        assert mqa_q.dtype == BF16
        torch.testing.assert_close(mqa_q.float(), q_pe_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_ds_mla"])
def test_fused_norm_rope_fp8_caches(kv_cache_dtype: str):
    torch.manual_seed(0)
    T, MAXPOS, BLOCK, NBLOCKS = 40, 256, 16, 4
    KV, IDX = 512, 128
    positions = torch.randint(0, MAXPOS, (T,), device=DEVICE, dtype=torch.int64)
    q_c = torch.randn(T, 1536, device=DEVICE, dtype=BF16)
    kv_c = torch.randn(T, KV, device=DEVICE, dtype=BF16)
    k_pe = torch.randn(T, 64, device=DEVICE, dtype=BF16)
    index_k = torch.randn(T, IDX, device=DEVICE, dtype=BF16)
    q_w = torch.rand(1536, device=DEVICE, dtype=BF16) + 0.5
    kv_w = torch.rand(KV, device=DEVICE, dtype=BF16) + 0.5
    ik_w = torch.rand(IDX, device=DEVICE) + 0.5
    ik_b = torch.randn(IDX, device=DEVICE) * 0.1
    k_cs = _cos_sin(MAXPOS, 32)
    ik_cs = _cos_sin(MAXPOS, 32)
    topk = torch.zeros(T, 64, device=DEVICE, dtype=torch.int32)
    slot_mapping = torch.randperm(NBLOCKS * BLOCK, device=DEVICE)[:T]
    slot_mapping[3] = -1  # padding row: no cache write
    idx_cache = torch.zeros(NBLOCKS, BLOCK, IDX + 4, device=DEVICE, dtype=torch.uint8)
    k_scale = torch.tensor([0.61], device=DEVICE)
    if kv_cache_dtype == "auto":
        mla_cache = torch.zeros(NBLOCKS, BLOCK, KV + 64, device=DEVICE, dtype=BF16)
    else:
        row = 656 if kv_cache_dtype == "fp8_ds_mla" else KV + 64
        mla_cache = torch.zeros(NBLOCKS, BLOCK, row, device=DEVICE, dtype=torch.uint8)

    q_c_out = fused_norm_rope(
        positions,
        q_c,
        q_w,
        1e-6,
        kv_c,
        kv_w,
        1e-6,
        k_pe,
        k_cs,
        index_k,
        ik_w,
        ik_b,
        1e-6,
        ik_cs,
        topk,
        slot_mapping=slot_mapping,
        indexer_k_cache=idx_cache,
        mla_kv_cache=mla_cache,
        mla_kv_cache_dtype=kv_cache_dtype,
        mla_k_scale=k_scale,
        has_indexer=True,
        index_rope_interleave=False,
    )
    torch.cuda.synchronize()

    def rms(x, w, eps=1e-6):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()

    torch.testing.assert_close(q_c_out.float(), rms(q_c, q_w), rtol=2e-2, atol=2e-2)

    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    blk, off = slots // BLOCK, slots % BLOCK

    kv_ref = rms(kv_c, kv_w)[valid]
    cos, sin = k_cs[positions].float().split(32, dim=-1)
    kpe_ref = _rope_interleaved(k_pe.float(), cos, sin)[valid]

    if kv_cache_dtype == "auto":
        got = mla_cache[blk, off].float()
        torch.testing.assert_close(got[:, :KV], kv_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(got[:, KV:], kpe_ref, rtol=2e-2, atol=2e-2)
    elif kv_cache_dtype == "fp8":
        ref = (torch.cat((kv_ref, kpe_ref), dim=-1) / k_scale).to(FP8)
        _assert_fp8_bytes_close(mla_cache[blk, off], ref.view(torch.uint8), "mla fp8")
    else:
        rows = mla_cache[blk, off]
        tiles = kv_ref.view(-1, 4, 128)
        tile_scale = (tiles.abs().amax(-1, keepdim=True) / 448.0).clamp_min(
            1.1754944e-38
        )
        ref_nope = (tiles / tile_scale).to(FP8).view(torch.uint8).view(-1, KV)
        _assert_fp8_bytes_close(rows[:, :KV], ref_nope, "ds_mla nope")
        got_scale = rows[:, KV : KV + 16].contiguous().view(torch.float32)
        torch.testing.assert_close(got_scale, tile_scale.view(-1, 4), rtol=1e-6, atol=0)
        got_rope = rows[:, KV + 16 :].contiguous().view(BF16).float()
        torch.testing.assert_close(got_rope, kpe_ref, rtol=2e-2, atol=2e-2)

    # Indexer K: layernorm + NeoX RoPE on the first 64 dims, ue8m0 fp8, scale
    # stored as fp32 after the block's values.
    ik = index_k.float()
    mean = ik.mean(-1, keepdim=True)
    var = ((ik - mean) ** 2).mean(-1, keepdim=True)
    normed = (ik - mean) * torch.rsqrt(var + 1e-6) * ik_w + ik_b
    cos, sin = ik_cs[positions].float().split(32, dim=-1)
    roped = torch.cat((_rope_neox(normed[:, :64], cos, sin), normed[:, 64:]), -1)
    ref_bytes, ref_scale = _ue8m0_quant(roped[valid])
    # Block layout: [BLOCK * IDX value bytes][BLOCK fp32 scales].
    flat = idx_cache.view(NBLOCKS, -1)
    val_idx = off[:, None] * IDX + torch.arange(IDX, device=DEVICE)
    got_vals = flat[blk].gather(1, val_idx)
    _assert_fp8_bytes_close(got_vals, ref_bytes, "indexer k")
    got_scale = flat[:, BLOCK * IDX :].contiguous().view(torch.float32)[blk, off]
    torch.testing.assert_close(got_scale, ref_scale, rtol=0, atol=0)
