# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek's V4.1 KV records on the SM8x Triton route (`nvfp4_ds_mla`).

The route shares its caches with SM100 and with the SM80 CuTe DSL path, so
the records are held to the latter's PyTorch reference: the NVFP4 writer byte
for byte, the gathers and the decode kernel on what they read back.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8_sm80 import _f32_to_e2m1_u8

from .nvidia.specialized_decode.ampere import reference

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="needs an SM80-or-newer CUDA GPU",
)

DEVICE = "cuda"
SWA_BLOCKS, SWA_BLOCK = 40, 32
KV_BLOCKS, KV_BLOCK = 20, 64


def _paged(blocks: int, block_size: int, record: int, pad: int) -> torch.Tensor:
    """A zeroed cache `[blocks, block_size, record]` with padded pages."""
    raw = torch.zeros(
        blocks, block_size * record + pad, dtype=torch.uint8, device=DEVICE
    )
    return raw[:, : block_size * record].view(blocks, block_size, record)


def _fill_mxfp8(cache: torch.Tensor) -> None:
    """Random MXFP8 records, some values E4M3 subnormals."""
    tokens = cache.shape[0] * cache.shape[1]
    x = torch.randn(tokens, 512, device=DEVICE) * 2
    x[:, :8] *= 1e-3
    blocks = x.unflatten(-1, (16, 32))
    amax = blocks.abs().amax(-1).clamp_min(1e-30)
    exponent = torch.ceil(torch.log2(amax / 448)).clamp(-127, 127)
    values = reference.f32_to_e4m3(blocks / torch.exp2(exponent)[..., None])
    records = torch.cat((values.flatten(-2), (exponent + 127).to(torch.uint8)), -1)
    index = torch.arange(tokens, device=DEVICE)
    cache[index // cache.shape[1], index % cache.shape[1]] = records


def _latents(kind: str, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 latents and a RoPE table that stress one side of the quantizer."""
    cos_sin = torch.randn(3 * tokens + 2, 64, device=DEVICE)
    if kind == "ties":
        # Every value lands on an e2m1 rounding threshold of its tile: the
        # tile amax is 6 * 2^e and the RoPE is the identity.
        grid = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.0, -0.0, -2.5, -5.0, 4.5],
            device=DEVICE,
        )
        x = grid[torch.randint(0, len(grid), (tokens, 512), device=DEVICE)]
        x[:, ::16] = 6.0
        x = x * torch.exp2(torch.randint(-12, 6, (tokens, 1), device=DEVICE))
        cos_sin[:, :32], cos_sin[:, 32:] = 1.0, 0.0
        return x.bfloat16(), cos_sin
    scale = {"normal": 1.0, "tiny": 1e-4, "huge": 3000.0}[kind]
    x = torch.randn(tokens, 512, device=DEVICE) * scale
    x[5] = 0
    return x.bfloat16(), cos_sin


@triton.jit
def _e2m1_kernel(x_ptr, out_ptr, N: tl.constexpr):
    i = tl.program_id(0) * N + tl.arange(0, N)
    tl.store(out_ptr + i, _f32_to_e2m1_u8(tl.load(x_ptr + i)))


def test_e2m1_encoder_rounds_to_nearest_even():
    """The software e2m1 encoder on every threshold, its FP32 neighbours,
    both signs of zero and values past the saturation point."""
    edges = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 0.0, 1e30])
    x = torch.cat(
        [torch.nextafter(edges, torch.tensor(s)) for s in (-1e38, 1e38)] + [edges]
    )
    x = torch.cat((x, -x, torch.randn(4096 - 2 * x.numel()) * 3)).to(DEVICE)
    got = torch.empty(4096, dtype=torch.uint8, device=DEVICE)
    _e2m1_kernel[(4,)](x, got, N=1024)
    assert torch.equal(got.int(), reference.f32_to_e2m1(x))


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("kind", ["normal", "ties", "tiny", "huge"])
def test_nvfp4_writer_matches_reference_bytes(ratio, kind):
    """The Triton NVFP4 insert stores `reference.nvfp4_kv_record` byte for
    byte -- and so the bytes of the SM80 CuTe DSL writer -- leaving skipped
    slots, open groups and page padding untouched."""
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        rope_quant_insert,
    )

    torch.manual_seed(0)
    tokens, block_size = 4096, 64
    blocks = tokens // block_size + 1
    latent, cos_sin = _latents(kind, tokens)
    positions = torch.arange(tokens, device=DEVICE) * 3 + 1
    slots = torch.randperm(blocks * block_size, device=DEVICE)[:tokens]
    slots[7] = -1
    keep = (slots >= 0) & ((positions + 1) % ratio == 0)
    want = _paged(blocks, block_size, 288, 64)
    want[slots[keep] // block_size, slots[keep] % block_size] = (
        reference.nvfp4_kv_record(latent[keep], positions[keep], cos_sin, ratio)
    )
    got = _paged(blocks, block_size, 288, 64)
    rope_quant_insert(latent, positions, cos_sin, got, slots, ratio)
    assert want.any()
    torch.testing.assert_close(got, want, rtol=0, atol=0)

    if current_platform.is_device_capability(80):
        from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import quantize

        cute = _paged(blocks, block_size, 288, 64)
        quantize.store_compressed_records(
            latent, positions, slots, cos_sin, ratio, cute
        )
        torch.testing.assert_close(got, cute, rtol=0, atol=0)


@pytest.mark.parametrize("record", [528, 288])
def test_gather_decodes_reference_values(record):
    """The prefill gather reads both V4.1 records exactly (BF16 holds every
    decoded value)."""
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        dequantize_and_gather_k_cache,
    )

    torch.manual_seed(0)
    block_size, blocks, tokens = 64, 6, 300
    cache = _paged(blocks, block_size, record, 64)
    if record == 528:
        _fill_mxfp8(cache)
        decode = reference.decode_mxfp8_record
    else:
        cache[:] = torch.randint(0, 256, cache.shape, device=DEVICE).to(torch.uint8)
        # No NaN scale codes: a writer never stores one.
        cache[..., 256:] &= 0x7E
        decode = reference.decode_nvfp4_record
    block_table = torch.randperm(blocks, device=DEVICE).int()[None]
    seq_lens = torch.tensor([tokens], dtype=torch.int32, device=DEVICE)
    out = torch.zeros(1, tokens, 512, dtype=torch.bfloat16, device=DEVICE)
    dequantize_and_gather_k_cache(
        out, cache, seq_lens, None, block_table, block_size, offset=0
    )
    index = torch.arange(tokens, device=DEVICE)
    want = decode(cache[block_table[0, index // block_size].long(), index % block_size])
    torch.testing.assert_close(out[0].float(), want, rtol=0, atol=0)


@pytest.mark.parametrize(
    "tokens,heads,empty,window_only,compressed_record",
    [
        (1, 8, False, False, 288),
        (4, 8, False, False, 288),
        (3, 16, True, False, 288),
        (70, 8, False, False, 288),
        (5, 20, False, False, 288),
        (300, 4, False, False, 288),
        (6, 8, True, True, 288),
        (40, 8, False, True, 288),
        (4, 8, False, False, 528),
    ],
)
def test_decode_matches_reference(tokens, heads, empty, window_only, compressed_record):
    """Split-K decode over the MXFP8 window cache and the NVFP4 (or MXFP8)
    compressed cache against `reference.sparse_mla`, within the BF16 rounding
    of the probabilities (the bound of the SM80 CuTe DSL kernel's own test):
    ragged lists, empty rows, unselected entries, window-only layers, padded
    pages and head counts that do not fill a head block."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode

    torch.manual_seed(0)
    window, top_k, scale = 128, 512, 512**-0.5
    cos_sin = torch.randn(4096, 64, device=DEVICE)
    swa = _paged(SWA_BLOCKS, SWA_BLOCK, 528, 512 - (SWA_BLOCK * 528) % 512)
    _fill_mxfp8(swa)
    compressed = _paged(KV_BLOCKS, KV_BLOCK, compressed_record, 64)
    if compressed_record == 288:
        index = torch.arange(KV_BLOCKS * KV_BLOCK, device=DEVICE)
        latents = torch.randn(index.numel(), 512, device=DEVICE).bfloat16()
        compressed[index // KV_BLOCK, index % KV_BLOCK] = reference.nvfp4_kv_record(
            latents, index, cos_sin, 1
        )
    else:
        _fill_mxfp8(compressed)
    q = torch.randn(tokens, heads, 512, device=DEVICE).bfloat16()
    swa_indices = torch.randint(
        0, SWA_BLOCKS * SWA_BLOCK, (tokens, window), device=DEVICE, dtype=torch.int32
    )
    swa_lens = torch.randint(
        0 if empty else window // 2, window + 1, (tokens,), device=DEVICE
    ).int()
    topk_indices = torch.randint(
        0, KV_BLOCKS * KV_BLOCK, (tokens, top_k), device=DEVICE, dtype=torch.int32
    )
    topk_lens = torch.randint(0, top_k + 1, (tokens,), device=DEVICE).int()
    if empty:
        swa_lens[0] = 0
        topk_lens[0] = 0
    topk_indices[:, 7::50] = -1
    sink = torch.randn(heads, device=DEVICE)
    positions = torch.randint(0, 4096, (tokens,), device=DEVICE)

    if compressed_record == 288:
        want = reference.sparse_mla(
            q,
            swa,
            swa_indices,
            swa_lens,
            None if window_only else compressed,
            None if window_only else topk_indices,
            None if window_only else topk_lens,
            sink,
            scale,
            positions,
            cos_sin,
        )
    else:
        # The reference reads MXFP8 keys from its window list only: hand it
        # both caches as one.
        keys = torch.cat((swa.flatten(0, 1), compressed.flatten(0, 1)))[:, None]
        lanes = torch.arange(top_k, device=DEVICE)
        both = torch.cat(
            (
                torch.where(
                    torch.arange(window, device=DEVICE) < swa_lens[:, None],
                    swa_indices,
                    -1,
                ),
                torch.where(
                    (lanes < topk_lens[:, None]) & (topk_indices >= 0),
                    topk_indices + SWA_BLOCKS * SWA_BLOCK,
                    -1,
                ),
            ),
            dim=1,
        )
        want = reference.sparse_mla(
            q,
            keys,
            both,
            torch.full_like(swa_lens, window + top_k),
            None,
            None,
            None,
            sink,
            scale,
            positions,
            cos_sin,
        )

    got = torch.full_like(q, float("nan"))
    rotated = rocm_sparse_attn_decode(
        q=q,
        kv_cache=None if window_only else compressed,
        swa_k_cache=swa,
        swa_only=window_only,
        topk_indices=None if window_only else topk_indices,
        topk_lens=topk_lens,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_ragged_indices=None,
        swa_ragged_indptr=None,
        topk_ragged_indices=None,
        topk_ragged_indptr=None,
        attn_sink=sink,
        scale=scale,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        output=got,
        inv_rope_positions=positions,
        inv_rope_cos_sin_cache=cos_sin,
    )
    assert rotated == tokens
    assert not got.isnan().any()
    assert (got.float() - want.float()).abs().max() < 2e-2 * want.float().abs().max()


@pytest.mark.parametrize("capability,want", [(80, True), (89, True), (90, False)])
def test_triton_route_record_choice(monkeypatch, capability, want):
    """`nvfp4_ds_mla` selects the V4.1 records on SM8x only; the route's
    default `fp8_ds_mla` and every other platform keep their records."""
    from vllm.models.deepseek_v41 import ampere
    from vllm.platforms.interface import DeviceCapability

    device = DeviceCapability(capability // 10, capability % 10)
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda device_id=0: device
    )
    assert ampere.triton_reads_v41_records("nvfp4_ds_mla") is want
    assert not ampere.triton_reads_v41_records("fp8_ds_mla")
    assert not ampere.triton_reads_v41_records(None)
