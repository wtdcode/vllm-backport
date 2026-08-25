# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.model_executor.kernels.mhc  # noqa: F401
from vllm.model_executor.kernels.mhc.tilelang import (
    _tilelang_hc_prenorm_gemm,
    _torch_hc_prenorm_gemm,
    mhc_pre_broadcast_tilelang,
)
from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC pre reference kernel from tilelang repo: https://github.com/tile-ai/tilelang/blob/d135bd1cd2d2eee74fbb41dd0a0831a427194c86/examples/deepseek_mhc/example_mhc_pre.py#L303"""
    hc_mult = residual.shape[-2]

    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    )

    hc_scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ],
    )
    mixes = mixes * hc_scale + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)

    res_mix = sinkhorn_normalize_ref(
        res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps
    )

    layer_input = (residual * pre_mix).sum(-2).bfloat16()

    return post_mix, res_mix, layer_input


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """mHC post reference kernel from tilelang repo: https://github.com/tile-ai/tilelang/blob/d135bd1cd2d2eee74fbb41dd0a0831a427194c86/examples/deepseek_mhc/example_mhc_post.py#L68"""
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def hc_head_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    residual_flat = residual.flatten(-2).float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre_mix = torch.nn.functional.linear(residual_norm, fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * residual.float(), dim=-2).bfloat16()


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("fuse_norm", [False, True])
def test_mhc_pre_tilelang(num_tokens, hidden_size, hc_mult, fuse_norm):
    """``fuse_norm`` selects the RMSNorm-fused kernel, which is the variant
    DeepSeek V4 actually runs (the model always passes ``norm_weight``)."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = 2 * hc_mult + hc_mult2
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(hc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float) * 0.1

    hc_sinkhorn_eps = hc_pre_eps = rms_eps = 1e-6
    sinkhorn_repeat = 20
    hc_post_alpha = 1.0
    norm_eps = 1e-5
    norm_weight = (
        torch.randn((hidden_size,), dtype=torch.bfloat16) * 0.1 + 1.0
        if fuse_norm
        else None
    )

    post_mix_ref, res_mix_ref, layer_input_ref = mhc_pre_ref(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )
    if norm_weight is not None:
        li = layer_input_ref.float()
        layer_input_ref = (
            li
            * torch.rsqrt(li.square().mean(-1, keepdim=True) + norm_eps)
            * norm_weight.float()
        ).bfloat16()

    out = torch.ops.vllm.mhc_pre_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
        1,
        norm_weight,
        norm_eps,
    )

    ref = (post_mix_ref, res_mix_ref, layer_input_ref)
    for actual, expected in zip(out, ref, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre_broadcast_tilelang(num_tokens, hidden_size, hc_mult):
    """First-layer variant: the (T, H) residual is broadcast across the
    hc_mult streams, so the result must match ``mhc_pre_ref`` on the
    explicitly expanded residual. RMSNorm fusion is mandatory here (the
    wrapper asserts ``norm_weight``), matching how the model calls it."""
    _run_mhc_pre_broadcast_case(num_tokens, hidden_size, hc_mult)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre_broadcast_tilelang_without_deep_gemm(
    num_tokens, hidden_size, hc_mult, monkeypatch
):
    """Same case with DeepGEMM forced unavailable.

    The broadcast variant is the one that reaches the prenorm GEMM through a
    separate branch from its two siblings, and on a pre-Hopper device the
    DeepGEMM branch aborts in ``hyperconnection.hpp`` rather than returning
    wrong numbers. Below SM90 this is what the test above already runs, so
    the point of forcing it is coverage on hardware where DeepGEMM *is*
    supported and the fallback would otherwise never execute.
    """
    monkeypatch.setattr(
        "vllm.utils.deep_gemm.is_deep_gemm_supported", lambda *a, **kw: False
    )
    _run_mhc_pre_broadcast_case(num_tokens, hidden_size, hc_mult)


def _run_mhc_pre_broadcast_case(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = 2 * hc_mult + hc_mult2
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(hc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    # The model precomputes fn_broadcast this way in
    # finalize_mhc_broadcast_weights: summing fn over the hc_mult axis is
    # exactly the GEMM against a residual that is identical in every stream.
    fn_broadcast = fn.view(hc_mult3, hc_mult, hidden_size).sum(dim=1)
    hc_scale = torch.randn((3,), dtype=torch.float) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float) * 0.1

    hc_sinkhorn_eps = hc_pre_eps = rms_eps = 1e-6
    sinkhorn_repeat = 20
    hc_post_alpha = 1.0
    norm_eps = 1e-5
    norm_weight = torch.randn((hidden_size,), dtype=torch.bfloat16) * 0.1 + 1.0

    residual_ref = x.unsqueeze(-2).expand(num_tokens, hc_mult, hidden_size)
    post_mix_ref, res_mix_ref, layer_input_ref = mhc_pre_ref(
        residual_ref,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )
    li = layer_input_ref.float()
    layer_input_ref = (
        li
        * torch.rsqrt(li.square().mean(-1, keepdim=True) + norm_eps)
        * norm_weight.float()
    ).bfloat16()

    residual_out, post_mix, res_mix, layer_input = mhc_pre_broadcast_tilelang(
        x,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
        1,
        norm_weight,
        norm_eps,
        fn_broadcast=fn_broadcast,
    )

    # The materialized broadcast must be an exact copy of the input rows.
    torch.testing.assert_close(residual_out, residual_ref.contiguous(), rtol=0, atol=0)
    torch.testing.assert_close(post_mix, post_mix_ref, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(res_mix, res_mix_ref, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(layer_input, layer_input_ref, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize(
    ("num_tokens", "hidden_size"),
    [
        # T crosses the routing boundary (_PRENORM_SMALL_T = 32): below it the
        # one-CTA-per-token tilelang kernel runs (fp32 fn, strict tolerance);
        # at and above it the cuBLAS bf16 route runs, whose fn and output are
        # rounded to bf16, bounded at rel 5e-3 directly on `out` — the
        # downstream mhc_pre tolerance alone cannot distinguish bf16-fn
        # rounding from a broken kernel.
        (1, 1280),
        (31, 1280),
        (32, 1280),
        (512, 1280),
        (2048, 1280),
        (1, 4096),
        (31, 4096),
        (32, 4096),
        (64, 4096),
        (512, 4096),
        (2048, 4096),
        (1, 7168),
        (31, 7168),
        (32, 7168),
        (64, 7168),
        (512, 7168),
        (2048, 7168),
    ],
)
def test_hc_prenorm_gemm_tilelang(num_tokens, hidden_size):
    from vllm.model_executor.kernels.mhc.tilelang import _PRENORM_SMALL_T

    torch.set_default_device(DEVICE)
    set_random_seed(0)

    hc_mult = 4
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    x = torch.randn((num_tokens, hc_mult * hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult3, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    out_ref = torch.empty((1, num_tokens, hc_mult3), dtype=torch.float32)
    sqrsum_ref = torch.empty((1, num_tokens), dtype=torch.float32)
    out = torch.empty_like(out_ref)
    sqrsum = torch.empty_like(sqrsum_ref)

    _torch_hc_prenorm_gemm(x, fn, out_ref, sqrsum_ref)
    _tilelang_hc_prenorm_gemm(x, fn, out, sqrsum, hidden_size, hc_mult)

    if num_tokens < _PRENORM_SMALL_T:
        torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-4)
    else:
        # bf16 fn plus bf16 GEMM output: rel 5e-3 against the tensor scale.
        scale = float(out_ref.abs().max())
        torch.testing.assert_close(out, out_ref, atol=5e-3 * scale, rtol=5e-3)
    torch.testing.assert_close(sqrsum, sqrsum_ref, atol=8.0, rtol=5e-4)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_post_tilelang(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((num_tokens, hc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32)

    ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
    out = torch.ops.vllm.mhc_post_tilelang(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )

    torch.testing.assert_close(out, ref, atol=5e-2, rtol=1e-2)


def test_prenorm_shard_requires_sqrsum_fold(monkeypatch):
    from vllm.model_executor.kernels.mhc import tilelang as tilelang_mod

    tilelang_mod._fuse_sqrsum_enabled.cache_clear()
    monkeypatch.setattr(tilelang_mod.envs, "VLLM_MHC_PRENORM_SHARD", True)
    monkeypatch.setattr(tilelang_mod.envs, "VLLM_MHC_POST_FUSE_SQRSUM", False)

    with pytest.raises(ValueError, match="VLLM_MHC_POST_FUSE_SQRSUM=1"):
        tilelang_mod.validate_mhc_optimization_flags()

    tilelang_mod._fuse_sqrsum_enabled.cache_clear()


@pytest.mark.parametrize("sqrsum_ready", [False, True])
def test_prenorm_router_skips_a_completed_sqrsum(monkeypatch, sqrsum_ready):
    import importlib

    from vllm.model_executor.kernels.mhc import tilelang as tilelang_mod

    triton_mod = importlib.import_module("vllm.model_executor.kernels.mhc.triton")
    seen = []
    monkeypatch.setattr(
        triton_mod,
        "hc_prenorm_gemm_cublas",
        lambda x, fn, out, sqrsum: seen.append(sqrsum),
    )
    hidden, hc_mult, num_tokens = 4096, 4, 64
    k = hidden * hc_mult
    x = torch.zeros(num_tokens, k, dtype=torch.bfloat16)
    fn = torch.zeros(24, k, dtype=torch.float32)
    out = torch.zeros(1, num_tokens, 24, dtype=torch.float32)
    sqrsum = torch.zeros(1, num_tokens, dtype=torch.float32)

    tilelang_mod._tilelang_hc_prenorm_gemm(
        x,
        fn,
        out,
        sqrsum,
        hidden,
        hc_mult,
        sqrsum_ready=sqrsum_ready,
    )
    assert len(seen) == 1
    assert seen[0] is (None if sqrsum_ready else sqrsum)


def test_int8_all_reduce_requires_custom_all_reduce(monkeypatch):
    from types import SimpleNamespace

    import vllm.distributed.parallel_state as parallel_state
    from vllm.model_executor.kernels.mhc.ar_int8 import assert_hoist_preconditions

    tp_group = SimpleNamespace(
        device_communicator=SimpleNamespace(ca_comm=None),
    )
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=64),
    )

    with pytest.raises(AssertionError, match="requires custom all-reduce"):
        assert_hoist_preconditions(config)


def test_mhc_custom_op_fake_signatures_match():
    import inspect

    from vllm.model_executor.kernels.mhc import tilelang as tilelang_mod

    pairs = (
        (
            tilelang_mod.mhc_post_tilelang,
            tilelang_mod._mhc_post_tilelang_fake,
        ),
        (
            tilelang_mod.mhc_fused_post_pre_tilelang,
            tilelang_mod._mhc_fused_post_pre_tilelang_fake,
        ),
    )
    for implementation, fake in pairs:
        assert inspect.signature(implementation) == inspect.signature(fake)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [32, 2048])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
def test_mhc_post_sqrsum_matches_standalone(num_tokens, hidden_size):
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_post_sqrsum_tilelang,
    )

    torch.set_default_device(DEVICE)
    set_random_seed(0)
    hc_mult = 4
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((num_tokens, hc_mult), dtype=torch.float32)
    comb_res_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32)

    out = torch.empty_like(residual)
    sqrsum = torch.empty((1, num_tokens), dtype=torch.float32)
    mhc_post_sqrsum_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix,
        x,
        out,
        sqrsum,
        hc_mult,
        hidden_size,
    )

    ref = mhc_post_ref(x, residual, post_layer_mix.unsqueeze(-1), comb_res_mix)
    ref_sqrsum = out.float().square().sum(dim=(-2, -1)).unsqueeze(0)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(sqrsum, ref_sqrsum, atol=0, rtol=1e-4)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("fuse_sqrsum", [False, True])
def test_mhc_post_int8_matches_dequantized_reference(fuse_sqrsum):
    from vllm.model_executor.kernels.mhc.ar_int8 import QUANT_BLOCK
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_post_int8_tilelang,
        mhc_post_sqrsum_int8_tilelang,
    )

    torch.set_default_device(DEVICE)
    set_random_seed(0)
    num_tokens, hc_mult, hidden_size = 32, 4, 4096
    x_q = torch.randint(-127, 128, (num_tokens, hidden_size), dtype=torch.int8)
    x_s = torch.rand((num_tokens, hidden_size // QUANT_BLOCK), dtype=torch.bfloat16)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((num_tokens, hc_mult), dtype=torch.float32)
    comb_res_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32)

    out = torch.empty_like(residual)
    args = [
        comb_res_mix,
        residual,
        post_layer_mix,
        x_q,
        x_s,
        out,
    ]
    sqrsum = torch.empty((1, num_tokens), dtype=torch.float32)
    if fuse_sqrsum:
        args.append(sqrsum)
        kernel = mhc_post_sqrsum_int8_tilelang
    else:
        kernel = mhc_post_int8_tilelang
    kernel(*args, hc_mult, hidden_size, QUANT_BLOCK)

    x_dequant = x_q.float() * x_s.float().repeat_interleave(QUANT_BLOCK, dim=-1)
    ref = (
        x_dequant.unsqueeze(-2) * post_layer_mix.unsqueeze(-1)
        + torch.bmm(comb_res_mix.mT, residual.float())
    ).bfloat16()
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=1e-2)
    if fuse_sqrsum:
        ref_sqrsum = out.float().square().sum(dim=(-2, -1)).unsqueeze(0)
        torch.testing.assert_close(sqrsum, ref_sqrsum, atol=0, rtol=1e-4)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_fused_post_pre(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((num_tokens, hc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32)

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(hc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float) * 0.1

    hc_sinkhorn_eps = hc_pre_eps = rms_eps = 1e-6
    sinkhorn_repeat = 20
    hc_post_alpha = 1.0

    def run_ref():
        residual_ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
        post_mix_ref, res_mix_ref, layer_input_ref = mhc_pre_ref(
            residual_ref,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_alpha,
            sinkhorn_repeat,
        )
        return residual_ref, post_mix_ref, res_mix_ref, layer_input_ref

    residual_ref, post_mix_ref, res_mix_ref, layer_input_ref = run_ref()

    residual, post_mix, res_mix, x = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )

    torch.testing.assert_close(residual, residual_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(post_mix, post_mix_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(res_mix, res_mix_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(x, layer_input_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_hc_head_triton(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    hc_scale = torch.randn((1,), dtype=torch.float32) * 0.1
    hc_base = torch.randn((hc_mult,), dtype=torch.float32) * 0.1
    rms_eps = hc_eps = 1e-6

    out = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16)
    out.fill_(float("nan"))

    result = torch.ops.vllm.hc_head_triton(
        residual,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )

    assert result is None
    assert not torch.isnan(out).any()

    out_ref = hc_head_ref(residual, fn, hc_scale, hc_base, rms_eps, hc_eps)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_hc_head_tilelang(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    hc_scale = torch.randn((1,), dtype=torch.float32) * 0.1
    hc_base = torch.randn((hc_mult,), dtype=torch.float32) * 0.1
    rms_eps = hc_eps = 1e-6

    out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
    )

    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()

    out_ref = hc_head_ref(residual, fn, hc_scale, hc_base, rms_eps, hc_eps)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=1e-2)
