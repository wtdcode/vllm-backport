# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`fuse_input_gemm_weights` merges the three bf16 attention input GEMMs.

The method is exercised unbound against a stub carrying only the attributes it
reads, so the test needs no model, no config and no weights on disk. What it
has to pin is that the merge is a re-layout and not a copy: after fusing, each
module's weight must be the same values it had, backed by the concatenated
buffer, because `attn_gemm_parallel_execute` splits that buffer's output by
those same row counts and the un-fused fallback still reads the module weights.
"""

import pytest
import torch
from torch import nn

from vllm.distributed.utils import balanced_row_counts
from vllm.models.deepseek_v4.attention import (
    _UNREPLICATE_MIN_TOKENS,
    DeepseekV4Attention,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="the fusion is gated to CUDA (XPU's indexer op wants bf16 weights)",
)

HIDDEN = 256
# Row counts in the checkpoint's proportion (compressor, indexer compressor,
# weights_proj), scaled down.
N_COMP, N_IDX, N_WP = 128, 32, 8


def _linear(n: int, device: str) -> nn.Module:
    mod = nn.Module()
    mod.weight = nn.Parameter(
        torch.randn(n, HIDDEN, dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )
    return mod


def _stub(device: str = "cuda"):
    compressor = nn.Module()
    compressor.fused_wkv_wgate = _linear(N_COMP, device)
    indexer = nn.Module()
    indexer.compressor = nn.Module()
    indexer.compressor.fused_wkv_wgate = _linear(N_IDX, device)
    indexer.weights_proj = _linear(N_WP, device)

    stub = nn.Module()
    stub.compressor = compressor
    stub.indexer = indexer
    stub.hidden_size = HIDDEN
    stub.fused_input_weight = None
    stub.fused_input_splits = []
    return stub


def test_fuse_input_gemm_weights_preserves_values_and_layout():
    stub = _stub()
    originals = [
        stub.compressor.fused_wkv_wgate.weight.clone(),
        stub.indexer.compressor.fused_wkv_wgate.weight.clone(),
        stub.indexer.weights_proj.weight.clone(),
    ]

    DeepseekV4Attention.fuse_input_gemm_weights(stub)

    assert stub.fused_input_splits == [N_COMP, N_IDX, N_WP]
    merged = stub.fused_input_weight
    assert merged.shape == (N_COMP + N_IDX + N_WP, HIDDEN)
    assert merged.is_contiguous()

    weights = [
        stub.compressor.fused_wkv_wgate.weight,
        stub.indexer.compressor.fused_wkv_wgate.weight,
        stub.indexer.weights_proj.weight,
    ]
    for original, weight in zip(originals, weights):
        assert torch.equal(weight, original)
        # A row-slice of a contiguous [N, K] tensor is contiguous, which is
        # what lets the modules keep using their weights unchanged.
        assert weight.is_contiguous()
        assert weight.data_ptr() >= merged.data_ptr()
        assert (
            weight.untyped_storage().data_ptr() == merged.untyped_storage().data_ptr()
        )

    # Splitting the merged GEMM's output must reproduce the separate GEMMs.
    x = torch.randn(3, HIDDEN, dtype=torch.bfloat16, device=merged.device)
    fused_out = torch.mm(x, merged.T, out_dtype=torch.float32).split(
        stub.fused_input_splits, dim=-1
    )
    for original, part in zip(originals, fused_out):
        reference = torch.mm(x, original.T, out_dtype=torch.float32)
        torch.testing.assert_close(part, reference, rtol=1e-5, atol=1e-5)


def test_fuse_input_gemm_weights_skips_layers_without_an_indexer():
    stub = _stub()
    stub.indexer = None
    DeepseekV4Attention.fuse_input_gemm_weights(stub)
    assert stub.fused_input_weight is None

    stub = _stub()
    stub.compressor = None
    DeepseekV4Attention.fuse_input_gemm_weights(stub)
    assert stub.fused_input_weight is None


def _token_shard_roundtrip(n_tokens: int, tp: int, hidden: int, n_out: int):
    """Run the shard/gather round-trip for every rank and rebuild the output.

    `all_gatherv` needs a real process group, so the collective is stood in
    for by concatenating what each rank computed — which is exactly what a
    size-aware gather along dim 0 produces.
    """
    import vllm.models.deepseek_v4.attention as attn_mod

    x = torch.randn(n_tokens, hidden, dtype=torch.bfloat16)
    w = torch.randn(n_out, hidden, dtype=torch.bfloat16) / 10
    # Explicit upcast rather than mm(out_dtype=): the sharding arithmetic is
    # device-independent, so this half of the file runs without a GPU.
    reference = x.float() @ w.float().T

    real_rank = attn_mod.get_tensor_model_parallel_rank
    real_world = attn_mod.get_tensor_model_parallel_world_size
    parts = []
    try:
        attn_mod.get_tensor_model_parallel_world_size = lambda: tp
        for rank in range(tp):
            attn_mod.get_tensor_model_parallel_rank = lambda r=rank: r
            shard, rows = DeepseekV4Attention._shard_tokens(x)
            assert rows == balanced_row_counts(n_tokens, tp)
            assert shard.shape[0] == rows[rank], "slice must match its size"
            assert max(rows) - min(rows) <= 1, "split must be balanced"
            assert sum(rows) == n_tokens, "split must cover every token once"
            parts.append(shard.float() @ w.float().T)
    finally:
        attn_mod.get_tensor_model_parallel_rank = real_rank
        attn_mod.get_tensor_model_parallel_world_size = real_world

    gathered = torch.cat(parts, dim=0)
    return gathered, reference


@pytest.mark.parametrize("n_tokens", [8192, 8191, 1024, 9])
@pytest.mark.parametrize("tp", [8, 4])
def test_token_sharding_matches_the_replicated_gemm(n_tokens, tp):
    """Sharding the token dim must reproduce the replicated GEMM.

    Correctness gate for VLLM_UNREPLICATE_ATTN_GEMMS. Rows are partitioned,
    never duplicated, so each output row is computed by one rank from the same
    inputs — the arithmetic this pins is that the shards tile the batch exactly
    and the zero padding never leaks into the kept rows. Ragged widths are
    covered because prefill batches are not multiples of the TP size.

    On the *exactness* of the result: at every width the flag actually runs at
    (>= _UNREPLICATE_MIN_TOKENS) this is bit-identical, which is why the
    threshold is asserted here and not only in the perf argument.

    Below the threshold only the partition is pinned, not the arithmetic. The
    balanced split hands some ranks a 1-row GEMM at tiny widths, and a 1-row
    matmul takes a different library path than a 9-row one, so the fp32
    results diverge by more than bf16 rounding absorbs. That is a property of
    the BLAS and it is unreachable in production — the flag does not engage
    below the threshold — so asserting numerics there would be pinning a
    configuration that never runs.
    """
    gathered, reference = _token_shard_roundtrip(n_tokens, tp, HIDDEN, N_COMP)
    assert gathered.shape == reference.shape
    if n_tokens >= _UNREPLICATE_MIN_TOKENS:
        torch.testing.assert_close(gathered, reference, rtol=0, atol=0)


def _gemm_stub(*, with_trio: bool, device: str = "cuda"):
    """Minimal stand-in for `attn_gemm_parallel_execute`'s reads.

    `aux_stream_list=None` takes `execute_in_parallel`'s sequential path, so
    the ln_events are never touched and only the GEMMs themselves need a
    device (`mm(out_dtype=)` is CUDA-only).
    """
    stub = nn.Module()
    stub.aux_stream_list = None
    stub.ln_events = [None] * 4
    stub._multi_stream_threshold = 0
    stub._unreplicate_gemms = True
    stub._unreplicate_all_layers = True
    stub._unreplicate_tokens = lambda n: n >= _UNREPLICATE_MIN_TOKENS
    stub.compressor = None
    stub.indexer = None
    stub.fused_input_splits = []
    stub.fused_input_weight = None
    if with_trio:
        stub.fused_input_weight = torch.zeros(
            N_COMP + N_IDX + N_WP, HIDDEN, dtype=torch.bfloat16, device=device
        )
        stub.fused_input_splits = [N_COMP, N_IDX, N_WP]

    seen: dict[str, int] = {}

    def fused_wqa_wkv_gemm(x):
        seen["wqa_wkv_rows"] = x.shape[0]
        return x.new_zeros(x.shape[0], 4)

    stub._fused_wqa_wkv_gemm = fused_wqa_wkv_gemm
    stub._shard_tokens = staticmethod(DeepseekV4Attention._shard_tokens).__func__
    # The collective is stood in for by its own contract: reassemble the rows
    # each rank owned. Only this rank's slice exists here, so widen to the
    # full batch the same way all_gatherv would.
    stub._gather_tokens = lambda y, rows: y.new_zeros(sum(rows), *y.shape[1:])
    return stub, seen


@pytest.mark.parametrize("with_trio", [True, False])
def test_token_shard_reaches_layers_without_the_fused_trio(with_trio):
    """The shard must not be gated on the merged input trio.

    `fused_wqa_wkv` is replicated on every rank in both branches, so both must
    hand it this rank's slice. Gating the shard on `fused_input_weight` (a
    proxy for "has an indexer") silently left the 22 ratio-128/SWA layers
    computing all 15,360 rows on all 8 ranks — the reachability bug rule 49
    names. This pins the fix at the branch, not at the flag.
    """
    import vllm.models.deepseek_v4.attention as attn_mod

    n_tokens, tp = 8192, 8
    stub, seen = _gemm_stub(with_trio=with_trio)
    x = torch.randn(n_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")

    real_rank = attn_mod.get_tensor_model_parallel_rank
    real_world = attn_mod.get_tensor_model_parallel_world_size
    try:
        attn_mod.get_tensor_model_parallel_world_size = lambda: tp
        attn_mod.get_tensor_model_parallel_rank = lambda: 0
        qr_kv, *_ = DeepseekV4Attention.attn_gemm_parallel_execute(stub, x)
    finally:
        attn_mod.get_tensor_model_parallel_rank = real_rank
        attn_mod.get_tensor_model_parallel_world_size = real_world

    assert seen["wqa_wkv_rows"] == balanced_row_counts(n_tokens, tp)[0]
    assert qr_kv.shape[0] == n_tokens, "the gather must restore the full batch"


def test_token_shard_still_declines_below_the_threshold():
    """Decode widths must keep the replicated path in both branches."""
    import vllm.models.deepseek_v4.attention as attn_mod

    n_tokens, tp = 8, 8
    stub, seen = _gemm_stub(with_trio=False)
    x = torch.randn(n_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")

    real_world = attn_mod.get_tensor_model_parallel_world_size
    try:
        attn_mod.get_tensor_model_parallel_world_size = lambda: tp
        qr_kv, *_ = DeepseekV4Attention.attn_gemm_parallel_execute(stub, x)
    finally:
        attn_mod.get_tensor_model_parallel_world_size = real_world

    assert seen["wqa_wkv_rows"] == n_tokens
    assert qr_kv.shape[0] == n_tokens


@pytest.mark.parametrize("use_int8", [False, True])
def test_hoisted_all_reduce_readds_the_suppressed_collective(monkeypatch, use_int8):
    import vllm.models.deepseek_v4.nvidia.model as model_mod

    stub = nn.Module()
    stub._ar_hoisted = True
    x = torch.ones(32, HIDDEN, dtype=torch.bfloat16)
    reduced = x + 1
    codes = torch.ones_like(x, dtype=torch.int8)
    scales = torch.ones(32, HIDDEN // 32, dtype=torch.bfloat16)
    seen: list[str] = []

    monkeypatch.setattr(model_mod, "use_int8_for", lambda _: use_int8)

    def reduce_bf16(_):
        seen.append("bf16")
        return reduced

    def reduce_int8(_):
        seen.append("int8")
        return codes, scales

    monkeypatch.setattr(
        model_mod,
        "tensor_model_parallel_all_reduce",
        reduce_bf16,
    )
    monkeypatch.setattr(
        model_mod,
        "int8_all_reduce",
        reduce_int8,
    )

    actual, actual_scales = model_mod.DeepseekV4DecoderLayer._hoisted_all_reduce(
        stub, x
    )
    if use_int8:
        assert actual is codes
        assert actual_scales is scales
        assert seen == ["int8"]
    else:
        assert actual is reduced
        assert actual_scales is None
        assert seen == ["bf16"]


def test_decoder_carries_int8_scales_to_each_mhc_post(monkeypatch):
    import vllm.models.deepseek_v4.nvidia.model as model_mod

    class Norm:
        weight = torch.ones(HIDDEN, dtype=torch.bfloat16)
        variance_epsilon = 1e-5

    stub = nn.Module()
    stub.attn_norm = Norm()
    stub.ffn_norm = Norm()
    stub.attn = lambda positions, x, _: x
    stub.ffn = lambda x, input_ids: x
    stub.hc_attn_fn = torch.empty(24, HIDDEN * 4)
    stub.hc_ffn_fn = torch.empty(24, HIDDEN * 4)
    stub.hc_attn_scale = torch.empty(3)
    stub.hc_ffn_scale = torch.empty(3)
    stub.hc_attn_base = torch.empty(24)
    stub.hc_ffn_base = torch.empty(24)
    stub.rms_norm_eps = 1e-6
    stub.hc_eps = 1e-6
    stub.hc_post_alpha = 2.0
    stub.hc_sinkhorn_iters = 20

    x = torch.ones(4, HIDDEN, dtype=torch.int8)
    residual = torch.ones(4, 4, HIDDEN, dtype=torch.bfloat16)
    post_mix = torch.ones(4, 4, 1)
    res_mix = torch.ones(4, 4, 4)
    incoming_scales = torch.ones(4, HIDDEN // 32, dtype=torch.bfloat16)
    attn_scales = incoming_scales + 1
    ffn_scales = incoming_scales + 2
    reductions = iter(((x, attn_scales), (x, ffn_scales)))
    stub._hoisted_all_reduce = lambda _: next(reductions)
    seen_scales = []

    def fused_post_pre(
        x,
        residual,
        post_mix,
        res_mix,
        *args,
        x_scales=None,
        **kwargs,
    ):
        seen_scales.append(x_scales)
        return residual, post_mix, res_mix, x.to(torch.bfloat16)

    monkeypatch.setattr(model_mod, "mhc_fused_post_pre_tilelang", fused_post_pre)

    *_, returned_scales = model_mod.DeepseekV4DecoderLayer.forward(
        stub,
        x,
        torch.arange(4),
        None,
        post_mix,
        res_mix,
        residual,
        incoming_scales,
    )
    assert seen_scales[0] is incoming_scales
    assert seen_scales[1] is attn_scales
    assert returned_scales is ffn_scales
