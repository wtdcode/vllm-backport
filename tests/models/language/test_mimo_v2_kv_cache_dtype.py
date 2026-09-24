# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo-V2 attention modules must thread cache_config into Attention.

`--kv-cache-dtype fp8` only halves the KV pool if every Attention layer
resolves its spec dtype from cache_config.cache_dtype. The MiMo decoder
layers (both the full-attention and the compressed-softmax/SWA branches)
and the MTP predictor layer historically constructed MiMoV2Attention
without cache_config, so the layers silently fell back to "auto" (model
dtype) specs: fp8 kernels ran on a bf16-sized allocation — correct output,
same token capacity, half of every cache row dead.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.attention.attention as attention_module
import vllm.model_executor.models.mimo_v2 as mimo_v2_module
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.mimo_v2 import MiMoV2FlashDecoderLayer
from vllm.model_executor.models.mimo_v2_mtp import MiMoV2MTPLayer
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=64,
        intermediate_size=128,
        hidden_act="silu",
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        v_head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=16,
        swa_v_head_dim=8,
        sliding_window_size=4,
        attention_bias=False,
        add_swa_attention_sink_bias=False,
        rope_theta=1000000.0,
        swa_rope_theta=1000000.0,
        max_position_embeddings=32768,
        attention_value_scale=None,
        partial_rotary_factor=1.0,
        layernorm_epsilon=1e-5,
        # layer 0: full attention; layer 1: compressed softmax (SWA)
        hybrid_layer_pattern=[0, 1],
        # no moe_layer_freq -> dense layers
    )


class _FakeDiffKV:
    """Pins the DiffKV selection so no device capability probe runs."""

    name = "TRITON_ATTN_DIFFKV"

    @classmethod
    def get_class(cls):
        from vllm.v1.attention.backends.triton_attn_diffkv import (
            TritonAttentionDiffKVBackend,
        )

        return TritonAttentionDiffKVBackend


@pytest.fixture()
def tp1_env(default_vllm_config):
    """Single-rank tensor-parallel group for the parallel-linear weights.
    Depends on default_vllm_config: initialize_model_parallel reads the
    current config, so it must already be set."""
    destroy_model_parallel()
    destroy_distributed_environment()
    with tempfile.TemporaryDirectory() as tmpdir:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{os.path.join(tmpdir, 'dist')}",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.fixture()
def fake_vllm_config(monkeypatch):
    def build(cache_dtype: str) -> SimpleNamespace:
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=_hf_config(),
                dtype=torch.bfloat16,
                is_mm_prefix_lm=False,
            ),
            quant_config=None,
            cache_config=SimpleNamespace(
                cache_dtype=cache_dtype,
                kv_cache_dtype_skip_layers=[],
                # Mirror the real checkpoint: MiMo ships a generic hf
                # `sliding_window` (128) alongside its per-layer
                # `sliding_window_size`, and config resolution copies it into
                # cache_config.sliding_window. The full-attention layers must
                # NOT inherit it.
                sliding_window=128,
                block_size=16,
                skip_page_size_padded=None,
                enable_prefix_caching=False,
            ),
            attention_config=SimpleNamespace(backend=_FakeDiffKV()),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        # Modules under test see the fake; CustomOps keep whatever real
        # config the default_vllm_config fixture installs.
        monkeypatch.setattr(
            mimo_v2_module, "get_current_vllm_config", lambda: cfg
        )
        monkeypatch.setattr(
            attention_module, "get_current_vllm_config", lambda: cfg
        )
        return cfg

    return build


@pytest.mark.parametrize("cache_dtype,expected", [
    ("fp8", torch.uint8),
    ("auto", torch.bfloat16),
])
@pytest.mark.parametrize("prefix", ["model.layers.0", "model.layers.1"])
def test_decoder_layer_threads_cache_dtype(
    tp1_env, default_vllm_config, fake_vllm_config,
    cache_dtype: str, expected: torch.dtype, prefix: str
):
    """Both the full-attention and SWA branches resolve the spec dtype from
    cache_config.cache_dtype, not from the model dtype."""
    cfg = fake_vllm_config(cache_dtype)
    layer = MiMoV2FlashDecoderLayer(vllm_config=cfg, prefix=prefix)
    assert layer.self_attn.attn.kv_cache_torch_dtype == expected


def test_full_attention_layer_is_not_windowed(tp1_env, default_vllm_config,
                                               fake_vllm_config):
    """Regression: threading cache_config must not turn the full-attention
    layers into sliding-window layers via the model-level -1 sentinel.
    A SlidingWindowSpec(-1) both mis-prices the pool (capacity explodes as
    ~1/max_in_flight) and window-masks the global layers (garbage output)."""
    cfg = fake_vllm_config("fp8")
    full = MiMoV2FlashDecoderLayer(vllm_config=cfg, prefix="model.layers.0")
    swa = MiMoV2FlashDecoderLayer(vllm_config=cfg, prefix="model.layers.1")
    assert full.self_attn.attn.sliding_window is None
    assert swa.self_attn.attn.sliding_window == 4
    full_spec = full.self_attn.attn.get_kv_cache_spec(cfg)
    swa_spec = swa.self_attn.attn.get_kv_cache_spec(cfg)
    assert isinstance(full_spec, FullAttentionSpec)
    assert isinstance(swa_spec, SlidingWindowSpec)
    assert swa_spec.sliding_window == 4
    assert full_spec.dtype == torch.uint8 and swa_spec.dtype == torch.uint8


@pytest.mark.parametrize("cache_dtype,expected", [
    ("fp8", torch.uint8),
    ("auto", torch.bfloat16),
])
def test_mtp_layer_threads_cache_dtype(
    tp1_env, default_vllm_config, fake_vllm_config,
    cache_dtype: str, expected: torch.dtype
):
    """The MTP predictor layer shares the same cache dtype as the target."""
    cfg = fake_vllm_config(cache_dtype)
    layer = MiMoV2MTPLayer(
        config=cfg.model_config.hf_text_config,
        prefix="model.mtp.layers.0",
        quant_config=None,
        cache_config=cfg.cache_config,
    )
    assert layer.self_attn.attn.kv_cache_torch_dtype == expected


def test_mtp_layer_cache_config_optional(
    tp1_env, default_vllm_config, fake_vllm_config
):
    """Omitting cache_config keeps the old behavior (auto)."""
    cfg = fake_vllm_config("fp8")
    layer = MiMoV2MTPLayer(
        config=cfg.model_config.hf_text_config,
        prefix="model.mtp.layers.0",
        quant_config=None,
    )
    assert layer.self_attn.attn.kv_cache_torch_dtype == torch.bfloat16


@pytest.mark.parametrize("env,expected_online", [("1", True), ("0", False)])
def test_mimo_oproj_fp8_env(
    tp1_env, default_vllm_config, fake_vllm_config, monkeypatch, env, expected_online
):
    """VLLM_MIMO_OPROJ_FP8=1 swaps the decoder o_proj to online per-tensor
    FP8 (Marlin W8A16 on sm86); off keeps the checkpoint quant config."""
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )

    monkeypatch.setenv("VLLM_MIMO_OPROJ_FP8", env)
    cfg = fake_vllm_config("fp8")
    layer = MiMoV2FlashDecoderLayer(vllm_config=cfg, prefix="model.layers.0")
    is_online = isinstance(
        layer.self_attn.o_proj.quant_method, Fp8PerTensorOnlineLinearMethod
    )
    assert is_online == expected_online
