# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's vocab-sharded Markov head: gating, layout, and the 8-rank path.

The reduction itself is covered on CPU by
``tests/v1/sample/test_local_argmax_reduce.py``; what is DSpark-specific is
*when* the sharded layout is chosen and whether the shipped draft loop agrees
with the replicated one. The latter needs a real TP group, so it runs under
torchrun via ``dspark_vocab_shard_torchrun.py`` and is skipped on boxes with
fewer than 8 GPUs.
"""

import dataclasses
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.qwen3_dspark import (
    DSparkMarkovHead,
    dspark_vocab_shard_enabled,
)
from vllm.platforms import current_platform

WORKER = os.path.join(os.path.dirname(__file__), "dspark_vocab_shard_torchrun.py")


@pytest.mark.cpu_test
def test_gate_is_off_without_a_config():
    """Constructed outside an engine (tests, tooling) the head stays replicated."""
    assert dspark_vocab_shard_enabled() is False


@pytest.mark.cpu_test
@pytest.mark.parametrize("flag", [False, True])
def test_gate_follows_use_local_argmax_reduction(
    monkeypatch: pytest.MonkeyPatch, flag: bool
):
    from vllm.model_executor.models import qwen3_dspark

    monkeypatch.setattr(
        qwen3_dspark,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            speculative_config=SimpleNamespace(use_local_argmax_reduction=flag)
        ),
    )
    assert dspark_vocab_shard_enabled() is flag


@pytest.mark.cpu_test
@pytest.mark.parametrize("shard_vocab", [False, True])
def test_markov_w2_layout_follows_the_gate(
    monkeypatch: pytest.MonkeyPatch, shard_vocab: bool
):
    """Sharding markov_w2 is what makes the draft GEMM smaller; the weight has
    to land split, not just be selected differently."""
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    vocab, rank_dim, tp = 128, 8, 4
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_world_size", lambda: tp
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(
        vocab, vocab, rank_dim, prefix="markov_head", shard_vocab=shard_vocab
    )
    expected_rows = vocab // tp if shard_vocab else vocab
    assert head.markov_w2.tp_size == (tp if shard_vocab else 1)
    assert head.markov_w2.weight.shape == (expected_rows, rank_dim)
    # markov_w1 is indexed by a token id, so it is replicated eitherway.
    assert head.markov_w1.weight.shape == (vocab, rank_dim)

    full = torch.randn(vocab, rank_dim)
    head.markov_w2.weight_loader(head.markov_w2.weight, full)
    start = head.markov_w2.shard_indices.org_vocab_start_index
    torch.testing.assert_close(
        head.markov_w2.weight.data, full[start : start + expected_rows]
    )


@pytest.mark.cpu_test
def test_probabilistic_drafting_is_rejected():
    """The reduction never materializes full logit rows, which is exactly what
    probabilistic verification consumes -- so the combination must not boot."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.use_local_argmax_reduction = True
    spec.speculative_config = SimpleNamespace(draft_sample_method="probabilistic")
    with pytest.raises(ValueError, match="probabilistic"):
        spec._validate_local_argmax_reduction()


@pytest.mark.cpu_test
def test_model_without_the_hooks_is_rejected():
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.use_local_argmax_reduction = True
    spec.speculative_config = SimpleNamespace(draft_sample_method="greedy")
    spec._draft_topk = None
    spec.model = SimpleNamespace()
    with pytest.raises(ValueError, match="compute_draft_logits_shard"):
        spec._validate_local_argmax_reduction()


def _markov_head_for_fusion(monkeypatch, tp=4, soft_cap=None, scale=1.0):
    """A sharded Markov head plus the LogitsProcessor the speculator pairs it
    with, built without an engine."""
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    vocab, rank_dim = 128, 8
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_world_size", lambda: tp
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )
    head = DSparkMarkovHead(
        vocab, vocab, rank_dim, prefix="markov_head", shard_vocab=True
    )
    lp = LogitsProcessor(vocab, scale=scale, soft_cap=soft_cap)
    return head, lp, vocab, rank_dim, tp


@pytest.mark.cpu_test
def test_fusion_operands_describe_the_shard(monkeypatch: pytest.MonkeyPatch):
    """The fused Markov kernel is handed shard geometry, not module internals;
    if these stop matching the head, the kernel reads the wrong rows."""
    head, lp, vocab, rank_dim, tp = _markov_head_for_fusion(monkeypatch)
    op = head.fusion_operands(lp)
    assert op is not None
    assert op.w2.shape == (vocab // tp, rank_dim)
    assert op.w1.shape == (vocab, rank_dim)
    assert op.tp_size == tp and op.tp_rank == 3
    assert op.vocab_start == head.markov_w2.shard_indices.org_vocab_start_index
    # No padding in this layout, so every column of the shard is a real token.
    assert op.num_valid == vocab // tp


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "kwargs", [{"soft_cap": 30.0}, {"scale": 2.0}], ids=["soft_cap", "scale"]
)
def test_fusion_declines_what_it_cannot_reproduce(
    monkeypatch: pytest.MonkeyPatch, kwargs
):
    """The kernel reimplements select_top_tokens end to end. A soft cap or a
    logit scale changes what the argmax compares, so the head must decline and
    leave the eager chain in place rather than silently drop the transform."""
    head, lp, *_ = _markov_head_for_fusion(monkeypatch, **kwargs)
    assert head.fusion_operands(lp) is None


@pytest.mark.cpu_test
def test_fusion_declines_a_mismatched_head_dtype(monkeypatch: pytest.MonkeyPatch):
    head, lp, *_ = _markov_head_for_fusion(monkeypatch)
    lp.head_dtype = torch.float64
    assert head.fusion_operands(lp) is None


@pytest.mark.cpu_test
def test_sampler_build_declines_a_model_without_the_hook():
    """A model that never grew markov_fusion_operands must fall back, not
    crash -- and must say so, because a silent fallback is an unmeasured arm."""
    from vllm.v1.worker.gpu.spec_decode.dspark.markov_argmax import (
        build_fused_markov_sampler,
    )

    got = build_fused_markov_sampler(
        SimpleNamespace(), max_num_reqs=8, n_spec=5, device=torch.device("cpu")
    )
    assert got is None


@pytest.mark.cpu_test
def test_speculative_config_carries_the_flag():
    """Guards the knob the A/B start uses: --speculative-config
    '{..., "use_local_argmax_reduction": true}'."""
    fields = {f.name: f for f in dataclasses.fields(SpeculativeConfig)}
    assert fields["use_local_argmax_reduction"].default is False


@pytest.mark.skipif(
    current_platform.device_count() < 8,
    reason="the sharded draft path is only meaningful on a real 8-rank TP group",
)
def test_sharded_draft_matches_replicated_on_8_ranks():
    """Drives DSparkSpeculator._sample_sequential on a real TP group, flag off
    vs on, and requires token-for-token equality."""
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=8", WORKER],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert "RESULT PASS" in result.stdout, result.stdout[-4000:] + result.stderr[-4000:]
