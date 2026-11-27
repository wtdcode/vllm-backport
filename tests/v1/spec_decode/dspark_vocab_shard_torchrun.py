# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""8-rank check of DSpark's vocab-sharded greedy drafting.

Not named ``test_*`` on purpose: it needs a real TP process group, so it is
launched by ``test_dspark_vocab_shard.py`` as

    torchrun --nproc_per_node=8 tests/v1/spec_decode/dspark_vocab_shard_torchrun.py

What it covers that a single-process test cannot: the shipped entry point.
``DSparkSpeculator._sample_sequential`` is driven directly -- the real
function, on real ``ParallelLMHead``\\ s whose weights went through the real
vocab-parallel ``weight_loader`` -- and every arm must agree token for token.
The previous iteration of this feature had 31 green unit tests over the
reduction while the shipped path could not run at all, which is the hole this
closes.

Four arms, all required to produce the same draft:

  ``replicated``   reduction off: full-vocab Markov bias, replicated markov_w2
  ``allgather``    reduction on, per-step all_gather of the (value, id) pairs
  ``allreduce``    reduction on, sum-all_reduce over a zeroed [B, 2*tp] buffer
  ``fused``        reduction on, the fused Triton Markov step (which uses the
                   all_reduce exchange internally)
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed.parallel_state import (  # noqa: E402
    ensure_model_parallel_initialized,
    graph_capture,
    init_distributed_environment,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: E402
    ParallelLMHead,
)
from vllm.model_executor.models.qwen3_dspark import (  # noqa: E402
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)
from vllm.v1.worker.gpu.spec_decode.dspark.markov_argmax import (  # noqa: E402
    build_fused_markov_sampler,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (  # noqa: E402
    DSparkSpeculator,
)

# Checkpoint constants (DeepSeek-V4-Flash-0731 config.json): full-vocab draft,
# markov_rank 256, dspark_block_size 5. hidden_size is cut to 512 -- it only
# sets the base GEMM's K, and nothing here depends on its value.
VOCAB = 129280
MARKOV_RANK = 256
HIDDEN = 512
N_SPEC = 5
BATCH = 4


class _StubDSpark(nn.Module):
    """A DSpark draft model reduced to its head, borrowing the shipped methods.

    The class attributes are the real function objects from
    ``Qwen3DSparkForCausalLM``, so this exercises the shipped code rather than
    a copy of it; only the backbone (which produces ``head_hidden``) is stubbed
    out, and the speculator is handed that hidden state directly.
    """

    compute_draft_logits = Qwen3DSparkForCausalLM.compute_draft_logits
    compute_draft_logits_shard = Qwen3DSparkForCausalLM.compute_draft_logits_shard
    select_draft_token_shard = Qwen3DSparkForCausalLM.select_draft_token_shard
    markov_embed = Qwen3DSparkForCausalLM.markov_embed
    markov_bias = Qwen3DSparkForCausalLM.markov_bias
    map_draft_to_target = Qwen3DSparkForCausalLM.map_draft_to_target
    markov_fusion_operands = Qwen3DSparkForCausalLM.markov_fusion_operands

    draft_id_to_target_id = None

    def __init__(self, lm_head, dtype: torch.dtype, shard_vocab: bool) -> None:
        super().__init__()
        self.lm_head = lm_head
        self.logits_processor = LogitsProcessor(VOCAB)
        self.model = nn.Module()
        self.model.markov_head = DSparkMarkovHead(
            VOCAB,
            VOCAB,
            MARKOV_RANK,
            prefix="markov_head",
            shard_vocab=shard_vocab,
        )
        self.model.markov_head.to(dtype=dtype, device="cuda")


def _make_speculator(model, use_shard: bool, num_reqs: int, fused: bool = False):
    """A DSparkSpeculator carrying only what _sample_sequential reads."""
    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.model = model
    spec.num_speculative_steps = N_SPEC
    spec.use_local_argmax_reduction = use_shard
    spec.draft_logits = None
    spec._fused_markov = None
    spec._draft_topk = None
    spec.use_confidence_head = False
    spec._fuse_markov_bias = False
    spec.acceptance_estimator = None
    if fused:
        assert use_shard, "the fused step only exists on the sharded path"
        spec._fused_markov = build_fused_markov_sampler(
            model, num_reqs, N_SPEC, torch.device("cuda")
        )
        assert spec._fused_markov is not None, (
            "the fused Markov step declined this head; the arm would silently "
            "measure the eager chain"
        )
    spec.sample_indices = torch.arange(
        num_reqs * N_SPEC, dtype=torch.int64, device="cuda"
    )
    spec.sample_idx_mapping = torch.zeros(
        num_reqs * N_SPEC, dtype=torch.int32, device="cuda"
    )
    spec.sample_pos = torch.arange(num_reqs * N_SPEC, dtype=torch.int64, device="cuda")
    spec._anchor_idx = torch.arange(num_reqs, dtype=torch.int64, device="cuda") * N_SPEC
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.randint(
            0, VOCAB, (num_reqs * N_SPEC,), dtype=torch.int64, device="cuda"
        )
    )
    spec.draft_tokens = torch.zeros(num_reqs, N_SPEC, dtype=torch.int64, device="cuda")
    return spec


# name, use_shard, fused, VLLM_LOCAL_ARGMAX_ALLREDUCE ("" = leave alone)
_ARMS = (
    ("replicated", False, False, ""),
    ("allgather", True, False, "0"),
    ("allreduce", True, False, "1"),
    ("fused", True, True, "1"),
)


def _poison(name):
    """Trap #8 guard: a green arm that ran the other path is worse than a red
    one. Every entry point an arm must NOT reach is replaced by this, so a
    silent fallback -- in either direction -- fails loudly instead of agreeing
    trivially."""

    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not run on this arm")

    return fail


# Entry points each arm is forbidden to touch. The fused arm's list is the
# long one on purpose: the Triton step does the embedding lookup, the bias
# projection, the selection and the draft-to-target map itself, so reaching
# any of these means the fusion silently did not happen.
_FORBIDDEN = {
    "replicated": ("select_draft_token_shard", "compute_draft_logits_shard"),
    "allgather": ("markov_bias", "compute_draft_logits"),
    "allreduce": ("markov_bias", "compute_draft_logits"),
    "fused": (
        "markov_bias",
        "compute_draft_logits",
        "select_draft_token_shard",
        "markov_embed",
        "map_draft_to_target",
    ),
}


def _build_arm(lm_head, dtype, use_shard, full_w2, w1_ref):
    model = _StubDSpark(lm_head, dtype, shard_vocab=use_shard)
    w2 = model.model.markov_head.markov_w2
    w2.weight_loader(w2.weight, full_w2)
    # markov_w1 must match across arms so `prev` feeds the same embedding.
    model.model.markov_head.markov_w1.weight.data.copy_(w1_ref)
    return model


def _draft_for_arm(
    name,
    use_shard,
    fused,
    allreduce,
    lm_head,
    dtype,
    full_w2,
    w1_ref,
    head_hidden,
    anchors,
):
    model = _build_arm(lm_head, dtype, use_shard, full_w2, w1_ref)
    prev_env = os.environ.get("VLLM_LOCAL_ARGMAX_ALLREDUCE")
    if allreduce:
        os.environ["VLLM_LOCAL_ARGMAX_ALLREDUCE"] = allreduce
    try:
        # The sampler is built before poisoning: construction reads weights,
        # not the forbidden entry points.
        spec = _make_speculator(model, use_shard, BATCH, fused=fused)
        for attr in _FORBIDDEN[name]:
            setattr(model, attr, _poison(f"{name}.{attr}"))
        spec.input_buffers.input_ids.copy_(anchors)
        spec._sample_sequential(BATCH, head_hidden)
        return spec.draft_tokens.clone()
    finally:
        if allreduce:
            if prev_env is None:
                os.environ.pop("VLLM_LOCAL_ARGMAX_ALLREDUCE", None)
            else:
                os.environ["VLLM_LOCAL_ARGMAX_ALLREDUCE"] = prev_env


def _run(dtype: torch.dtype, rank: int) -> bool:
    torch.manual_seed(0)  # identical draw on every rank
    lm_head = ParallelLMHead(VOCAB, HIDDEN, bias=False, params_dtype=dtype).cuda()
    # Every arm shares the base head (it is vocab-parallel either way) and
    # loads the same full markov_w2 through its own layout's weight_loader.
    full_lm = torch.randn(VOCAB, HIDDEN, dtype=torch.float32).to(dtype)
    full_w2 = torch.randn(VOCAB, MARKOV_RANK, dtype=torch.float32).to(dtype)
    lm_head.weight_loader(lm_head.weight, full_lm)
    w1_ref = (torch.randn(VOCAB, MARKOV_RANK, dtype=torch.float32) * 0.02).to(dtype)

    expected_width = VOCAB // dist.get_world_size()
    probe = _build_arm(lm_head, dtype, True, full_w2, w1_ref)
    got_width = probe.model.markov_head.markov_w2.weight.shape[0]
    assert got_width == expected_width, f"{got_width} != {expected_width}"
    del probe

    head_hidden = torch.randn(BATCH * N_SPEC, HIDDEN, device="cuda").to(dtype)
    # Same anchor tokens on every arm and every rank.
    anchors = (
        torch.arange(BATCH * N_SPEC, dtype=torch.int64, device="cuda") * 977 % VOCAB
    )

    drafts = {}
    for name, use_shard, fused, allreduce in _ARMS:
        drafts[name] = _draft_for_arm(
            name,
            use_shard,
            fused,
            allreduce,
            lm_head,
            dtype,
            full_w2,
            w1_ref,
            head_hidden,
            anchors,
        )

    ref = drafts["replicated"]
    ok = True
    for name in drafts:
        if not torch.equal(ref, drafts[name]):
            ok = False
            if rank == 0:
                diff = (ref != drafts[name]).nonzero()
                print(f"  [{name}] differs from replicated at {diff.tolist()}")
                print(f"    replicated {ref.tolist()}")
                print(f"    {name:<10} {drafts[name].tolist()}")
    # The reduction has to be reaching across ranks: if every selected id came
    # from one shard, agreement would say nothing about the exchange.
    owners = (drafts["fused"] // expected_width).unique()
    if owners.numel() < 2:
        ok = False
        if rank == 0:
            print(f"  degenerate: all draft ids from shard(s) {owners.tolist()}")
    if rank == 0:
        tag = str(dtype).replace("torch.", "")
        print(
            f"[{tag}] all {len(_ARMS)} arms agree token for token: {ok} "
            f"(ids span shards {owners.tolist()})"
        )
    return ok


def _guards_fire(rank: int) -> bool:
    """Prove the poison is live: an arm that *does* reach a forbidden entry
    point must raise. Without this the whole matrix could pass because the
    poison was never installed on anything that runs."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    lm_head = ParallelLMHead(VOCAB, HIDDEN, bias=False, params_dtype=dtype).cuda()
    lm_head.weight_loader(
        lm_head.weight, torch.randn(VOCAB, HIDDEN, dtype=torch.float32).to(dtype)
    )
    full_w2 = torch.randn(VOCAB, MARKOV_RANK, dtype=torch.float32).to(dtype)
    w1_ref = (torch.randn(VOCAB, MARKOV_RANK, dtype=torch.float32) * 0.02).to(dtype)
    head_hidden = torch.randn(BATCH * N_SPEC, HIDDEN, device="cuda").to(dtype)
    anchors = (
        torch.arange(BATCH * N_SPEC, dtype=torch.int64, device="cuda") * 977 % VOCAB
    )

    # The fused arm's forbidden list, but running the UNFUSED sharded chain:
    # select_draft_token_shard is reached on the first step and must trip.
    fired = False
    try:
        _draft_for_arm(
            "fused",
            True,
            False,
            "1",
            lm_head,
            dtype,
            full_w2,
            w1_ref,
            head_hidden,
            anchors,
        )
    except AssertionError as exc:
        fired = "must not run on this arm" in str(exc)
    if rank == 0:
        print(f"[guard] poison fires when the fused arm falls back: {fired}")
    return fired


def _capture(rank: int, fused: bool, allreduce: str = "") -> bool:
    """The draft step is captured under a FULL cudagraph, so the sharded
    selection -- which contains a collective -- has to be capturable."""
    torch.manual_seed(0)
    lm_head = ParallelLMHead(
        VOCAB, HIDDEN, bias=False, params_dtype=torch.bfloat16
    ).cuda()
    # Load it: ParallelLMHead allocates with torch.empty, and uninitialized
    # bytes decode as NaN often enough that an unloaded head turns this into a
    # NaN-robustness test by accident (which is how the sentinel bug in
    # reduce_global_argmax was found -- it now has its own test).
    lm_head.weight_loader(
        lm_head.weight, torch.randn(VOCAB, HIDDEN, dtype=torch.float32).bfloat16()
    )
    model = _StubDSpark(lm_head, torch.bfloat16, shard_vocab=True)
    w2 = model.model.markov_head.markov_w2
    w2.weight_loader(
        w2.weight, torch.randn(VOCAB, MARKOV_RANK, dtype=torch.float32).bfloat16()
    )
    spec = _make_speculator(model, True, BATCH, fused=fused)
    head_hidden = torch.randn(
        BATCH * N_SPEC, HIDDEN, device="cuda", dtype=torch.bfloat16
    )
    tag = "fused" if fused else "eager"
    if allreduce:
        tag += "+ar"
        os.environ["VLLM_LOCAL_ARGMAX_ALLREDUCE"] = allreduce
    try:
        # The reference runs OUTSIDE graph_capture, and that placement is
        # load-bearing, not stylistic. Inside the context but not actually
        # capturing, `custom_all_reduce` deliberately returns
        # `torch.empty_like(input)` -- uninitialized memory
        # (custom_all_reduce.py:387-391); that warm-up exists only to fix the
        # allocation pattern for the capture that follows. A reference taken
        # from there is garbage, which on this path means an out-of-range token
        # id and a device-side assert in the next embedding lookup.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            spec._sample_sequential(BATCH, head_hidden)
        torch.cuda.current_stream().wait_stream(s)
        torch.accelerator.synchronize()
        eager = spec.draft_tokens.clone()

        # vLLM's own `graph_capture()`, not a bare `torch.cuda.graph`: the pair
        # exchange takes the custom all-reduce path, whose graph buffers are
        # only registered when that context exits. This is how the real draft
        # step is captured, so a bare capture would be replaying a different
        # collective from the one production runs.
        with graph_capture(device=torch.device(f"cuda:{rank}")) as ctx:
            with torch.cuda.stream(ctx.stream):
                spec._sample_sequential(BATCH, head_hidden)
            torch.accelerator.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=ctx.stream):
                spec._sample_sequential(BATCH, head_hidden)
        spec.draft_tokens.zero_()
        g.replay()
        torch.accelerator.synchronize()
        ok = torch.equal(eager, spec.draft_tokens)
        if rank == 0:
            print(f"[capture/{tag}] captured replay == eager: {ok}")
        return ok
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            print(f"[capture/{tag}] FAILED: {type(exc).__name__}: {exc}")
        return False
    finally:
        if allreduce:
            os.environ.pop("VLLM_LOCAL_ARGMAX_ALLREDUCE", None)


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(rank)
    # Held in a variable: inlining lets the context manager be collected and
    # every rank then dies on "Current vLLM config is not set".
    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank, backend="nccl"
    )
    ensure_model_parallel_initialized(world, 1)

    results = [
        _run(torch.float32, rank),
        _run(torch.bfloat16, rank),
        _guards_fire(rank),
        _capture(rank, fused=False),
        # The all_reduce exchange under capture: its warm-up inside
        # graph_capture() sees uninitialized bytes by design, so this arm is
        # what pins the clamp that keeps them out of an embedding lookup.
        _capture(rank, fused=False, allreduce="1"),
        _capture(rank, fused=True),
    ]
    dist.barrier()
    if rank == 0:
        print("RESULT", "PASS" if all(results) else "FAIL")
    dist.destroy_process_group()
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
