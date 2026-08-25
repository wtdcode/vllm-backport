# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating and entry point for the int8 compressed prefill all-reduce (task #35).

The prefill all-reduces are hoisted out of RowParallelLinear and the MoE runner
into the decoder layer so they can be transported as int8 + one bf16 scale per
32 elements, halving the wire. mhc_post then consumes the quantized form
directly; a bf16 phase 2 would hand back half the saving.

Everything here is decided ONCE at construction, because suppressing the
downstream all-reduces is itself a construction-time decision -- `reduce_results`
is read when the layer is built, long before any per-forward token count exists.
That asymmetry is the whole reason this module exists: the set of all-reduce
sites we suppress must exactly equal the set the decoder layer re-adds, and
those two sets would otherwise be computed in different files, by different
mechanisms, under different names.

A missing all-reduce and a double all-reduce are both silent -- wrong numerics,
no crash -- and both are indistinguishable from a quality regression in the
serving gate. So the rule this module enforces is: **verify the resolved
outcome, never predict it.**
"""

import functools

import torch

import vllm.envs as envs

# Prefill-only. Must exceed max_cudagraph_capture_size, which is asserted rather
# than assumed -- see assert_hoist_preconditions. 8K prefill runs at 8192 tokens.
MIN_INT8_TOKENS = 2048

QUANT_BLOCK = 32


def ar_hoisted(vllm_config) -> bool:
    """The single predicate. Both halves of the hoist derive from this."""
    if not envs.VLLM_MHC_AR_INT8:
        return False
    # parallel_state is imported inside each function in this file, not at
    # module level: this module is imported by kernel suites that run
    # single-process, before (or without) distributed init.
    from vllm.distributed.parallel_state import (
        get_tp_group,
        model_parallel_is_initialized,
    )

    # Must stay inert single-process: the kernel suites run without a TP group
    # and would otherwise trip get_tp_group() at construction.
    if not model_parallel_is_initialized():
        return False
    return get_tp_group().world_size > 1


def assert_hoist_preconditions(
    vllm_config, moe_config=None, routed_output_transform=None
) -> None:
    """Turn every remaining unknown into a startup failure.

    Called at construction with whatever is in scope; each check is skipped only
    when its subject is absent, never when it is merely inconvenient.
    """
    # (1) Capture cap. The int8 op does not replicate the bf16 path's warm-up
    # branch (CustomAllreduce.custom_all_reduce returns uninitialised memory
    # under _IS_CAPTURING when the stream is not actually capturing), and its IPC
    # buffer registration under graph capture is unvalidated. Keeping the
    # threshold above the largest captured size makes both unreachable rather
    # than merely unlikely. A comment cannot survive someone raising the cap or
    # lowering the threshold; this can.
    cap = getattr(vllm_config.compilation_config, "max_cudagraph_capture_size", 0) or 0
    assert cap < MIN_INT8_TOKENS, (
        f"VLLM_MHC_AR_INT8 requires the int8 token threshold to exceed the "
        f"cudagraph capture cap so prefill is always eager and no replayed "
        f"graph can contain the int8 op; got threshold={MIN_INT8_TOKENS} "
        f"vs max_cudagraph_capture_size={cap}"
    )

    # (2) Registered-buffer capacity. The custom all-reduce's IPC buffer is
    # sized by VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE and defaults to 8 MiB, far
    # below a 64 MiB prefill payload -- and the quantize writes the int8 codes
    # AND their scales straight into that buffer. Without this the failure lands
    # deep in the op at first prefill; with it, at startup with the number
    # needed. Largest payload is max_num_batched_tokens x hidden x (1 + 1/16).
    from vllm.distributed.parallel_state import get_tp_group

    ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
    assert ca is not None and not ca.disabled, (
        "VLLM_MHC_AR_INT8 requires custom all-reduce; remove "
        "--disable-custom-all-reduce"
    )
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    hidden = vllm_config.model_config.get_hidden_size()
    need = max_tokens * hidden + (max_tokens * hidden // QUANT_BLOCK) * 2
    assert ca.max_size >= need, (
        f"VLLM_MHC_AR_INT8 needs the custom all-reduce buffer to hold the "
        f"int8 payload plus its block scales: need {need / 2**20:.1f} MiB "
        f"for {max_tokens} tokens x {hidden}, but the communicator was "
        f"built with {ca.max_size / 2**20:.1f} MiB. Raise "
        f"VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE to at least "
        f"{int(need / 2**20) + 1}."
    )

    # (3) The pre-transform all-reduce (moe_runner.py:452-460) is mathematically
    # unhoistable: the transform contains RMSNorm, and normalising a partial sum
    # then summing is wrong, not merely differently rounded. Inactive for DSv4
    # (set only by nemotron_h and kimi_k3), but a future config that sets it
    # would make the hoist silently wrong.
    assert routed_output_transform is None, (
        "VLLM_MHC_AR_INT8 cannot hoist the all-reduce when "
        "routed_output_transform is set: that reduction happens before an "
        "RMSNorm and is not hoistable"
    )

    # (4) The conjunction. skip_final_all_reduce is resolved from FOUR terms
    # (fused_moe/layer.py:232-237) -- reduce_results, use_all2all_kernels,
    # is_sequence_parallel, zero_expert_type -- so reduce_results=False is
    # NECESSARY BUT NOT SUFFICIENT. Enabling expert parallelism, for instance,
    # silently turns suppression back off while the decoder layer keeps
    # reducing, giving a double all-reduce with no crash. Read the resolved
    # value back rather than re-deriving it: that cannot be wrong however the
    # other three terms move.
    if moe_config is not None:
        assert moe_config.skip_final_all_reduce, (
            "VLLM_MHC_AR_INT8 hoists the MoE all-reduce, but the constructed "
            "MoE config resolved skip_final_all_reduce=False -- the decoder "
            "layer would reduce a tensor the MoE already reduced. Check the "
            "other terms of the conjunction at fused_moe/layer.py:232-237 "
            "(expert parallelism and sequence parallelism both defeat it)."
        )
    # The shared-expert interaction (moe_runner.py:438) needs no assert here:
    # moe_runner.py:477-480 already enforces `skip_final_all_reduce implies not
    # _fused_output_is_reduced` at runtime on every call, so a quant-method
    # change that flips output_is_reduced() trips that invariant rather than
    # corrupting. Cited rather than duplicated so the two cannot drift.


def use_int8_for(x: torch.Tensor) -> bool:
    """Discriminator: shapes only, never values.

    Deriving this from tensor VALUES would read uninitialised memory during
    cudagraph warmup and can diverge across ranks -- the hang/corruption case.
    Token count is replicated metadata and therefore rank-uniform.
    """
    return x.shape[0] >= MIN_INT8_TOKENS


@functools.cache
def _ca_comm():
    """The custom all-reduce communicator, resolved once.

    `assert_hoist_preconditions` already validated it at startup; the
    communicator is created before the first forward and never replaced.
    """
    from vllm.distributed.parallel_state import get_tp_group

    ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
    assert ca is not None and not ca.disabled, (
        "VLLM_MHC_AR_INT8 requires the custom all-reduce communicator"
    )
    return ca


def int8_all_reduce(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compressed all-reduce of `x`, returning (int8 codes, bf16 block scales).

    `x` must be passed through UNTOUCHED from the producing layer. Any
    reshape/view/.contiguous()/slice in between can flip is_weak_contiguous or
    change nbytes, which silently re-dispatches to a different communicator with
    a different reduction order and no crash.
    """
    ca = _ca_comm()
    n = x.numel()
    out_q = torch.empty(n, device=x.device, dtype=torch.int8)
    out_s = torch.empty(n // QUANT_BLOCK, device=x.device, dtype=torch.bfloat16)
    torch.ops._C_custom_ar.all_reduce_int8(
        ca._ptr, x, out_q, out_s, ca.buffer_ptrs[ca.rank], ca.max_size
    )
    return out_q.view(x.shape), out_s.view(x.shape[0], -1)
