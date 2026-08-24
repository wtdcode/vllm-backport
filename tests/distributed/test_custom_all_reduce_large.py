# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom all-reduce above the default 8 MB cap.

`VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE` raises the payload ceiling so that a
tensor-parallel PREFILL all-reduce (num_tokens x hidden x 2 B, e.g. 16 MB for a
2048-token chunk at hidden 4096) takes the custom two-shot kernel instead of
falling back to NCCL. That turns on a path which, at these sizes, has never run
before, so this covers both halves:

  * the WIRING -- the knob has to reach `CustomAllreduce`'s constructor through
    `CudaCommunicator`, which is the shipped entry point; a test that builds
    `CustomAllreduce` directly would pass while the env var did nothing.
  * the NUMERICS -- per-element random data, scored against an fp32 reduction.
    A constant fill is not sufficient: two-shot is reduce-scatter plus
    all-gather, so a chunk mapped to the wrong rank still reads back correct
    when every element holds the same value.

The dispatch assertion is the anti-vacuity guard: the run WITHOUT the knob must
be observed taking the NCCL path, otherwise both arms could be measuring the
same thing and agreeing for free.

Run directly under torchrun, or via the pytest wrapper below:
    torchrun --nproc_per_node=8 tests/distributed/test_custom_all_reduce_large.py
"""

import os
import subprocess
import sys

import pytest
import torch

from vllm.platforms import current_platform

PAYLOAD_MB = 16
HIDDEN = 4096
DTYPE = torch.bfloat16
MiB = 1024 * 1024


def _worker() -> int:
    import torch.distributed as dist

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    expect_custom = os.environ.get("EXPECT_CUSTOM") == "1"
    torch.accelerator.set_device_index(rank)
    # Held in a variable: inlining lets the context manager be collected and
    # every rank then dies on "Current vLLM config is not set".
    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank, backend="nccl"
    )
    ensure_model_parallel_initialized(world, 1)

    rows = PAYLOAD_MB * MiB // DTYPE.itemsize // HIDDEN
    torch.manual_seed(1234)
    payload = (torch.randn(rows, HIDDEN, device="cuda") * (rank + 1)).to(DTYPE)

    if os.environ.get("RUN_INT8") == "1":
        from vllm.model_executor.kernels.mhc.ar_int8 import int8_all_reduce

        ref = payload.float()
        dist.all_reduce(ref)
        out_q, out_s = int8_all_reduce(payload)
        got = out_q.float() * out_s.float().repeat_interleave(32, dim=-1)
        torch.accelerator.synchronize()
        rel = (got - ref).abs().max().item() / ref.abs().max().item()
        ok = rel < 5e-2
        if rank == 0:
            print(f"int8_ar rel_err={rel:.3e} -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    ca = get_tp_group().device_communicator.ca_comm
    took_custom = ca is not None and ca.should_custom_ar(payload)
    if took_custom != expect_custom:
        if rank == 0:
            cap = None if ca is None else ca.max_size
            print(
                f"FAIL: custom-AR dispatch {took_custom}, expected "
                f"{expect_custom} (max_size={cap})"
            )
        return 1

    ref = payload.float()
    dist.all_reduce(ref)
    got = tensor_model_parallel_all_reduce(payload)
    torch.accelerator.synchronize()
    rel = (got.float() - ref).abs().max().item() / ref.abs().max().item()

    # bf16 rounding over 8 addends; the ring accumulates more of it than
    # two-shot does, so the bound has to admit the worse of the two.
    ok = rel < 2e-2
    if rank == 0:
        print(
            f"custom_ar={took_custom} rel_err={rel:.3e} -> {'PASS' if ok else 'FAIL'}"
        )
    return 0 if ok else 1


@pytest.mark.skipif(
    current_platform.device_count() < 8,
    reason="prefill-sized all-reduce needs the full 8-rank TP group",
)
@pytest.mark.parametrize("knob_mb", [None, 32])
def test_large_payload_all_reduce(knob_mb: int | None):
    """Without the knob a 16 MB all-reduce must fall back to NCCL; with it,
    the custom kernel must take the payload AND stay numerically correct."""
    env = dict(os.environ)
    env["EXPECT_CUSTOM"] = "0" if knob_mb is None else "1"
    if knob_mb is None:
        env.pop("VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE", None)
    else:
        env["VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE"] = str(knob_mb)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=8",
            os.path.abspath(__file__),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


@pytest.mark.skipif(
    current_platform.device_count() < 8,
    reason="compressed prefill all-reduce needs the full 8-rank TP group",
)
def test_int8_payload_all_reduce():
    env = dict(os.environ)
    env["RUN_INT8"] = "1"
    env["VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE"] = "32"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=8",
            os.path.abspath(__file__),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


if __name__ == "__main__":
    sys.exit(_worker())
