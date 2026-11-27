# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA multi-stream overlap for MoE shared experts.

At decode widths the shared experts must run on the aux stream (in parallel with
the routed experts) and produce the same output as the serial path; above the
token threshold they stay serial on the main stream.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA shared-expert stream path"
)

HIDDEN = 64


class _StreamRecordingMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(
            HIDDEN, HIDDEN, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.streams: list[torch.cuda.Stream] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.streams.append(torch.cuda.current_stream())
        return self.proj(x)


def _shared_experts(layer: torch.nn.Module) -> SharedExperts:
    parallel = SimpleNamespace(
        enable_eplb=False,
        all2all_backend="allgather_reducescatter",
        use_fi_nvl_two_sided_kernels=False,
        dp_size=1,
        tp_size=1,
    )
    return SharedExperts(
        layer,
        moe_config=SimpleNamespace(moe_parallel_config=parallel),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: False,
    )


@pytest.mark.skipif(
    envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM, reason="shared experts stream disabled"
)
def test_decode_width_runs_on_aux_stream_and_matches_serial() -> None:
    torch.manual_seed(0)
    layer = _StreamRecordingMLP()
    shared = _shared_experts(layer)
    x = torch.randn(8, HIDDEN, device="cuda", dtype=torch.bfloat16)
    expected = layer.proj(x)

    assert shared.maybe_sync_shared_experts_stream(x)
    assert shared(x, SharedExpertsOrder.NO_OVERLAP) is None
    shared(x, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)

    torch.testing.assert_close(shared.output, expected)
    assert layer.streams == [shared._stream]
    assert shared._stream != torch.cuda.current_stream()


def test_above_threshold_stays_serial_on_main_stream() -> None:
    torch.manual_seed(0)
    layer = _StreamRecordingMLP()
    shared = _shared_experts(layer)
    x = torch.randn(
        envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD + 1,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = layer.proj(x)

    assert not shared.maybe_sync_shared_experts_stream(x)
    assert not shared.maybe_forward_async(x)
    shared(x, SharedExpertsOrder.NO_OVERLAP)
    assert shared(x, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED) is None

    torch.testing.assert_close(shared.output, expected)
    assert layer.streams == [torch.cuda.current_stream()]
