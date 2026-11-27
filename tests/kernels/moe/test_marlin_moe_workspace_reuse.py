# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A Marlin workspace must survive being reused across calls.

The Marlin expert path reuses one persistent workspace (`get_marlin_workspace`)
instead of running a `torch.zeros` per call. That is only correct if the
kernel leaves the workspace in a reusable state -- it is a per-SM lock/counter
array that every Marlin linear layer and expert call now shares.

The failure this guards against is not a crash: a stale lock array corrupts
intermittently, on some later call, which is exactly the kind of thing that
survives a smoke test. So the assertion is that call N is identical to call 1
for several consecutive calls on one workspace.
"""

import pytest
import torch

from tests.kernels.moe.test_moe import MarlinMoEWeightData
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

NUM_CALLS = 6


@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("m", [1, 64])
@torch.inference_mode()
def test_shared_workspace_matches_per_call_workspace(m: int) -> None:
    set_random_seed(1)
    dtype = torch.bfloat16
    n, k, e, topk = 128, 512, 8, 2
    b_type = scalar_types.uint4b8

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    w1_data = MarlinMoEWeightData.make(w=w1, quant_type=b_type, group_size=128)
    w2_data = MarlinMoEWeightData.make(w=w2, quant_type=b_type, group_size=128)

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)

    def run(workspace):
        return fused_marlin_moe(
            a,
            w1_data.qweight,
            w2_data.qweight,
            None,
            None,
            w1_data.scales,
            w2_data.scales,
            topk_weights,
            topk_ids,
            global_num_experts=e,
            expert_map=None,
            global_scale1=w1_data.global_scale,
            global_scale2=w2_data.global_scale,
            w1_zeros=w1_data.zeros,
            w2_zeros=w2_data.zeros,
            quant_type_id=b_type.id,
            workspace=workspace,
        ).clone()

    # Baseline: today's behaviour, a freshly zeroed workspace every call.
    reference = run(None)

    # One workspace, reused. Every call must reproduce the baseline bit for
    # bit -- same inputs, same kernel, so anything else means the workspace
    # carried state between calls.
    shared = marlin_make_workspace_new(a.device, 4)
    for call in range(NUM_CALLS):
        got = run(shared)
        assert torch.equal(got, reference), (
            f"call {call + 1} of {NUM_CALLS} diverged on a reused workspace: "
            f"max|diff|={(got.float() - reference.float()).abs().max().item():.3e}"
        )

    # And the workspace itself must come back zeroed, which is the property
    # that makes reuse safe in the first place.
    assert torch.equal(shared, torch.zeros_like(shared)), (
        f"workspace not left zeroed: {shared[shared != 0][:8].tolist()}"
    )
