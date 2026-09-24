# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A8 activation-dtype selection for Marlin experts (VLLM_MARLIN_INPUT_DTYPE).

The mxfp4/Marlin MoE path on Ampere runs W4A16 by default;
VLLM_MARLIN_INPUT_DTYPE=int8 switches it to W4A8-INT8 (per-token activation
quant, int tensor cores, sm75+). The fp8 choice is gated to sm89/sm12x
(no fp8 tensor cores below that; it would be slower than W4A16).
"""

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, None),
        ("int8", torch.int8),
    ],
)
def test_marlin_input_dtype_selection(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    expected: torch.dtype | None,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MARLIN_INPUT_DTYPE", env_value)
    assert get_marlin_input_dtype() == expected


def test_marlin_input_dtype_fp8_arch_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """fp8 activations are sm89/sm12x-only; below that the env must fail loudly
    instead of silently falling back to W4A16."""
    monkeypatch.setattr(envs, "VLLM_MARLIN_INPUT_DTYPE", "fp8")
    if current_platform.is_device_capability(
        89
    ) or current_platform.is_device_capability_family(120):
        assert get_marlin_input_dtype() == torch.float8_e4m3fn
    else:
        with pytest.raises(ValueError, match="SM89 or SM12x"):
            get_marlin_input_dtype()


def test_marlin_experts_reads_env_at_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """MarlinExperts (the mxfp4 MoE experts class on Ampere) must consume the
    env so the knob reaches fused_marlin_moe."""
    # The read happens in __init__ via get_marlin_input_dtype(); asserting
    # the wiring without constructing full expert weights:
    import inspect

    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts

    init_src = inspect.getsource(MarlinExperts.__init__)
    assert "get_marlin_input_dtype()" in init_src


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernel selection"
)
def test_online_fp8_picks_marlin_without_native_fp8() -> None:
    """Online (dynamically quantized) fp8 linear layers must not select the
    per-tensor torch fallback on parts without native fp8 (sm86): it accepts
    the config but never quantizes dynamic activations, producing a
    bf16 x fp8 matmul crash at runtime. Weight-only Marlin (W8A16) is the
    supported route there, matching the offline fp8 path."""
    from vllm.model_executor.kernels.linear.scaled_mm import (
        MarlinFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )

    with torch.device("cuda"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "vllm.model_executor.layers.quantization.online.fp8."
            "cutlass_fp8_supported",
            lambda: False,
        )
        method = Fp8PerTensorOnlineLinearMethod()
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=1024,
            output_partition_sizes=[4096],
            input_size=1024,
            output_size=4096,
            params_dtype=torch.bfloat16,
        )
        assert not isinstance(
            method.fp8_linear, PerTensorTorchFP8ScaledMMLinearKernel
        )
        assert isinstance(method.fp8_linear, MarlinFP8ScaledMMLinearKernel)
