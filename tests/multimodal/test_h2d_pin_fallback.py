# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned-H2D fallback behavior for multimodal input transfers.

cudaHostAlloc can fail during encoder profiling under host-memory pressure
or memlock limits (observed on 8-rank TP workers at high
gpu-memory-utilization). The transfer must degrade to a pageable copy
instead of crashing, and VLLM_MM_DISABLE_PINNED_H2D=1 must skip the
pinned path entirely.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.multimodal.inputs import _nested_tensors_h2d


def test_h2d_pin_failure_falls_back_to_pageable(monkeypatch: pytest.MonkeyPatch):
    """A raising pinned allocation degrades to a pageable copy."""

    def _raise_pin(self, *args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError("cudaHostAlloc failed: invalid argument")
        return torch.Tensor.new_empty(self, *args, **{**kwargs, "pin_memory": False})

    monkeypatch.setattr(torch.Tensor, "new_empty", _raise_pin)
    src = {"pixel_values": torch.randn(2, 3, 4, 4)}
    out = _nested_tensors_h2d(src, torch.device("cpu"), pin_memory=True)
    torch.testing.assert_close(out["pixel_values"], src["pixel_values"])


def test_h2d_env_disables_pinned_path(monkeypatch: pytest.MonkeyPatch):
    """VLLM_MM_DISABLE_PINNED_H2D=1 skips the pinned allocation attempt."""

    calls = []
    orig_new_empty = torch.Tensor.new_empty

    def _spy_pin(self, *args, **kwargs):
        calls.append(kwargs.get("pin_memory"))
        return orig_new_empty(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "new_empty", _spy_pin)
    monkeypatch.setattr(envs, "VLLM_MM_DISABLE_PINNED_H2D", True)

    src = {"pixel_values": torch.randn(2, 3, 4, 4)}
    out = _nested_tensors_h2d(src, torch.device("cpu"), pin_memory=True)
    torch.testing.assert_close(out["pixel_values"], src["pixel_values"])
    assert not any(c for c in calls if c), "pinned allocation was attempted"
