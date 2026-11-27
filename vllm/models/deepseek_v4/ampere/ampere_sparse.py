# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 sparse MLA attention for SM8x (Ampere/Ada: sm80/sm86/sm89).

Reuses the ROCm Triton sparse-MLA implementation: its kernels, ragged
metadata builders, and bf16 o_proj reference path are plain Triton/torch (the
aiter-only paths self-disable off ROCm), and `vllm.v1.attention.ops.fp8_sm80`
supplies e4m3 encode/decode below SM89 where Triton refuses native fp8
converts. The ROCm stream fork/join is not inherited: CUDA keeps the shared
multi-stream input projections of `DeepseekV4Attention`.
"""

from typing import ClassVar

from vllm.config.cache import CacheDType
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseBackend,
)
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4SparseMLABackend
from vllm.platforms.interface import DeviceCapability


class DeepseekV4AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    # Plain per-tensor fp8_e4m3 rows are an aiter-only (gfx950) layout.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(
        DeepseekV4SparseMLABackend.supported_kv_cache_dtypes
    )

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV4AmpereMLAAttention(DeepseekV4ROCMAiterMLAAttention):
    """SM8x DeepSeek V4 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV4AmpereMLASparseBackend

    def __init__(
        self,
        vllm_config,
        prefix: str,
        topk_indices_buffer=None,
        aux_stream_list=None,
    ) -> None:
        super().__init__(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        # The ROCm layer drops these for its own in-capture fork/join.
        self.aux_stream_list = aux_stream_list
        if self.indexer is not None and aux_stream_list is not None:
            self.indexer.aux_stream = aux_stream_list[2]

    # The ROCm overrides sequence the projections for HIP streams; SM8x runs
    # the shared pipeline (merged input GEMMs, token shard, aux streams).
    forward = DeepseekV4Attention.forward
    _prepare_and_attn = DeepseekV4Attention._prepare_and_attn
