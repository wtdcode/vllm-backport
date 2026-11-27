# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink-capable sparse MLA for HY V4 on SM8x (Ampere/Ada: sm80/sm86/sm89).

FlashMLA needs SM90+, so SM8x runs the ROCm BF16 Triton sparse MLA kernels,
which already take `attn_sink` and are plain Triton. The AITER dispatch of
the ROCm backend is not inherited. Top-k rows are packed and counted before
each read, as on the ROCm DFly draft path, so any indexer row layout works.
An fp8_ds_mla cache holds upstream's 656-byte records, which the DeepSeek V4
single-pass Triton kernel reads with a table decode instead of native fp8.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.models.hy_v4.amd.aiter_mla_sparse import pack_live_indices_
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionLayer
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseImpl,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_sparse_attn_fp8_ds_mla,
    rocm_sparse_attn_prefill,
)


@dataclass(kw_only=True)
class HYV4AmpereMLASparseMetadata(SparseMLACommonMetadata):
    # Ragged physical KV rows per query token, filled by the impl.
    paged_kv_indices: torch.Tensor | None = None
    paged_kv_indptr: torch.Tensor | None = None


class HYV4AmpereMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[HYV4AmpereMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    metadata_cls = HYV4AmpereMLASparseMetadata

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        assert layer.topk_buffer_width is not None
        self.topk_tokens = layer.topk_buffer_width
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.paged_kv_indices = torch.zeros(
            max_tokens * self.topk_tokens, dtype=torch.int32, device=device
        )
        self.paged_kv_indptr = torch.zeros(
            max_tokens + 1, dtype=torch.int32, device=device
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        metadata.paged_kv_indices = self.paged_kv_indices
        metadata.paged_kv_indptr = self.paged_kv_indptr
        return metadata


class HYV4AmpereMLASparseImpl(ROCMAiterMLASparseImpl):
    """Triton sparse MLA with learnable sinks, without AITER."""

    def __init__(self, *args, kv_cache_dtype: str, **kwargs) -> None:
        if kv_cache_dtype not in HYV4AmpereMLASparseBackend.supported_kv_cache_dtypes:
            raise ValueError(
                "HYV4 sparse MLA on SM8x requires a BF16 or fp8_ds_mla KV cache, "
                f"got kv_cache_dtype={kv_cache_dtype}"
            )
        # The ROCm impl ties fp8_ds_mla to AITER's reader (gfx950) and ShardQ.
        base_dtype = "auto" if kv_cache_dtype == "fp8_ds_mla" else kv_cache_dtype
        super().__init__(*args, kv_cache_dtype=base_dtype, **kwargs)
        self.kv_cache_dtype = kv_cache_dtype
        if self.dcp_world_size > 1 or self.pcp_world_size > 1:
            raise NotImplementedError(
                "HYV4 sparse MLA on SM8x does not support context parallelism"
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: HYV4AmpereMLASparseMetadata,  # type: ignore[override]
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)
        num_tokens, num_heads, head_dim = q.shape
        kv, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        assert self.topk_indices_buffer is not None
        assert attn_metadata.paged_kv_indptr is not None
        assert attn_metadata.paged_kv_indices is not None
        start = attn_metadata.topk_start
        topk = self.topk_indices_buffer[start : start + num_tokens]
        # The ragged kernels read each row's first `indptr` delta entries.
        indptr = attn_metadata.paged_kv_indptr[: num_tokens + 1]
        torch.cumsum(pack_live_indices_(topk), dim=0, out=indptr[1:])
        triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk,
            indptr,
            attn_metadata.paged_kv_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk.shape[1],
        )
        output = torch.empty(
            (num_tokens, num_heads, self.kv_lora_rank), dtype=q.dtype, device=q.device
        )
        # Single pass for decode rows too: the split-K decodes reduce over a
        # power-of-two row width or read the DeepSeek V4 packed record.
        if self.kv_cache_dtype == "fp8_ds_mla":
            rocm_sparse_attn_fp8_ds_mla(
                q,
                kv,
                attn_metadata.paged_kv_indices,
                indptr,
                self.scale,
                self._attn_sink(q),
                output,
            )
            return output, None
        rocm_sparse_attn_prefill(
            q=q,
            kv=kv.unsqueeze(1),
            indices=None,
            topk_length=None,
            scale=self.scale,
            head_dim=head_dim,
            nope_head_dim=self.kv_lora_rank,
            rope_head_dim=head_dim - self.kv_lora_rank,
            attn_sink=self._attn_sink(q),
            output=output,
            ragged_indices=attn_metadata.paged_kv_indices,
            ragged_indptr=indptr,
        )
        return output, None


class HYV4AmpereMLASparseBackend(ROCMAiterMLASparseBackend):
    # Keeps the inherited name: shared code keys the row-addressed (non-flat)
    # KV cache view and JIT warmups off it.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_metadata_cls() -> type[HYV4AmpereMLASparseMetadata]:  # type: ignore[override]
        return HYV4AmpereMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type[HYV4AmpereMLASparseMetadataBuilder]:  # type: ignore[override]
        return HYV4AmpereMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[HYV4AmpereMLASparseImpl]:
        return HYV4AmpereMLASparseImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_dcp(cls) -> bool:
        return False
