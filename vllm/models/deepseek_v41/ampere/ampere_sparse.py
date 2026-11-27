# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 Triton sparse MLA attention for SM8x (A6000/3090, 4090).

`nvidia/specialized_decode/ampere` serves A100 (SM80) only: its CuTe DSL
kernels are sized for 164 KB of shared memory per SM. The other SM8x parts
(and SM80 with `VLLM_DSV41_SM80_TRITON=1`) reuse the ROCm Triton sparse-MLA
implementation wholesale, exactly like the V4 Ampere layer: its kernels, ragged
metadata builders and bf16 o_proj reference path are plain Triton/torch (the
aiter and gfx-only paths self-disable off ROCm), and
`vllm.v1.attention.ops.fp8_sm80` supplies e4m3 encode/decode below SM89 where
Triton refuses native fp8 converts. The KV caches keep the V4 records
(584 B, `fp8_ds_mla`) and the FP8 index keys by default. `--kv-cache-dtype
nvfp4_ds_mla` stores DeepSeek's V4.1 records instead, byte for byte those of
SM100 and of `nvidia/specialized_decode/ampere`: the 528 B MXFP8
sliding-window record and the 288 B NVFP4 compressed record.
"""

import functools
from typing import Any, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.lora.layers.custom_layer import DeepseekV4GroupedWoAWithLoRA
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v41.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterMLASparseBackend,
    compute_global_topk_ragged_indices_and_indptr,
)
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4SparseMLABackend
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum


class DeepseekV41AmpereMLASparseBackend(DeepseekV41ROCMAiterMLASparseBackend):
    # No plain per-tensor fp8 rows: those are aiter sparse MLA only.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        *DeepseekV4SparseMLABackend.supported_kv_cache_dtypes,
        "nvfp4_ds_mla",
    ]

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereIndexer(DeepseekV4Indexer):
    """The dense FP8 indexer, with the head count its Triton MQA-logits
    kernels are warmed up for."""

    def _attn_kwargs(self) -> dict[str, Any]:
        return {"num_heads": self.n_head}


class DeepseekV41AmpereMLAAttention(DeepseekV41ROCMAiterMLAAttention):
    """SM8x DeepSeek V4.1 attention: ROCm Triton path on CUDA Ampere/Ada."""

    backend_cls = DeepseekV41AmpereMLASparseBackend
    # The replay-start clamp of the Triton gather is validated on ROCm only;
    # the sliding-window cache takes part in prefix caching instead.
    supports_swa_bounded_replay = False
    # Its packed 584/528/288 B records and 132 B index keys are rows
    # `copy_pipeline_cache_rows` moves.
    supports_pipeline_relay = True

    def _indexer_cls(
        self, k_cache: DeepseekV4IndexerCache | None
    ) -> type[DeepseekV4Indexer]:
        return DeepseekV41AmpereIndexer

    # The ROCm overrides sequence the projections for HIP streams; SM8x runs
    # the shared multi-stream pipeline.
    forward = DeepseekV4Attention.forward
    _run_parallel_input_projections = (
        DeepseekV4Attention._run_parallel_input_projections
    )

    def _index_source(self) -> DeepseekV41ROCMAiterMLAAttention | None:
        """The layer's index source, or None when an earlier pipeline stage
        holds it and relays its top-k."""
        assert self._index_source_prefix is not None
        source = self._static_forward_context.get(self._index_source_prefix)
        if isinstance(source, DeepseekV41ROCMAiterMLAAttention):
            return source
        return None

    @functools.cached_property
    def _index_k_cache_prefix(self) -> str | None:
        if self._index_source() is None:
            return None
        return DeepseekV41ROCMAiterMLAAttention._index_k_cache_prefix.func(self)

    def _decode_topk_ragged(
        self, swa_metadata, attn_metadata, num_decodes: int, num_decode_tokens: int
    ):
        if self._index_source() is not None:
            return super()._decode_topk_ragged(
                swa_metadata, attn_metadata, num_decodes, num_decode_tokens
            )
        # No local source to own the per-step memo: pack the relayed top-k.
        return compute_global_topk_ragged_indices_and_indptr(
            self.topk_indices_buffer[:num_decode_tokens],
            swa_metadata.token_to_req_indices,
            attn_metadata.block_table[:num_decodes],
            attn_metadata.block_size // self.compress_ratio,
            swa_metadata.is_valid_token[:num_decode_tokens],
        )

    def _o_proj(
        self, attn_out: torch.Tensor | QuantizedActivation, positions: torch.Tensor
    ) -> torch.Tensor:
        """The BF16 `wo_a` einsum of the ROCm layer, then `wo_b`."""
        assert isinstance(attn_out, torch.Tensor)
        o = attn_out[:, : self.n_local_heads, :]
        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
            inverse_rope=False,
        )
        if isinstance(self.wo_a, DeepseekV4GroupedWoAWithLoRA):
            # The einsum reads the base weight, so the adapter is added here.
            self.wo_a.add_grouped_lora(
                z, o.reshape(o.shape[0], self.n_local_groups, -1)
            )
        return self._wo_b_after_wo_a(z.flatten(1))

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs["vllm_config"]
        if vllm_config.attention_config.indexer_sparse_logits:
            raise ValueError(
                "DeepSeek V4.1 on SM8x scores the indexer densely; "
                "attention_config.indexer_sparse_logits is not supported."
            )
        super().__init__(*args, **kwargs)
        # The ROCm layer routes wo_a through the ordinary MXFP8 linear kernel
        # (is_bmm=False) because its bf16 einsum dequantizes the raw fp8 weight
        # itself. On CUDA that ordinary list is Marlin, whose repack would turn
        # the weight into packed garbage for the einsum. Keep the grouped-BMM
        # kernel list instead: on SM8x it resolves to the emulation kernel,
        # which dequantizes wo_a to a plain bf16 [g*r, d] weight at load time;
        # _get_cached_wo_a_bf16 then only views it (no second dequant).
        self.wo_a.is_bmm = True
        if self.indexer is not None:
            # ShardQ's token-sharded indexer is the ROCm layer's; SM8x keeps
            # the head-sharded indexer and its own TP decode shard.
            k_cache = self.indexer.k_cache
            k_cache.use_shardq = k_cache.gather_shardq_topk = False
