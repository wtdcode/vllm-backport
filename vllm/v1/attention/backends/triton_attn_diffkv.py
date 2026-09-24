# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attention backend with different K/V head dimensions (DiffKV).

The KV cache layout is identical to ``FlashAttentionDiffKVBackend``: K and V
are packed along the last dim in the logical shape
``[num_blocks, num_kv_heads, block_size, head_size_qk + head_size_v]``.
"""

import os
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class TritonAttentionDiffKVMetadataBuilder(TritonAttentionMetadataBuilder):
    """Override the parent's softmax buffer last-dim to head_size_v.

    The parent allocates ``softmax_segm_output`` with last-dim sized to
    ``next_power_of_2(head_size)`` (== Q/K head size).  For DiffKV the
    accumulator and per-segment partial outputs are V-shaped, so we
    re-allocate with ``next_power_of_2(head_size_v)`` instead.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # diffkv split-KV knob (diffbot recipe): full-attention groups get
        # more split-KV segments. The spec-verify launch covers a whole
        # request per program (BLOCK_M = q_len x GQA group), so the segment
        # axis must supply the CTAs: 64 segments = 1.3-1.5x (q=4) / 1.8-2.6x
        # (q=8) per layer vs stock 16 on 46K-180K contexts. SWA groups keep
        # the stock count (their loops are window-bounded). Stock-off
        # default: unset keeps the stock 16 segments; set
        # Upstream default 16. 64 = split-KV segments for the global
        # (full-attention) layers: recipe-claimed 1.3-2.6x per layer at
        # 46K-180K ctx; ladder-validated on sm86 TP8.
        full_attn_segments = int(
            os.environ.get("VLLM_DIFFKV_FULL_ATTN_SEGMENTS", "64")
        )
        if (
            getattr(kv_cache_spec, "sliding_window", None) is None
            and full_attn_segments > self.num_par_softmax_segments
        ):
            self.num_par_softmax_segments = full_attn_segments
            self.softmax_segm_max = torch.empty(
                (self.seq_threshold_3D, self.num_heads_q, full_attn_segments),
                dtype=torch.float32,
                device=device,
            )
            self.softmax_segm_expsum = torch.empty(
                (self.seq_threshold_3D, self.num_heads_q, full_attn_segments),
                dtype=torch.float32,
                device=device,
            )

        head_size_v = TritonAttentionDiffKVBackend.head_size_v
        head_size_v_padded = next_power_of_2(head_size_v)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                head_size_v_padded,
            ),
            dtype=torch.float32,
            device=device,
        )


class TritonAttentionDiffKVBackend(TritonAttentionBackend):
    # V head dim — set per layer via ``set_head_size_v`` before instantiation.
    head_size_v: int = 128

    # Packed DiffKV supports unquantized cache and per-tensor E4M3 scales
    # (diffbot recipe port of local-inference-lab/vllm #830; sm86 uses the
    # kernel's LUT dequant path automatically).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        cls.head_size_v = head_size_v

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionDiffKVImpl"]:
        return TritonAttentionDiffKVImpl

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionDiffKVMetadataBuilder"]:
        return TritonAttentionDiffKVMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # DiffKV K head sizes (e.g. 192 for MiMo-V2.5) need to be allowed.
        return head_size >= 32

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        # DiffKV only implements decoder self-attention.  Unlike the parent
        # TritonAttentionBackend (which advertises all types), encoder
        # attention is not supported, so gate it here at backend selection.
        return attn_type == AttentionType.DECODER


class TritonAttentionDiffKVImpl(TritonAttentionImpl):
    """Triton attention impl for the DiffKV packed KV cache layout."""

    # fp8 K/V is dequantized via the 256-entry E4M3 LUT before the dot
    # products (no native fp8 math), so the parent's SM89 architectural
    # gate does not apply. See TritonAttentionImpl.__init__.
    fp8_kv_lut_dequant = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # DiffKV dequantizes K/V to the query dtype before its dot products.
        # Inheriting the parent's CUDA query-quantization flag would instead
        # quantize Q without supplying a query descale to this implementation.
        self.supports_quant_query_input = False
        if is_quantized_kv_cache(self.kv_cache_dtype) and self.kv_cache_dtype not in (
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend only supports per-tensor E4M3 "
                f"quantized KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )
        if self._is_per_token_head_quant:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support per-token-head "
                "quantization."
            )
        if self.chunk_lookback > -1:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support chunked "
                "attention with lookback."
            )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Cache is logical (B, H, N, C); the diffkv reshape kernel expects
        # (B, N, H, C).
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache.transpose(1, 2),
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        # The fused rope+cache path assumes the standard 2-tensor layout.
        return False

    # Latched at first qualifying forward (weights are loaded by then):
    # True once the layer's sink values are proven all-zero, False if not.
    _prefill_cuda_sinks_zero: bool | None = None

    def _use_prefill_cuda(
        self,
        num_actual_tokens: int,
        max_query_len: int,
        head_size_qk: int,
        head_size_v: int,
    ) -> bool:
        """Route global-layer prefill to the CUDA kernel when configured.

        Only full-attention layers with the MiMo TP8 per-rank shape (8 Q
        heads, K 192 / V 128, 1 KV head) qualify. Full attention is the
        parent impl's (-1, -1) window tuple (never None). The CUDA kernel
        has no sink support: with add_swa_attention_sink_bias the checkpoint
        hands a sink Parameter to every layer, so a layer qualifies only
        when its sink values are provably all-zero (add_full_attention_sink
        bias is false, i.e. global layers carry inert zero sinks) — checked
        once and latched at first use, after weights load.

        Prefill-shaped batches only (max_query_len > 16): the kernel gets
        its parallelism from the query dimension (a 1024-token chunk is 64
        CTAs), while spec-decode verify batches (max_query_len =
        num_spec_tokens + 1) would leave it CTA-starved — those stay on the
        tuned split-KV spec-3D Triton path regardless of batch width.
        """
        from vllm.v1.attention.ops.prefill_attn_cuda import prefill_cuda_enabled

        if not prefill_cuda_enabled():
            return False
        if self.sliding_window != (-1, -1):
            return False
        if self.sinks is not None:
            if self._prefill_cuda_sinks_zero is None:
                self._prefill_cuda_sinks_zero = bool((self.sinks == 0).all())
                if not self._prefill_cuda_sinks_zero:
                    logger.warning_once(
                        "prefill_attn_sm86: layer %s has non-zero sinks; "
                        "keeping Triton prefill for it",
                        getattr(self, "layer_name", "<unknown>"),
                    )
            if not self._prefill_cuda_sinks_zero:
                return False
        return (
            num_actual_tokens > 16
            and max_query_len > 16
            and self.num_heads == 8
            and self.num_kv_heads == 1
            and head_size_qk == 192
            and head_size_v == 128
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Shapes:
            query:    [num_tokens, num_heads, head_size_qk]
            key:      [num_tokens, num_kv_heads, head_size_qk]
            value:    [num_tokens, num_kv_heads, head_size_v]
            kv_cache: [num_blocks, num_kv_heads, block_size,
                       head_size_qk + head_size_v]
            output:   [num_tokens, num_heads, head_size_v]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for "
                "TritonAttentionDiffKVImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False, (
            "Cascade attention not supported for TritonAttentionDiffKVImpl"
        )

        num_actual_tokens = attn_metadata.num_actual_tokens
        head_size_qk = self.head_size
        head_size_v = TritonAttentionDiffKVBackend.head_size_v

        if self._use_prefill_cuda(
            num_actual_tokens,
            attn_metadata.max_query_len,
            head_size_qk,
            head_size_v,
        ):
            # Global (full-attention) layers' prefill: purpose-built CUDA
            # kernel (ldmatrix + mma.m16n8k16; fp8/bf16 KV converted in the
            # load path — sm86 has no cvt.rn e4m3 hardware, so fp8 converts
            # via exact bit-repack). Consumes the physical [pages, 1, page,
            # 320] layout, so hook BEFORE the Triton (B, N, H, D) transpose.
            from vllm.v1.attention.ops.prefill_attn_cuda import prefill_attn_cuda

            try:
                cache = (
                    kv_cache.view(self.fp8_dtype)
                    if is_quantized_kv_cache(self.kv_cache_dtype)
                    else kv_cache
                )
                prefill_attn_cuda(
                    q=query[:num_actual_tokens],
                    kv_cache=cache,
                    block_table=attn_metadata.block_table,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    seq_lens=attn_metadata.seq_lens,
                    softmax_scale=self.scale,
                    k_descale=(
                        float(layer._k_scale)
                        if layer._k_scale is not None
                        else 1.0
                    ),
                    v_descale=(
                        float(layer._v_scale)
                        if layer._v_scale is not None
                        else 1.0
                    ),
                    out=output[:num_actual_tokens],
                )
                return output
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "prefill_attn_sm86 failed; falling back to Triton",
                    exc_info=True,
                )
                logger.warning_once(
                    "prefill_attn_sm86 failed (%s: %s); falling back to "
                    "Triton (full traceback logged at debug level)",
                    type(e).__name__,
                    e,
                )

        # Triton DiffKV kernels consume (B, N, H, D) cache views.
        kv_cache = kv_cache.transpose(1, 2)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            kv_cache = kv_cache.view(self.fp8_dtype)
        key_cache = kv_cache[..., :head_size_qk]
        value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]

        unified_attention_diffkv(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            sinks=self.sinks,
            max_seqlen_q=attn_metadata.max_query_len,
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
        )
        return output
