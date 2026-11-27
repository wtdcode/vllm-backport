# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash attention for SM8x (Ampere/Ada: sm80/sm86/sm89).

Keeps the NVIDIA projections and window indexer, and replaces the SM90+ parts:
FlashMLA with the BF16 Triton sparse MLA backends and the DeepGEMM IndexPool
with the Triton-scored one.
"""

from vllm.models.hy_v4_flash.nvidia.attention import HYV4FlashAttention

from .compressed_indexer import FlashAmpereCompressedIndexer
from .swa import FlashAmpereSparseBackend, FlashAmpereSWABackend


class HYV4FlashAmpereAttention(HYV4FlashAttention):
    capability_majors = (8,)
    compressed_indexer_cls = FlashAmpereCompressedIndexer
    sparse_backend_cls = FlashAmpereSparseBackend
    swa_backend_cls = FlashAmpereSWABackend
