# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash coarse-only gated IndexPool for SM8x.

The NVIDIA IndexPool needs DeepGEMM paged MQA and native FP8 Triton casts
(SM89+). SM8x runs the device-independent ROCm IndexPool and scores it with
the LUT-decoding Triton MQA kernel, which reads the same block-flat pool pages.
"""

from vllm.models.hy_v4_flash.amd.compressed_indexer import (
    FlashCompressedIndexer,
)
from vllm.models.hy_v4_flash.amd.indexpool_ops import INDEX_DIM
from vllm.models.hy_v4_flash.common.indexpool_cache import POOL_PAGE_SIZE
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_paged_mqa_logits_triton,
    warmup_fp8_paged_mqa_logits_triton,
)


class FlashAmpereCompressedIndexer(FlashCompressedIndexer):
    def __init__(self, vllm_config, config, topk_indices_buffer, prefix):
        super().__init__(vllm_config, config, topk_indices_buffer, prefix)
        # The autotuner's synchronizing benchmark is illegal under capture.
        warmup_fp8_paged_mqa_logits_triton(
            max(16, next_power_of_2(self.num_heads)),
            INDEX_DIM,
            POOL_PAGE_SIZE,
            topk_indices_buffer.device,
        )

    def _paged_mqa_logits(self, query, cache, weights, lengths, table, width):
        # Columns at or past a row's length stay unwritten; group selection
        # masks them.
        return fp8_paged_mqa_logits_triton(
            query, cache, weights, lengths.view(-1), table, width, clean_logits=False
        )
