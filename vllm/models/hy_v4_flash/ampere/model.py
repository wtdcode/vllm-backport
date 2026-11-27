# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash backbone for SM8x: the NVIDIA model with SM8x attention."""

from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.hy_v4.weight_utils import HYV4_MLP_MAPPER
from vllm.models.hy_v4_flash.nvidia import model as nvidia

from .attention import HYV4FlashAmpereAttention

validate_platform = nvidia.validate_platform


class HYV4FlashDecoderLayer(nvidia.HYV4FlashDecoderLayer):
    attention_cls = HYV4FlashAmpereAttention


class HYV4FlashModel(nvidia.HYV4FlashModel):
    decoder_layer_cls = HYV4FlashDecoderLayer


class HYV4FlashForCausalLM(nvidia.HYV4FlashForCausalLM):
    model_cls = HYV4FlashModel
    # The SM8x IndexPool keeps its value and gate projections separate.
    hf_to_vllm_mapper = HYV4_MLP_MAPPER | WeightsMapper(
        orig_to_new_substr={".compressed_indexer.": ".indexer."},
        orig_to_new_stacked={
            ".self_attn.q_a_proj.": (".self_attn.fused_qkv_a_proj.", 0),
            ".self_attn.kv_a_proj_with_mqa.": (".self_attn.fused_qkv_a_proj.", 1),
        },
    )
