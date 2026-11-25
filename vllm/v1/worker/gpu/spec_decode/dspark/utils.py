# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, replace
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)


def _resolve_dspark_attention_backend(
    draft_model_config: ModelConfig,
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
) -> AttentionBackendEnum | None:
    if draft_backend is not None:
        return draft_backend
    # DeepSeek-V4(.1) draft layers share the target's KV-cache layout. Other
    # DSpark architectures may use a different attention kind.
    if draft_model_config.hf_config.model_type in ("deepseek_v4", "deepseek_v41"):
        if target_backend is not None:
            logger.info_once(
                "Using the target model's %s attention backend for the "
                "DeepSeek-V4 DSpark drafter.",
                target_backend.name,
            )
        return target_backend
    return None


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import PPMissingLayer, get_draft_quant_config
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        get_target_lm_head,
        maybe_share_target_embed,
    )

    draft_attention_backend = _resolve_dspark_attention_backend(
        draft_model_config,
        speculative_config.attention_backend,
        vllm_config.attention_config.backend,
    )

    draft_vllm_config = replace(
        vllm_config,
        parallel_config=replace(
            vllm_config.parallel_config,
            pipeline_parallel_size=1,
            tensor_parallel_size=(
                speculative_config.draft_parallel_config.tensor_parallel_size
            ),
        ),
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    target_vocab_size = vllm_config.model_config.get_vocab_size()
    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None)
        or draft_model_config.get_vocab_size()
    )
    target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
        target_inner, "embedding", None
    )
    # V4.1's CUDA drafter has no checkpoint weights for these modules. Avoid
    # allocating throwaway copies, but attach the shared modules only after
    # loading so draft post-processing cannot touch the target's weights.
    defer_shared_vocab = (
        current_platform.is_cuda()
        and draft_model_config.architectures == ["DSparkV41DraftModel"]
        and vllm_config.parallel_config.pipeline_parallel_size == 1
        and draft_vllm_config.parallel_config.tensor_parallel_size
        == vllm_config.parallel_config.tensor_parallel_size
        and draft_model_config.get_vocab_size() == target_vocab_size
        and draft_output_vocab_size == target_vocab_size
        and target_embed is not None
        and not isinstance(target_embed, PPMissingLayer)
        and target_lm_head is not None
        and not isinstance(target_lm_head, PPMissingLayer)
    )
    vocab_context = nullcontext()
    if defer_shared_vocab:
        from vllm.models.deepseek_v4_1.nvidia.dspark import skip_shared_vocab_alloc

        vocab_context = skip_shared_vocab_alloc()
        logger.info("DSpark V4.1: deferring shared embedding and LM head allocation")

    with set_model_tag("dspark_head"), vocab_context:
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    draft_inner = draft_model.model
    if draft_model_config.get_vocab_size() <= target_vocab_size:
        maybe_share_target_embed(draft_model, draft_inner, target_inner)

    draft_lm_head = getattr(draft_model, "lm_head", None)
    if (
        target_lm_head is not None
        and draft_output_vocab_size == target_vocab_size
        and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    if defer_shared_vocab and (
        draft_inner.embed_tokens is None or draft_model.lm_head is not target_lm_head
    ):
        raise RuntimeError("DSpark V4.1 failed to attach shared vocabulary modules")
    return draft_model
