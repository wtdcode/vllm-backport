# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import logging
import os

import torch
import torch.nn as nn

from vllm.config import CompilationMode, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.model_loader import get_model

logger = logging.getLogger(__name__)

# Candidate checkpoint keys for the token embedding, most specific first.
# GLM5-Next multimodal checkpoints prefix the text tower weights.
_EMBED_KEYS = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "model.embed.weight",
    "embed_tokens.weight",
)


def _has_real_weight(module) -> bool:
    """True for a materialised layer; False for PPMissingLayer / None.

    Under pipeline parallelism vLLM replaces the layers a rank does not own with
    PPMissingLayer, which has no ``weight``. Aliasing one of those into the draft
    silently produces a no-op layer rather than an error, so check explicitly.
    """
    return module is not None and getattr(module, "weight", None) is not None


def _load_embed_from_checkpoint(embed: nn.Module, model_path: str) -> None:
    """Fill the draft's own token embedding straight from the checkpoint.

    Only needed under PP: the drafter runs on the LAST pipeline rank, but the
    target's ``embed_tokens`` lives on the FIRST, so there is nothing local to
    alias. The MTP ``load_weights()`` only consumes spec-layer tensors (the
    top-level ``embed_tokens`` key is skipped by the spec-layer filter), so the
    draft's own embedding would otherwise stay uninitialised memory -- the draft
    then emits constant garbage tokens (acceptance ~= 1). Reading the one tensor
    off disk avoids adding a cross-rank collective. Mirrors the dspark+PP fix.
    """
    from safetensors import safe_open

    model_path = os.path.expanduser(model_path)
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    key = shard = None
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        for cand in _EMBED_KEYS:
            if cand in weight_map:
                key, shard = cand, os.path.join(model_path, weight_map[cand])
                break
    if key is None:
        raise RuntimeError(
            f"PP spec-decode: could not find a token-embedding tensor in "
            f"{index_path}; looked for {_EMBED_KEYS}. The draft needs its own "
            "copy because the target's embedding lives on PP rank 0."
        )

    with safe_open(shard, framework="pt") as f:
        w = f.get_tensor(key)
    with torch.no_grad():
        # VocabParallelEmbedding pads the vocab dimension, so copy into the
        # leading rows rather than assigning the whole tensor.
        embed.weight.data[: w.shape[0]].copy_(
            w.to(dtype=embed.weight.dtype, device=embed.weight.device)
        )
    logger.info(
        "PP spec-decode: loaded draft token embedding %s from key %r",
        tuple(w.shape),
        key,
    )


def _should_share(eagle: nn.Module, flag: str, draft, target) -> bool:
    """Share when the draft has no own copy, or its copy matches the target."""

    if not getattr(eagle, flag, False) or draft is None:
        return True
    if target is None:
        return False
    # torch.equal on GPU allocates a bool mask the size of the input.
    # Use the faster GPU path when there is plenty of headroom;
    # otherwise compare on CPU.
    w = draft.weight
    if w.is_cuda and torch.accelerator.get_memory_info(w.device)[0] < w.numel() * 2:
        return torch.equal(w.cpu(), target.weight.cpu())
    return torch.equal(w, target.weight)


def get_target_lm_head(target_model: nn.Module, target_language_model: nn.Module):
    """The target's lm_head — from get_language_model() for
    *ForConditionalGeneration targets, else the top-level module."""
    return getattr(target_language_model, "lm_head", None) or getattr(
        target_model, "lm_head", None
    )


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    if speculative_config.kv_cache_dtype is not None:
        vllm_config = replace(
            vllm_config,
            cache_config=replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            ),
        )
    # enforce_eager on the speculative config must make the DRAFT eager too.
    # Without this the draft inherits the target's VLLM_COMPILE mode; dynamo
    # then fails on data-dependent asserts in some draft models (observed:
    # Qwen3.5 MTP head), and concurrent draft+target AOT compiles race in
    # TritonBundler's cache.
    #
    # Mutate the mode in place instead of dataclasses.replace(): the draft's
    # attention layers register into THIS compilation_config's
    # static_forward_context at construction, and the runtime forward context
    # looks them up in the original object — a replaced copy strands the
    # draft's layers in a dict nobody reads (KeyError
    # 'mtp.layers.0.self_attn.attn' during profiling).
    compilation_config = vllm_config.compilation_config
    original_mode = compilation_config.mode
    if speculative_config.enforce_eager:
        compilation_config.mode = CompilationMode.NONE
    try:
        with set_model_tag("eagle_head"):
            eagle_model = get_model(
                vllm_config=vllm_config, model_config=draft_model_config
            )
    finally:
        compilation_config.mode = original_mode

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = eagle_model.model

    # Embedding sharing: only possible when the target's embed_tokens is
    # materialised on this rank (under PP it lives on rank 0, and the drafter
    # runs on the LAST rank, where the target is a PPMissingLayer).
    target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
        target_inner, "embedding", None
    )
    # If the target's embedding is LoRA-wrapped, share the underlying base
    # layer. The draft is not part of the LoRA adapter; sharing the wrapper
    # would make the draft run the LoRA embedding kernel with the target's
    # punica metadata (sized for the target's token count), causing an
    # out-of-bounds GPU access during multi-step draft decode.
    if isinstance(target_embed, BaseLayerWithLoRA):
        target_embed = target_embed.base_layer
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if _has_real_weight(target_embed) and _should_share(
        eagle_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed
    elif get_pp_group().world_size != 1 and draft_embed is not None:
        # PP: the target's embedding is a PPMissingLayer here. Keep the
        # draft's own module and populate it from the checkpoint -- the MTP
        # load_weights() filters out every non-spec-layer key, so its own
        # embed_tokens would otherwise stay uninitialised and the draft would
        # emit constant garbage tokens (acceptance ~= 1, corrupt output).
        _load_embed_from_checkpoint(draft_embed, draft_model_config.model)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(eagle_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        eagle_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del eagle_model.lm_head
        eagle_model.lm_head = target_lm_head

        # MTP layers route logits through layer.shared_head.head, not
        # eagle_model.lm_head, so the per-layer copies need fixing up too.
        layers = getattr(draft_inner, "layers", None)
        if layers is not None:
            items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
            for layer in items:
                sh = getattr(layer, "shared_head", None)
                if sh is not None and hasattr(sh, "head"):
                    del sh.head
                    sh.head = target_lm_head

    # MTP shares topk_indices_buffer with the target model. We update
    # every module in the draft that holds a buffer reference so that
    # the per-layer indexer and sparse-attention backends all point to
    # the target's buffer.
    if hasattr(target_inner, "topk_indices_buffer"):
        target_buffer = target_inner.topk_indices_buffer
        if target_buffer is not None:
            for _, module in draft_inner.named_modules():
                if hasattr(module, "topk_indices_buffer"):
                    module.topk_indices_buffer = target_buffer

    return eagle_model
