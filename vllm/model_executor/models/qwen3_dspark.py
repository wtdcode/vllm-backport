# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 DSpark draft model for semi-autoregressive drafting.

DSpark drafts a whole block in one parallel pass (DFlash-style: context-KV
precompute + a non-causal query-block forward) and then injects intra-block
dependency with a lightweight sequential Markov head.

The parallel backbone is a standard Qwen3 decoder stack reused from the
DFlash Qwen3 draft (see qwen3_dflash.py). DSpark adds:
  * `markov_head`: low-rank V x r / r x V transition bias added to the base
    logits, sampled left-to-right by the speculator (the sequential stage).
  * `confidence_head`: per-position acceptance-probability estimate.

DSparkMarkovHead and DSparkConfidenceHead are shared with the DSV4-style DSpark model.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    nvfp4_gathered_bias,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dspark.markov_argmax import (
        MarkovFusionOperands,
    )

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


def dspark_vocab_shard_enabled() -> bool:
    """Whether the Markov head shards `markov_w2` over the TP group.

    Keyed on the existing `use_local_argmax_reduction` speculative-config
    flag rather than a DSpark-private switch: that flag names exactly this
    trade -- vocab-parallel local argmax instead of an all-gather of full
    logits, greedy selection only -- and the weight layout has to be fixed
    here, at construction, from the same switch the speculator dispatches on
    later.
    """
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.speculative_config is None:
        return False
    return vllm_config.speculative_config.use_local_argmax_reduction


class DSparkMarkovHead(nn.Module):
    """Sequential transition-bias head (low-rank V x r, r x V).

    `markov_w1[token]` embeds the previously sampled token (target vocab,
    `vocab_size`); `markov_w2` projects it to a draft-vocab bias
    (`draft_vocab_size`) added to the base draft logits. The two sizes
    coincide for full-vocab drafts.

    `markov_w1` is always replicated: it is indexed by a token id, so
    sharding it would buy nothing. `markov_w2` is replicated by default for
    the reason the head's original docstring gives -- it runs once per draft
    position, and sharding it would add a full-vocab gather to each. That
    argument holds for the *probabilistic* path, which needs whole processed
    logit rows to verify against, and fails for the greedy path, where the only
    thing read off the vocab axis is an argmax, and an argmax reduces. So under
    `use_local_argmax_reduction` (greedy only, validated by the speculator)
    `markov_w2` becomes vocab-parallel and selection goes through
    `select_top_tokens`, which exchanges one (value, id) pair per rank
    instead of a [B, V] row.
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        retain_weight_for_gather: bool = False,
        quant_method: QuantizeMethodBase | None = None,
        *,
        shard_vocab: bool | None = None,
    ) -> None:
        super().__init__()
        if shard_vocab is None:
            shard_vocab = dspark_vocab_shard_enabled()
        self.shard_vocab = shard_vocab
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size,
            markov_rank,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "markov_w2"),
            disable_tp=not shard_vocab,
            quant_method=quant_method,
        )
        self.markov_w2._retain_weight_for_gather = retain_weight_for_gather
        self.markov_w2.is_w4a16_nvfp4 = False

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """r-dim Markov embedding of `token_ids` ([B] -> [B, r])."""
        return self.markov_w1(token_ids)

    def bias(
        self,
        markov_embed: torch.Tensor,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
        return logits_processor(self.markov_w2, markov_embed)

    def apply_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Apply the Markov bias only to selected rows of `logits`.

        The caller initializes `logits` to `-inf` once for all draft
        positions. This method scatters the corrected candidate values into
        that dense buffer so the normal sampler sees the truncated proposal.
        """
        if self.markov_w2._retain_weight_for_gather and self.markov_w2.is_w4a16_nvfp4:
            # Use the fast Triton kernel for W4A16 NVFP4 weights with BF16
            # activations on CUDA. FP16 activations, non-CUDA accelerators, and
            # CPU use the slower dequantize/scatter fallback below.
            if (
                current_platform.is_cuda()
                and markov_embed.is_cuda
                and markov_embed.dtype == torch.bfloat16
                and values.dtype == torch.bfloat16
                and logits.dtype == torch.bfloat16
            ):
                nvfp4_gathered_bias(
                    markov_embed,
                    self.markov_w2._nvfp4_weight_for_gather,
                    self.markov_w2._nvfp4_weight_scale_for_gather,
                    self.markov_w2._nvfp4_weight_global_scale_for_gather,
                    values,
                    index,
                    logits,
                    scale,
                )
                return logits

            flat_index = index.reshape(-1)
            packed_weight = self.markov_w2._nvfp4_weight_for_gather.index_select(
                0, flat_index
            )
            weight_scale = self.markov_w2._nvfp4_weight_scale_for_gather.index_select(
                0, flat_index
            )
            weight = dequantize_to_dtype(
                packed_weight,
                weight_scale,
                self.markov_w2._nvfp4_weight_global_scale_for_gather,
                dtype=markov_embed.dtype,
                block_size=self.markov_w2._nvfp4_group_size_for_gather,
                swizzle=False,
            ).view(*index.shape, -1)
        else:
            weight = self.markov_w2.weight[index]
            if weight.dtype != markov_embed.dtype:
                raise RuntimeError(
                    "Gathered Markov projection requires unquantized weights or "
                    "retained ModelOpt W4A16 NVFP4 weights."
                )
        corrected = values.unsqueeze(-1).clone()
        corrected.baddbmm_(
            weight,
            markov_embed.unsqueeze(-1),
            beta=1.0,
            alpha=scale,
        )
        return logits.scatter_(1, index, corrected.squeeze(-1))

    def select_top_tokens(
        self,
        markov_embed: torch.Tensor,
        base_shard_logits: torch.Tensor,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        """Greedy draft ids from base logits + this head's transition bias.

        `base_shard_logits` is this rank's slice of the base draft logits
        ([B, V/TP], from `LogitsProcessor.get_shard_logits`); the argmax runs
        over the sum of the two heads, which share the shard layout because
        both are `ParallelLMHead`s over the draft vocab. Returns draft-vocab
        ids -- the caller still applies its draft-to-target mapping.
        """
        return logits_processor.get_top_tokens(
            self.markov_w2, markov_embed, extra_logits=base_shard_logits
        )

    def fusion_operands(
        self, logits_processor: LogitsProcessor
    ) -> "MarkovFusionOperands | None":
        """Operands for the fused Markov step, or `None` to decline.

        The fused kernel reimplements `select_top_tokens` end to end, so
        it can only run where that path is exactly an unquantized GEMV plus an
        add: no soft cap, no logit scale, no separate head dtype, no quantized
        or non-row-major weight. Anything else falls back to the eager chain
        rather than silently changing what the argmax sees.
        """
        from vllm.v1.worker.gpu.spec_decode.dspark.markov_argmax import (
            MarkovFusionOperands,
        )

        w2 = self.markov_w2
        w1 = self.markov_w1.weight
        why: str | None = None
        if logits_processor.soft_cap is not None:
            why = "soft_cap is set"
        elif logits_processor.scale != 1.0:
            why = f"logit scale is {logits_processor.scale}"
        elif logits_processor.head_dtype not in (None, w1.dtype):
            why = f"head_dtype {logits_processor.head_dtype} != {w1.dtype}"
        elif not isinstance(w2.quant_method, UnquantizedEmbeddingMethod):
            why = f"markov_w2 uses {type(w2.quant_method).__name__}"
        elif w1.dtype != w2.weight.dtype:
            why = f"markov_w1 is {w1.dtype}, markov_w2 is {w2.weight.dtype}"
        elif w1.stride(-1) != 1 or w2.weight.stride(-1) != 1:
            why = "a Markov weight is not row-major"
        if why is not None:
            logger.info_once("DSpark: not fusing the Markov step (%s).", why)
            return None
        return MarkovFusionOperands(
            w1=w1,
            w2=w2.weight,
            num_valid=w2.weight.shape[0] - w2.shard_indices.num_org_vocab_padding,
            vocab_start=w2.shard_indices.org_vocab_start_index,
            tp_size=w2.tp_size,
            tp_rank=w2.tp_rank,
        )


class DSparkConfidenceHead(nn.Module):
    """DSpark acceptance-confidence head."""

    def __init__(
        self,
        input_dim: int,
        prefix: str,
        bias: bool = False,
        with_markov: bool = True,
    ) -> None:
        super().__init__()
        self.with_markov = with_markov
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=bias,
            return_bias=False,
            params_dtype=torch.float32,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = (
            torch.cat([hidden, markov_embed], dim=-1) if self.with_markov else hidden
        ).float()
        return self.proj(x).squeeze(-1)


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash Qwen3 backbone + DSpark Markov / confidence heads."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=self.quant_config,
            retain_weight_for_gather=(
                getattr(config, "dspark_draft_topk", None) is not None
            ),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", False):
            with_markov = getattr(config, "confidence_head_with_markov", False)
            input_dim = config.hidden_size
            if with_markov:
                input_dim += config.markov_rank
            self.confidence_head = DSparkConfidenceHead(
                input_dim,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=with_markov,
            )


class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    # The speculator this is a draft model of (see SpeculativeDraftModel).
    speculator_config_class = DSparkSpeculatorConfig
    model_cls = Qwen3DSparkModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.draft_model_config = speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        # Only a pruned draft (subset of the target vocab) needs a d2t mapping to
        # scatter ids up. A draft >= the target vocab (e.g. padded to the target's
        # physical embedding size) maps identically, so no d2t is needed.
        if self.config.draft_vocab_size < self.target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Draft-vocab logits without the d2t scatter: the speculator adds the
        # Markov bias in draft space, then remaps via map_draft_to_target.
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits_shard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Same logits, this rank's vocab columns only (no gather).
        return self.logits_processor.get_shard_logits(self.lm_head, hidden_states)

    def select_draft_token_shard(
        self, markov_embed: torch.Tensor, base_shard_logits: torch.Tensor
    ) -> torch.Tensor:
        return self.model.markov_head.select_top_tokens(
            markov_embed, base_shard_logits, self.logits_processor
        )

    def compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This TP rank's vocab shard of compute_draft_logits, not gathered."""
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        # Map draft-vocab ids to target ids (identity for full-vocab drafts).
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def apply_markov_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.markov_head.apply_bias_gathered(
            markov_embed,
            logits,
            values,
            index,
            self.logits_processor.scale,
        )

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def markov_fusion_operands(self) -> "MarkovFusionOperands | None":
        return self.model.markov_head.fusion_operands(self.logits_processor)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False
        for name, loaded_weight in weights:
            # t2d is training-only; the draft remaps via d2t at sampling time.
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            model_weights[name] = loaded_weight
            # Sets has_own_embed_tokens / has_own_lm_head so load_dspark_model
            # knows whether to keep these or alias the target's.
            process_eagle_weight(self, name)

        # mask_embedding is an unused placeholder param; DSpark masks via the vocab row.
        # embed_tokens / lm_head are optional; when omitted they are shared from
        # the target by load_dspark_model, so skip the unloaded params here.

        orig_to_new_substr = {"mask_embedding": None}
        if not includes_embed_tokens:
            orig_to_new_substr["embed_tokens"] = None
        if not includes_lm_head:
            orig_to_new_substr["lm_head"] = None
        if not includes_draft_id_mapping:
            orig_to_new_substr["draft_id_to_target_id"] = None
        if self.model.confidence_head is None:
            orig_to_new_substr["confidence_head"] = None
        mapper = WeightsMapper(orig_to_new_substr=orig_to_new_substr)
        loader = AutoWeightsLoader(self)
        loader.load_weights(model_weights.items(), mapper=mapper)
