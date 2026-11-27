# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, overload

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up


def _moe_align_block_size_deterministic(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None,
    scatter_idx: torch.Tensor | None = None,
) -> None:
    """Deterministic replacement for ops.moe_align_block_size.

    The CUDA kernel claims each entry's slot with a first-come-first-served
    atomicAdd, so the permutation within an expert varies run-to-run. The
    Marlin MoE GEMM's k-slice partitioning depends on an entry's slot, so
    the same token's output differs by ulps between identical calls — one of
    the temp=0 nondeterminism sources of #50576. Here entries are ordered by
    (expert, flat index) via a stable sort; every op is deterministic and
    capturable in CUDA graphs (no host sync).
    """
    # Same input contract as the CUDA kernel.
    if not topk_ids.is_contiguous():
        raise RuntimeError("topk_ids must be contiguous")
    numel = topk_ids.numel()
    flat = topk_ids.view(-1).to(torch.int64)
    if expert_map is not None:
        in_range = (flat >= 0) & (flat < expert_map.numel())
        local = torch.where(
            in_range,
            expert_map[flat.clamp(0, expert_map.numel() - 1)].to(torch.int64),
            -1,
        )
    else:
        local = flat
    valid = (local >= 0) & (local < num_experts)
    if scatter_idx is not None:
        rows = torch.arange(numel, dtype=torch.int32, device=topk_ids.device)
        scatter_idx.view(-1).copy_(torch.where(valid, rows, -1))
    # Invalid entries get key == num_experts so they sort last and are dropped.
    key = torch.where(valid, local, num_experts)

    order = torch.argsort(key, stable=True)
    sorted_key = key[order]

    # Integer scatter_add is exact and commutative, so the counts are
    # deterministic; bincount would host-sync and break CUDA graph capture.
    counts_full = torch.zeros(
        num_experts + 1, dtype=torch.int64, device=topk_ids.device
    )
    counts_full.scatter_add_(0, key, torch.ones_like(key))
    counts = counts_full[:num_experts]
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    cum_padded = torch.zeros(num_experts + 1, dtype=torch.int64, device=topk_ids.device)
    torch.cumsum(padded_counts, 0, out=cum_padded[1:])
    cum_unpadded = torch.zeros_like(cum_padded)
    torch.cumsum(counts, 0, out=cum_unpadded[1:])

    # Position in the padded layout: expert base + rank within the expert.
    # After the stable sort, valid entries are grouped by expert in ascending
    # order, so the rank is just the offset from the expert's unpadded base.
    # Invalid entries land in a scratch overflow region past the real layout
    # (static shapes only: boolean-mask compaction would host-sync inside
    # CUDA graph capture).
    max_padded = sorted_ids.numel()
    pos_in_sorted = torch.arange(numel, dtype=torch.int64, device=topk_ids.device)
    safe_key = sorted_key.clamp(max=num_experts - 1)
    rank = pos_in_sorted - cum_unpadded[safe_key]
    dest_valid = cum_padded[safe_key] + rank
    dest_invalid = max_padded + pos_in_sorted - cum_unpadded[num_experts]
    dest = torch.where(sorted_key < num_experts, dest_valid, dest_invalid)

    scratch = torch.full(
        (max_padded + numel,),
        numel,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    scratch.scatter_(0, dest, order.to(torch.int32))
    sorted_ids.copy_(scratch[:max_padded])

    # Per-block expert ids: block b (starting at b*block_size) belongs to the
    # expert whose padded range contains it; blocks past the padded total are
    # inactive (-1), matching the CUDA kernel.
    nblocks = expert_ids.numel()
    block_starts = (
        torch.arange(nblocks, dtype=torch.int64, device=topk_ids.device) * block_size
    )
    block_expert = torch.searchsorted(cum_padded[1:], block_starts, right=True)
    total_padded = cum_padded[num_experts]
    block_expert = torch.where(block_starts < total_padded, block_expert, -1)
    expert_ids.copy_(block_expert.to(torch.int32))
    num_tokens_post_pad.copy_(total_padded.to(torch.int32).reshape(1))


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: bool,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
): ...


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    return_scatter_idx: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Args:
        topk_ids: A tensor of shape [total_tokens, top_k] representing the
            top-k expert indices for each token.
        block_size: The block size used in block matrix multiplication.
        num_experts: The total number of experts.
        expert_map: A tensor of shape [num_experts] that maps the expert index
            from the global space to the local index space of the current
            expert parallel shard. If the expert is not in the current expert
            parallel shard, the mapping is set to -1.
        pad_sorted_ids: A flag indicating whether the sorted_token_ids length
            should be padded to a multiple of block_size,
        ignore_invalid_experts: A flag indicating whether to ignore invalid
            experts. When False, all expert_ids in topk_ids will participate in
            counting and ranking, but invalid experts in expert_ids will be marked
            as -1. When True, all invalid expert_ids in topk_ids will be ignored
            and will not participate in counting or ranking, and there will be no
            -1 in expert_ids.
        return_scatter_idx: Whether to additionally return a masked identity
            mapping for the original routed rows.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.
    - scatter_idx: Only returned when return_scatter_idx=True. An int32 tensor
        shaped [topk_ids.numel(), 1], containing the original routed row index
        for valid routes and -1 otherwise. With expert_map, this requires
        ignore_invalid_experts=True.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.

    """
    if return_scatter_idx and expert_map is not None and not ignore_invalid_experts:
        raise ValueError("scatter_idx with expert_map requires ignore_invalid_experts")
    scatter_idx = None
    if return_scatter_idx:
        scatter_idx = torch.empty(
            (topk_ids.numel(), 1),
            dtype=torch.int32,
            device=topk_ids.device,
        )
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    if envs.VLLM_DETERMINISTIC_MOE_ALIGN:
        _moe_align_block_size_deterministic(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map if ignore_invalid_experts else None,
            scatter_idx,
        )
    else:
        ops.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map if ignore_invalid_experts else None,
            scatter_idx,
        )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    if scatter_idx is not None:
        return sorted_ids, expert_ids, num_tokens_post_pad, scatter_idx
    return sorted_ids, expert_ids, num_tokens_post_pad


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Args:
        max_tokens_per_batch (int): Number of tokens in each batch (both
            valid and invalid).
        block_size (int): block_size to align the data to.
        expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
            the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.

    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.

    """
    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad
