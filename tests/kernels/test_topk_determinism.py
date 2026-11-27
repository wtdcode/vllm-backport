# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Determinism regression for the DSv4 indexer top-k kernels (#50576).

The indexer feeds top-k indices into an order-sensitive online-softmax, so
temp=0 decoding requires the kernels to be bit-deterministic in both the
selected SET and the output ORDER. The historical kernels claimed output
slots with first-come-first-served atomicAdds, which made the order depend
on warp scheduling on every input, and the set depend on it whenever exact
value ties straddle the k boundary — needle-in-haystack prompts (a tiny
repeated vocabulary) and relu'd logits (a large exact-0.0 mass) hit that
constantly, observed as distinct answers for identical temp=0 requests
above ~52k context.

Each case runs the kernel repeatedly on identical input and asserts:
  1. identical output every run (order + set),
  2. the selected set equals the (value desc, index asc) reference.

Scenarios: `zeros-tie` (boundary inside an exact-0.0 mass), `dup-vals`
(small value alphabet, ties everywhere), `unique` (all-distinct values).
Single-row lengths cover the register histogram (<=32768) and the multi-CTA
radix path (>32768); the batched cases cover the persistent short/medium
rescan, FilteredTopK long rows and sampled top-k.
"""

import pytest
import torch

import vllm._custom_ops as ops
from vllm.platforms import current_platform

K = 512
ITERS = 10
WORKSPACE_BYTES = 1024 * 1024

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only ops"
)


def _make_logits(scenario: str, n: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    if scenario == "zeros-tie":
        return torch.relu(torch.randn(n, generator=g, device="cuda") - 2.5)
    if scenario == "dup-vals":
        alphabet = torch.randn(997, generator=g, device="cuda").abs()
        return alphabet[torch.randint(0, 997, (n,), generator=g, device="cuda")]
    return torch.randperm(n, generator=g, device="cuda").float() / n


def _reference_set(logits_row: torch.Tensor, k: int) -> frozenset[int]:
    order = torch.argsort(-logits_row, stable=True)
    return frozenset(order[:k].tolist())


def _run_persistent(logits, lengths, k, max_len):
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32, device="cuda")
    ws = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    torch.ops._C.persistent_topk(logits, lengths, out, ws, k, max_len)
    return out


def _run_decode(logits, lengths, k, max_len):
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32, device="cuda")
    ops.top_k_per_row_decode(
        logits,
        1,
        lengths,
        out,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        k,
    )
    return out


def _run_prefill(logits, lengths, k, max_len):
    rows = logits.shape[0]
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    out = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")
    ops.top_k_per_row_prefill(
        logits, ks, lengths, out, rows, logits.stride(0), logits.stride(1), k
    )
    return out


KERNELS = {
    "persistent_topk": _run_persistent,
    "top_k_per_row_decode": _run_decode,
    "top_k_per_row_prefill": _run_prefill,
}


@requires_cuda
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("scenario", ["zeros-tie", "dup-vals", "unique"])
@pytest.mark.parametrize("n", [6000, 16000, 60000])
@torch.inference_mode()
def test_topk_deterministic_and_exact(kernel: str, scenario: str, n: int):
    fn = KERNELS[kernel]
    logits = _make_logits(scenario, n).unsqueeze(0)
    lengths = torch.tensor([n], dtype=torch.int32, device="cuda")

    first = fn(logits, lengths, K, n)
    first_row = first[0].tolist()
    got = frozenset(x for x in first_row if x >= 0)
    assert got == _reference_set(logits[0], K), (
        f"{kernel} selected a wrong set for {scenario} n={n}"
    )
    for it in range(ITERS - 1):
        again = fn(logits, lengths, K, n)[0].tolist()
        assert again == first_row, (
            f"{kernel} output differs between identical runs "
            f"({scenario}, n={n}, iter {it + 1})"
        )


@requires_cuda
@torch.inference_mode()
def test_persistent_topk_mixed_lengths_batch():
    """Short rows interleaved with multi-CTA rows in one batch.

    Regression for the triple-buffered histogram chain: a short-row group
    iteration runs no radix rounds, so the next large row's round-0
    histogram inherited stale counts, corrupting its pivot and output.
    """
    g = torch.Generator(device="cuda").manual_seed(11)
    lengths_list = [4000, 60000, 3000, 70000, 62000, 2500, 65000, 61000]
    n_max = max(lengths_list)
    rows = len(lengths_list)
    logits = torch.randn(rows, n_max, generator=g, device="cuda")
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device="cuda")

    first = _run_persistent(logits, lengths, K, n_max)
    for r, ln in enumerate(lengths_list):
        got = frozenset(x for x in first[r].tolist() if x >= 0)
        expected = _reference_set(logits[r, :ln], min(K, ln))
        assert got == expected, f"row {r} (len {ln}) selected a wrong set"
    for _ in range(ITERS - 1):
        again = _run_persistent(logits, lengths, K, n_max)
        assert torch.equal(again, first), "batch output differs between runs"


@requires_cuda
@pytest.mark.parametrize("scenario", ["zeros-tie", "dup-vals"])
@pytest.mark.parametrize(
    ("rows", "row_lengths"),
    [
        # <=32 rows wider than RADIX_THRESHOLD: persistent kernel, every tier.
        (4, [6000, 16000, 30000, 40000]),
        # >32 rows: FilteredTopK register histogram and coarse + rescan.
        (40, [20000, 40000]),
        # >64 long rows: sampled top-k and its short-row fallback.
        (80, [100000, 50000]),
    ],
)
@torch.inference_mode()
def test_persistent_topk_batched_paths(
    scenario: str, rows: int, row_lengths: list[int]
):
    """Batched dispatch tiers stay exact, index-tie-broken and repeatable."""
    n_max = max(row_lengths)
    lengths_list = [row_lengths[r % len(row_lengths)] for r in range(rows)]
    logits = torch.stack([_make_logits(scenario, n_max, seed=r) for r in range(rows)])
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device="cuda")

    first = _run_persistent(logits, lengths, K, n_max)
    for r, ln in enumerate(lengths_list):
        assert frozenset(first[r].tolist()) == _reference_set(logits[r, :ln], K), (
            f"row {r} (len {ln}) selected a wrong set"
        )
    for _ in range(ITERS - 1):
        again = _run_persistent(logits, lengths, K, n_max)
        assert torch.equal(again, first), "batch output differs between runs"
