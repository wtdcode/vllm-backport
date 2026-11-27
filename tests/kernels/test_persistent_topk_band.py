# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the k < n < 2k band of persistent_topk.

An external report against this branch (vllm-project/vllm#50576, comment
5158563519) claimed the radix persistent_topk returns wrong indices when the
candidate count falls strictly between k and 2k, observed as degenerate
output at prompt lengths 2049-4096 (index_topk=512, compress_ratio=4).

The claim was audited on SM80 (2026-08-03) and did NOT reproduce, at either
level: 2x 286/286 correct end-to-end requests across every kernel-path tier,
and 183/183 direct kernel checks (this test's cases). The reporter ran a
0.25.2-era fork with a vendored third-party kernel, not this tree's
csrc/attention/dsa/topk/topk.cu. This test pins the in-band behaviour so any
future topk.cu change that regresses it fails loudly rather than silently.

Poison values above ``lengths`` mirror production, which calls the indexer
logits kernels with clean_logits=False: memory beyond seq_len is garbage,
and the kernel must respect ``lengths`` internally.
"""

import pytest
import torch

from vllm.platforms import current_platform

K = 512
MAX_SEQ_LEN = 4352
WORKSPACE_BYTES = 1024 * 1024
POISON = 1.0e9

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only op"
)


@requires_cuda
@pytest.mark.parametrize(
    "n",
    [
        496,  # n < k: top-k selects everything
        523,  # in the reported corruption band (k < n < 2k) ...
        749,
        971,
        1021,
        4096,  # n > 2k
    ],
)
@pytest.mark.parametrize("num_rows", [1, 8, 32])
def test_persistent_topk_in_band(n: int, num_rows: int):
    torch.manual_seed(0)
    device = "cuda"
    logits = torch.full(
        (num_rows, MAX_SEQ_LEN), POISON, dtype=torch.float32, device=device
    )
    logits[:, :n] = torch.randn(num_rows, n, device=device)
    lengths = torch.full((num_rows,), n, dtype=torch.int32, device=device)
    out = torch.full((num_rows, K), -1, dtype=torch.int32, device=device)
    ws = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    torch.ops._C.persistent_topk(logits, lengths, out, ws, K, MAX_SEQ_LEN)
    # .tolist() below synchronizes; no explicit torch.cuda call needed.

    kk = min(n, K)
    for r in range(num_rows):
        ref = set(torch.topk(logits[r, :n], kk).indices.tolist())
        got_raw = [i for i in out[r].tolist() if i >= 0]
        # No index may point past the row's candidate count (poison region).
        assert all(i < n for i in got_raw), f"row {r}: index >= n leaked"
        got = set(got_raw)
        assert len(got) == len(got_raw), f"row {r}: duplicate indices"
        assert got == ref, (
            f"row {r} (n={n}): {len(ref - got)} missing / "
            f"{len(got - ref)} extra vs torch.topk"
        )


# Rows at or below 32 dispatch to FilteredTopK only while `max_seq_len` stays
# within RADIX_THRESHOLD; past it topk.cu keeps the multi-CTA radix path, whose
# cooperative barrier and RadixRowState workspace nothing else exercises at
# these row counts. One case per side of that boundary, so a dispatch change
# cannot silently leave one path untested.
@requires_cuda
@pytest.mark.parametrize("n", [32768, 40960])
@pytest.mark.parametrize("num_rows", [1, 4])
def test_persistent_topk_across_radix_threshold(n: int, num_rows: int):
    torch.manual_seed(0)
    device = "cuda"
    width = 40960  # > RADIX_THRESHOLD (32768), 128-aligned like production
    logits = torch.full((num_rows, width), POISON, dtype=torch.float32, device=device)
    logits[:, :n] = torch.randn(num_rows, n, device=device)
    lengths = torch.full((num_rows,), n, dtype=torch.int32, device=device)
    out = torch.full((num_rows, K), -1, dtype=torch.int32, device=device)
    ws = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    torch.ops._C.persistent_topk(logits, lengths, out, ws, K, width)

    for r in range(num_rows):
        ref = set(torch.topk(logits[r, :n], K).indices.tolist())
        got_raw = [i for i in out[r].tolist() if i >= 0]
        assert all(i < n for i in got_raw), f"row {r}: index >= n leaked"
        got = set(got_raw)
        assert len(got) == len(got_raw), f"row {r}: duplicate indices"
        assert got == ref, (
            f"row {r} (n={n}): {len(ref - got)} missing / "
            f"{len(got - ref)} extra vs torch.topk"
        )
