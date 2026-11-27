# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash sparse and sliding-window MLA backends for SM8x."""

from vllm.models.hy_v4.ampere.sparse_mla import HYV4AmpereMLASparseBackend


class FlashAmpereSparseBackend(HYV4AmpereMLASparseBackend):
    pass


class FlashAmpereSWABackend(FlashAmpereSparseBackend):
    # Window rows index only live pages, which the ragged kernels read paged.
    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True
