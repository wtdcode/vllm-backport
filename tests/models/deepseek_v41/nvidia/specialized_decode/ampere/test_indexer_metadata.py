# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM80 indexer metadata builder's index K cache choice: MXFP4, by
default or named, although the stock check accepts it on SM100 only. Host
arithmetic only."""

from types import SimpleNamespace

import pytest

from vllm.config import AttentionConfig
from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.indexer_metadata import (  # noqa: E501
    AmpereIndexerMetadataBuilder,
)


def _builder(indexer_kv_dtype: str) -> AmpereIndexerMetadataBuilder:
    builder = object.__new__(AmpereIndexerMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        attention_config=AttentionConfig(indexer_kv_dtype=indexer_kv_dtype)
    )
    return builder


@pytest.mark.parametrize("indexer_kv_dtype", ["auto", "mxfp4"])
def test_mxfp4_index_cache_accepted(indexer_kv_dtype: str) -> None:
    assert _builder(indexer_kv_dtype)._indexer_uses_fp4()


def test_fp8_index_cache_rejected() -> None:
    with pytest.raises(ValueError, match="MXFP4 indexer cache only"):
        _builder("fp8")._indexer_uses_fp4()
