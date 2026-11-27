# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 on the SM8x parts `nvidia/specialized_decode/ampere` does
not serve (SM86/SM89, and SM80 with `VLLM_DSV41_SM80_TRITON=1`): the Triton
sparse-MLA route. This module holds only its KV record choice, so that shared
DeepSeek-V4.1 code can ask without importing the layer."""

from vllm.platforms import current_platform


def triton_reads_v41_records(kv_cache_dtype: str | None) -> bool:
    """Whether the Triton route keeps DeepSeek's V4.1 KV records: with
    `nvfp4_ds_mla`, the 528-byte MXFP8 sliding-window record and the
    288-byte NVFP4 compressed record of SM100 and SM80. `fp8_ds_mla`, its
    default, keeps the 584-byte V4 records."""
    if kv_cache_dtype != "nvfp4_ds_mla" or not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major == 8
