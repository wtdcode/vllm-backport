# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 e4m3fn encode/decode for Triton kernels on pre-SM89 CUDA.

Triton refuses every conversion between ``fp8e4nv`` and other float types
below SM89 (``bitcast`` is allowed), so these substitute:

- store side: a manual round-to-nearest-even saturating encoder on the fp32
  bit pattern, bit-exact with ``x.to(tl.float8e4nv)`` /
  ``torch.Tensor.to(torch.float8_e4m3fn)`` (satfinite semantics).
- load side: a 256-entry bf16 lookup table. Measured 1.35-1.48x faster than
  an ALU unpack in the sparse-MLA decode kernels -- the table stays in L1
  while the ALU version's live values spill (68 spills vs 6), so prefer it
  wherever a kernel can accept the extra pointer argument.

``USE_FNUZ`` branches stay on the hardware convert: FNUZ only occurs on
gfx942, which has it.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def native_fp8_cast_supported() -> bool:
    """Whether Triton may emit native fp8e4nv converts on this device.

    Only CUDA below SM89 lacks the convert; other Triton backends
    (XPU/ROCm) keep their native paths.
    """
    if not current_platform.is_cuda():
        return True
    return current_platform.supports_fp8()


# Frozen into Triton kernels at first compile (globals referenced inside
# @triton.jit are compile-time constants), so SM89+ keeps the hardware
# convert and SM80 transparently gets the software path with no launcher
# plumbing. Defaults to the software path: it is bit-exact everywhere, just
# slower, so a failed probe degrades performance rather than compilation.
try:
    _NATIVE_FP8_CAST = tl.constexpr(native_fp8_cast_supported())
except Exception:
    _NATIVE_FP8_CAST = tl.constexpr(False)


_E4M3FN_BF16_LUT_CACHE: dict[tuple[torch.device, float | None], torch.Tensor] = {}


def get_e4m3fn_bf16_lut(
    device: torch.device, nan_value: float | None = None
) -> torch.Tensor:
    """256-entry e4m3fn -> bf16 decode table.

    ``nan_value`` replaces the two NaN encodings (0x7F/0xFF, signed): pass
    a finite magnitude where NaN would poison a downstream reduction (the
    indexer's dot product uses +-480), or leave it None for kernels that
    scrub NaN themselves and need NaN in, NaN out.
    """
    key = (device, nan_value)
    lut = _E4M3FN_BF16_LUT_CACHE.get(key)
    if lut is None:
        lut = (
            torch.arange(256, dtype=torch.uint8, device=device)
            .view(torch.float8_e4m3fn)
            .to(torch.bfloat16)
        )
        if nan_value is not None:
            lut[0x7F] = nan_value
            lut[0xFF] = -nan_value
        _E4M3FN_BF16_LUT_CACHE[key] = lut
    return lut


@triton.jit
def _f32_to_e4m3fn_u8(x):
    """fp32 -> e4m3fn byte: RNE, saturating to +-448, NaN -> 0x7F | sign.

    Unified integer rounding: the f32 mantissa (implicit bit materialized)
    is shifted so the kept bits land at the e4m3 granularity; the same
    add-carry then handles mantissa-overflow into the exponent and
    subnormal -> min-normal promotion.
    """
    fbits = x.to(tl.float32).to(tl.uint32, bitcast=True)
    sign8 = ((fbits >> 24) & 0x80).to(tl.uint8)
    mag = fbits & 0x7FFFFFFF

    is_nan = mag > 0x7F800000

    exp = (mag >> 23).to(tl.int32) - 127
    mant = (mag & 0x7FFFFF) | 0x800000

    # Normal e4m3 target keeps 3 mantissa bits -> drop 20; subnormal
    # targets (exp < -6) drop progressively more. The cap keeps the shift in
    # range for f32 subnormals and zero, which give exp <= -126; every value
    # reaching it has rest < half, so it still rounds to zero.
    shift = tl.minimum(20 + tl.maximum(-6 - exp, 0), 30)
    keep = (mant >> shift).to(tl.uint32)
    rest = mant & ((1 << shift) - 1)
    half = (1 << (shift - 1)).to(tl.uint32)
    round_up = (rest > half) | ((rest == half) & ((keep & 1) == 1))
    keep = keep + round_up.to(tl.uint32)

    # keep carries the implicit bit for normals: val = ((E-1)<<3) + keep.
    val_normal = (((exp + 6) << 3).to(tl.uint32) + keep).to(tl.uint32)
    val = tl.where(exp >= -6, val_normal, keep)
    # satfinite: every overflow (incl. inf and mantissa-carry past 448)
    # lands above 0x7E and clamps to the max finite.
    val = tl.minimum(val, 0x7E)
    val = tl.where(is_nan, 0x7F, val)
    return (val.to(tl.uint8) | sign8).to(tl.uint8)


@triton.jit
def _encode_e4m3fn_u8(x):
    """Drop-in for ``x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)``."""
    if _NATIVE_FP8_CAST:
        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        return _f32_to_e4m3fn_u8(x)


@triton.jit
def _encode_fp8_u8(x, USE_FNUZ: tl.constexpr):
    """FNUZ-aware byte encode: e4m3fnuz on gfx942, e4m3fn elsewhere."""
    if USE_FNUZ:
        return x.to(tl.float8e4b8).to(tl.uint8, bitcast=True)
    else:
        return _encode_e4m3fn_u8(x)


@triton.jit
def _e4m3fn_to_f32_alu(u):
    """e4m3fn byte -> exact f32 via integer math, NaN preserved.

    For kernels that cannot take a LUT pointer; prefer ``_decode_fp8_lut``.
    """
    u32 = u.to(tl.uint32)
    sign = tl.where((u32 & 0x80) != 0, -1.0, 1.0)
    exp = ((u32 >> 3) & 0xF).to(tl.float32)
    mant = (u32 & 0x7).to(tl.float32)
    is_normal = exp > 0.0
    m = tl.where(is_normal, mant + 8.0, mant)
    e = tl.where(is_normal, exp, 1.0) - 10.0
    val = sign * m * tl.exp2(e)
    return tl.where((u32 & 0x7F) == 0x7F, float("nan"), val)


@triton.jit
def _decode_fp8_f32(u, USE_FNUZ: tl.constexpr):
    """FNUZ-aware byte decode to f32, without a LUT."""
    if USE_FNUZ:
        return u.to(tl.float8e4b8, bitcast=True).to(tl.float32)
    elif _NATIVE_FP8_CAST:
        return u.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        return _e4m3fn_to_f32_alu(u)


@triton.jit
def _decode_fp8_lut(u, USE_FNUZ: tl.constexpr, lut_ptr):
    """FNUZ-aware byte decode; pre-SM89 CUDA reads ``lut_ptr``.

    ``lut_ptr`` must come from :func:`get_e4m3fn_bf16_nan_lut` and is
    ignored (compile-time) wherever the hardware convert exists.
    """
    if USE_FNUZ:
        return u.to(tl.float8e4b8, bitcast=True).to(tl.float32)
    elif _NATIVE_FP8_CAST:
        return u.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        return tl.load(lut_ptr + u.to(tl.uint32))


def e4m3fn_cache_as_u8(
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Kernel-ready `(cache, lut)` for an fp8 cache read by a Triton kernel.

    Pre-SM89 CUDA gets a uint8 view of an e4m3fn cache plus the bf16 table
    for `_decode_fp8_lut`; everything else is returned unchanged with no
    table, so the kernel keeps its native fp8 load.
    """
    if cache.dtype != torch.float8_e4m3fn or native_fp8_cast_supported():
        return cache, None
    return cache.view(torch.uint8), get_e4m3fn_bf16_lut(cache.device)
