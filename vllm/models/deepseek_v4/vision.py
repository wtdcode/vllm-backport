# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vision tower and aligner for DeepSeek-V4-Flash-Vision.

One image is encoded independently: patches attend bidirectionally over the
whole image with 2D RoPE, then the aligner folds each ``downsample_ratio``
square of ViT tokens into a single language-model token.

The tower is small (~0.3B) and is replicated on every rank rather than sharded.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


@lru_cache(8)
def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the 2D rotary tables for an ``n_h`` x ``n_w`` patch grid.

    Half the rotary dimensions encode the row index and half the column index,
    so a patch is addressed by both axes.

    The tables are built on CPU and then moved: a single fp32 ULP of difference
    here (which is what building them on device costs) grows to ~1e-1 in the
    tower output, because bf16 rounding compounds across the 32 residual
    blocks. Building on CPU keeps us bit-exact against the reference
    implementation. They are small and cached, so the copy is not on any hot
    path.

    Args:
        n_h: Patch-grid height.
        n_w: Patch-grid width.
        dim: Rotary dimensions per axis (``head_dim // 2``).
        theta: Rotary base.
        device: Device the tables are moved to.

    Returns:
        ``(cos, sin)``, each ``[n_h * n_w, 1, dim]``.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return (
        freqs.cos().unsqueeze(1).to(device),
        freqs.sin().unsqueeze(1).to(device),
    )


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4VisionRMSNorm(nn.Module):
    """RMSNorm computed in fp32, as in the reference vision tower."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV4PatchEmbed(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.proj = nn.Linear(3 * config.vision_patch_size**2, config.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.norm1 = DeepseekV4VisionRMSNorm(config.vision_dim)
        self.attn = DeepseekV4VisionAttention(config)
        self.norm2 = DeepseekV4VisionRMSNorm(config.vision_dim)
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4ViT(nn.Module):
    """Bidirectional ViT over the patches of a single image."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV4PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [DeepseekV4VisionBlock(config) for _ in range(config.vision_n_layers)]
        )
        self.norm = DeepseekV4VisionRMSNorm(config.vision_dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Encode one image.

        Args:
            patches: ``[n_h * n_w, 3, patch, patch]`` normalized pixels.
            n_h: Patch-grid height.
            n_w: Patch-grid width.

        Returns:
            ``[n_h * n_w, vision_dim]`` patch features.
        """
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(
            n_h, n_w, self.rope_dim, self.rope_theta, x.device
        )
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4Aligner(nn.Module):
    """Fold ``r`` x ``r`` ViT tokens into one language-model token."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Args:
            x: ``[n_h * n_w, vision_dim]`` patch features.
            n_h: Patch-grid height.
            n_w: Patch-grid width.

        Returns:
            ``[ceil(n_h / r) * ceil(n_w / r), hidden_size]`` image tokens, in
            row-major order over the downsampled grid.
        """
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
