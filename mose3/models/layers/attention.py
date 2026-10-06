# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
# Modified by the Pi3 and MoSE3 authors; Apache-2.0 text in licenses/LICENSE-DINOv2.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from copy import deepcopy

import torch
from torch import Tensor
from torch import nn
from torch.nn.attention import SDPBackend
from torch.nn.functional import scaled_dot_product_attention


def _flash_attention_supported() -> bool:
    """The flash SDPA kernel needs Ampere or newer (sm80+); on Turing (RTX 20xx) it must not be requested."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def _sdpa(q: Tensor, k: Tensor, v: Tensor, half_dtypes) -> Tensor:
    """Flash kernel for half precision, math / mem-efficient kernels otherwise."""
    if q.dtype in half_dtypes and _flash_attention_supported():
        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return scaled_dot_product_attention(q, k, v)
    with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        return scaled_dot_product_attention(q, k, v)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)


class FlashAttention(Attention):
    """Self-attention of the DINOv2 encoder blocks."""

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]

        x = _sdpa(q, k, v, half_dtypes=(torch.bfloat16,))
        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.rope = rope


class FlashAttentionRope(AttentionRope):
    """Self-attention with 2D RoPE (Pi3 decoder and the per-task decoders)."""

    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        x = _sdpa(q, k, v, half_dtypes=(torch.bfloat16,))
        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttentionRopeTracking(nn.Module):
    """Attention of the tracking branch.

    Queries come from the tracking tokens; keys/values are the concatenation of the tracking
    tokens' own K/V and K/V projected from the frozen geometry tokens of the same layer, so one
    softmax allocates attention between the two streams.

    Built from the pi3 decoder attention it forks off: the fused QKV is duplicated into a trainable
    tracking-side Q/K/V and a frozen geometry-side K/V.
    """

    def __init__(self, original_attn: AttentionRope):
        super().__init__()
        self.num_heads = original_attn.num_heads
        dim = original_attn.qkv.in_features
        self.head_dim = dim // self.num_heads
        bias = original_attn.qkv.bias is not None

        self.q_track = nn.Linear(dim, dim, bias=bias)
        self.k_track = nn.Linear(dim, dim, bias=bias)
        self.v_track = nn.Linear(dim, dim, bias=bias)
        self.k_geom = nn.Linear(dim, dim, bias=bias)
        self.v_geom = nn.Linear(dim, dim, bias=bias)
        with torch.no_grad():
            q, k, v = original_attn.qkv.weight.chunk(3, dim=0)
            for lin, w in ((self.q_track, q), (self.k_track, k), (self.v_track, v), (self.k_geom, k), (self.v_geom, v)):
                lin.weight.copy_(w)
            if bias:
                q, k, v = original_attn.qkv.bias.chunk(3, dim=0)
                for lin, b in ((self.q_track, q), (self.k_track, k), (self.v_track, v), (self.k_geom, k), (self.v_geom, v)):
                    lin.bias.copy_(b)

        self.q_norm_track = deepcopy(original_attn.q_norm)
        self.k_norm_track = deepcopy(original_attn.k_norm)
        self.k_norm_geom = deepcopy(original_attn.k_norm)

        self.rope = original_attn.rope
        self.proj = deepcopy(original_attn.proj)

    def forward(self, x_track: Tensor, x_geom: Tensor, xpos_track=None, xpos_geom=None) -> Tensor:
        B, Nt, C = x_track.shape
        Ng = x_geom.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.q_track(x_track).view(B, Nt, H, D).transpose(1, 2)
        k_geom = self.k_geom(x_geom).view(B, Ng, H, D).transpose(1, 2)
        k_track = self.k_track(x_track).view(B, Nt, H, D).transpose(1, 2)
        v_geom = self.v_geom(x_geom).view(B, Ng, H, D).transpose(1, 2)
        v_track = self.v_track(x_track).view(B, Nt, H, D).transpose(1, 2)

        q = self.q_norm_track(q).to(v_geom.dtype)
        k_geom = self.k_norm_geom(k_geom).to(v_geom.dtype)
        k_track = self.k_norm_track(k_track).to(v_geom.dtype)

        if self.rope is not None:
            if xpos_track is not None:
                q = self.rope(q, xpos_track)
                k_track = self.rope(k_track, xpos_track)
            if xpos_geom is not None:
                k_geom = self.rope(k_geom, xpos_geom)

        k = torch.cat([k_track, k_geom], dim=2)
        v = torch.cat([v_track, v_geom], dim=2)
        out = _sdpa(q, k, v, half_dtypes=(torch.float16, torch.bfloat16))

        out = out.transpose(1, 2).reshape(B, Nt, C)
        return self.proj(out)
