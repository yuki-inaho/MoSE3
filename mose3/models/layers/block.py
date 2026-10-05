# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
# Modified by the Pi3 and MoSE3 authors; Apache-2.0 text in licenses/LICENSE-DINOv2.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from copy import deepcopy
from typing import Callable

import torch
from torch import nn, Tensor

from .attention import FlashAttentionRope, FlashAttentionRopeTracking
from ..dinov2.layers.layer_scale import LayerScale
from ..dinov2.layers.mlp import Mlp


class BlockRope(nn.Module):
    """Pre-norm transformer block with 2D RoPE attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,  # stochastic depth is training-only; accepted and ignored
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = FlashAttentionRope,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, xpos=None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), xpos=xpos))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class AdaLNBlockRope(nn.Module):
    """BlockRope with adaptive LayerNorm conditioning.

    Wraps an existing BlockRope (taking over its submodules) and adds a per-condition
    (gamma, beta, alpha) modulation on the attention and FFN residual paths. The tracker uses two
    conditions: 0 = search frame, 1 = query frame.
    """

    def __init__(self, block: BlockRope, num_conditions: int = 2):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = block.attn
        self.ls1 = block.ls1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2

        dim = block.norm1.normalized_shape[0]
        # (num_conditions, 6*D): [gamma1, beta1, alpha1, gamma2, beta2, alpha2]
        self.adaln_params = nn.Parameter(torch.zeros(num_conditions, 6 * dim))
        with torch.no_grad():  # identity modulation: gamma = beta = 0, alpha = 1
            self.adaln_params[:, 2 * dim: 3 * dim] = 1.0
            self.adaln_params[:, 5 * dim: 6 * dim] = 1.0

    def forward(self, x: Tensor, xpos, condition_idx: Tensor) -> Tensor:
        params = self.adaln_params[condition_idx]
        if params.dim() == 2:
            # per-frame case: (B*N, 6D) -> (B*N, 1, 6D) to broadcast over the sequence
            params = params.unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)

        norm_x = self.norm1(x) * (1 + gamma1) + beta1
        x = x + self.ls1(self.attn(norm_x, xpos=xpos)) * alpha1

        norm_x = self.norm2(x) * (1 + gamma2) + beta2
        x = x + self.ls2(self.mlp(norm_x)) * alpha2
        return x


class TrackingBlockRope(nn.Module):
    """Tracking-branch counterpart of one pi3 decoder block.

    Runs alongside the frozen pi3 block it is initialized from: the tracking tokens are updated by
    attending to themselves and to the geometry tokens of the same layer. Everything is trainable
    except the geometry-side projections and norms, which keep reading pi3's tokens as pi3 does.
    """

    def __init__(self, original_block: BlockRope):
        super().__init__()
        self.attn = FlashAttentionRopeTracking(original_block.attn)
        self.norm1_track = deepcopy(original_block.norm1)
        self.norm1_geom = deepcopy(original_block.norm1)
        self.norm2 = deepcopy(original_block.norm2)
        self.mlp = deepcopy(original_block.mlp)
        self.ls1 = deepcopy(original_block.ls1)
        self.ls2 = deepcopy(original_block.ls2)

        self.requires_grad_(True)
        for frozen in (self.attn.k_geom, self.attn.v_geom, self.attn.k_norm_geom, self.norm1_geom):
            frozen.requires_grad_(False)

    def forward(self, x_track: Tensor, x_geom: Tensor, xpos_track=None, xpos_geom=None) -> Tensor:
        attn_out = self.attn(
            self.norm1_track(x_track),
            self.norm1_geom(x_geom),
            xpos_track=xpos_track,
            xpos_geom=xpos_geom,
        )
        x_track = x_track + self.ls1(attn_out)
        x_track = x_track + self.ls2(self.mlp(self.norm2(x_track)))
        return x_track
