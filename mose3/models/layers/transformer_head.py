from copy import deepcopy
from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from .attention import FlashAttentionRope
from .block import BlockRope, AdaLNBlockRope
from ..dinov2.layers import Mlp


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,  # gradient checkpointing is training-only; accepted and ignored
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()

        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for blk in self.blocks:
            hidden = blk(hidden, xpos=xpos)
        return self.linear_out(hidden)


class TemporalTrackingDecoder(nn.Module):
    """Track decoder with interleaved per-frame / cross-frame attention and AdaLN.

    Built from a ``TransformerDecoder``: every original block becomes a LOCAL layer (attention
    within a frame, tokens shaped ``(B*N, seq, D)``) followed by a GLOBAL layer (attention across
    all frames, ``(B, N*seq, D)``), so the depth doubles. AdaLN conditioning tells each layer
    whether a token belongs to the query frame or to a search frame.
    """

    def __init__(self, td: TransformerDecoder, num_conditions: int = 2):
        super().__init__()
        self.projects = td.projects
        self.linear_out = td.linear_out

        blocks = []
        for blk in td.blocks:
            blocks.append(AdaLNBlockRope(blk, num_conditions=num_conditions))            # local
            blocks.append(AdaLNBlockRope(deepcopy(blk), num_conditions=num_conditions))  # global
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden, xpos, N, condition_idx):
        """
        Args:
            hidden:        (B*N, seq, in_dim)
            xpos:          (B*N, seq, 2)   2D RoPE positions
            N:             number of frames
            condition_idx: (B*N,)  0 = search frame, 1 = query frame
        Returns:
            (B*N, seq, out_dim)
        """
        B_N, seq, _ = hidden.shape
        B = B_N // N

        hidden = self.projects(hidden)

        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:  # local
                h = hidden.reshape(B_N, seq, -1)
                p = xpos.reshape(B_N, seq, -1)
                cond = condition_idx
            else:           # global
                h = hidden.reshape(B, N * seq, -1)
                p = xpos.reshape(B, N * seq, -1)
                cond = condition_idx.reshape(B, N).unsqueeze(-1).expand(B, N, seq).reshape(B, N * seq)
            hidden = blk(h, xpos=p, condition_idx=cond)

        hidden = hidden.reshape(B_N, seq, -1)
        return self.linear_out(hidden)


# Adapted from DUSt3R, Copyright (C) 2024-present Naver Corporation. CC BY-NC-SA 4.0 (non-commercial);
# see licenses/LICENSE-DUSt3R.
class LinearPts3d(nn.Module):
    """Linear per-patch head (from DUSt3R): each token predicts a patch_size x patch_size block."""

    def __init__(self, patch_size, dec_embed_dim, output_dim=3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(dec_embed_dim, output_dim * self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,C,H,W

        return feat.permute(0, 2, 3, 1)
