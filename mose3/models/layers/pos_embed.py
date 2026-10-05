# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Modified from CroCo (https://github.com/naver/croco); see licenses/LICENSE-CroCo.


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import torch


# ----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D (pure PyTorch)
# ----------------------------------------------------------

class RoPE2D(torch.nn.Module):

    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        cache_key = (D, device, dtype)
        cached = self.cache.get(cache_key)
        if cached is None or cached[0].shape[0] < seq_len:
            # Grow geometrically so variable image sizes do not create one
            # cache entry per shape. The sequence length is a safe upper
            # bound for both 2D coordinate axes and is available as a
            # Python integer without synchronizing the GPU.
            cache_len = 1 << (max(1, seq_len) - 1).bit_length()
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(cache_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[cache_key] = (cos, sin)
        cos, sin = self.cache[cache_key]
        return cos[:seq_len], sin[:seq_len]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
        cos, sin = self.get_cos_sin(D, positions.shape[1], tokens.device, tokens.dtype)
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        tokens = torch.cat((y, x), dim=-1)
        return tokens


class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if not (h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)  # (h, w, 2)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()
        return pos
