"""Acoustics foundation model v1.

Predicts the complex transfer function H(f) between a source and a receiver
in a 2D urban geometry:

    (occupancy grid, facade absorptions, src pos, rcv pos) -> H at N_FREQS

Structure:
- CNN encoder over the occupancy grid (with coordinate channels) producing a
  feature map of geometry tokens.
- A query embedding built from Fourier features of the source/receiver
  positions plus the absorption scalars.
- Cross-attention blocks: the query attends to geometry tokens.
- Two linear heads: Re and Im of H at each frequency bin.

Positions are normalized to [0, 1] over the fixed physical domain.
"""

import numpy as np
import torch
import torch.nn as nn


def fourier_features(pos, n_bands=16):
    """pos (B, 2) in [0,1] -> (B, 4*n_bands) sin/cos features."""
    freqs = 2.0 ** torch.arange(n_bands, device=pos.device, dtype=pos.dtype)
    ang = 2 * np.pi * pos.unsqueeze(-1) * freqs  # (B, 2, n_bands)
    feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    return feat.flatten(1)


class GeometryEncoder(nn.Module):
    """CNN over the occupancy grid with coordinate channels."""

    def __init__(self, channels=256):
        super().__init__()
        widths = [32, 64, 128, channels]
        layers = []
        in_ch = 3  # occupancy + x/y coordinate channels
        for w in widths:
            layers += [nn.Conv2d(in_ch, w, 3, stride=2, padding=1),
                       nn.GroupNorm(8, w), nn.GELU(),
                       nn.Conv2d(w, w, 3, padding=1),
                       nn.GroupNorm(8, w), nn.GELU()]
            in_ch = w
        self.net = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, occ):
        """occ (B, H, W) -> tokens (B, h*w, C)."""
        B, H, W = occ.shape
        ys = torch.linspace(0, 1, H, device=occ.device)
        xs = torch.linspace(0, 1, W, device=occ.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([yy, xx]).expand(B, 2, H, W)
        x = torch.cat([occ.unsqueeze(1), coords], dim=1)
        fmap = self.net(x)  # (B, C, h, w)
        return fmap.flatten(2).transpose(1, 2)  # (B, h*w, C)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=8, mlp_ratio=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim))

    def forward(self, q, kv):
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv),
                                self.norm_kv(kv), need_weights=False)
        q = q + attn_out
        q = q + self.mlp(self.norm2(q))
        return q


class AcousticsModelV1(nn.Module):
    def __init__(self, n_freqs=64, dim=320, n_blocks=6, n_heads=8,
                 n_fourier=16):
        super().__init__()
        self.encoder = GeometryEncoder(channels=dim)
        self.token_pos = nn.Parameter(torch.zeros(1, 4 * 6, dim))
        nn.init.trunc_normal_(self.token_pos, std=0.02)

        query_in = 2 * (4 * n_fourier) + 2  # src ff + rcv ff + alphas
        self.n_fourier = n_fourier
        self.query_mlp = nn.Sequential(
            nn.Linear(query_in, dim), nn.GELU(), nn.Linear(dim, dim))

        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(dim, n_heads) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Linear(2 * dim, 2 * n_freqs))
        self.n_freqs = n_freqs

    def forward(self, occ, src, rcv, alpha):
        """
        occ (B, H, W) occupancy; src, rcv (B, Q, 2) normalized positions
        for Q queries sharing each grid; alpha (B, 2).
        Returns (B, Q, n_freqs) complex.
        """
        tokens = self.encoder(occ)
        if tokens.shape[1] == self.token_pos.shape[1]:
            tokens = tokens + self.token_pos

        B, Q, _ = src.shape
        qin = torch.cat([
            fourier_features(src.reshape(B * Q, 2), self.n_fourier),
            fourier_features(rcv.reshape(B * Q, 2), self.n_fourier),
            alpha.unsqueeze(1).expand(B, Q, 2).reshape(B * Q, 2),
        ], dim=1)
        q = self.query_mlp(qin).reshape(B, Q, -1)

        for blk in self.blocks:
            q = blk(q, tokens)

        out = self.head(q)  # (B, Q, 2*n_freqs)
        re, im = out.split(self.n_freqs, dim=-1)
        return torch.complex(re, im)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
