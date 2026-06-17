"""Shared fake timm-like ViT building blocks for fast, CPU-only model tests.

Not a real ViT (no patch embedding/conv, no positional encoding) -- just
enough structure (`blocks`, `embed_dim`, `patch_embed.num_patches`,
`num_prefix_tokens`, and standard `attn.qkv/proj` + `mlp.fc1/fc2` submodule
names) to satisfy `ThunderBackboneAdapter` and exercise LoRA-scoping /
pruning-hook logic without GPU, weights, or datasets.
"""

import torch.nn as nn


class FakeAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * (D ** -0.5)).softmax(-1)
        return self.proj(attn @ v)


class FakeMlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 2)
        self.fc2 = nn.Linear(d * 2, d)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class FakeBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = FakeAttn(d)
        self.mlp = FakeMlp(d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class FakePatchEmbed:
    def __init__(self, n):
        self.num_patches = n


class FakeViT(nn.Module):
    """forward_features takes pre-tokenized input directly (skips patch
    embedding/conv), which is fine for testing pruning/LoRA mechanics."""

    def __init__(self, d=16, n_blocks=6, n_patches=20, num_prefix=1):
        super().__init__()
        self.embed_dim = d
        self.num_prefix_tokens = num_prefix
        self.patch_embed = FakePatchEmbed(n_patches)
        self.blocks = nn.ModuleList([FakeBlock(d) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d)

    def forward_features(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x):
        return self.forward_features(x)[:, 0]
