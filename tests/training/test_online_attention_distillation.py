import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.online_attention_distillation import (  # noqa: E402
    FrozenTimmAttentionTeacher,
    spearman_correlation,
    topk_recall,
)


class FakeAttention(nn.Module):
    def __init__(self, dim=8, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(output)


class FakeBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.attn = FakeAttention(dim)

    def forward(self, x):
        return x + self.attn(x)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x[:, 0]


class FakeAdapter:
    n_blocks = 2
    num_prefix_tokens = 1

    def __init__(self, model):
        self.model = model

    def get_attn_module(self, index):
        return self.model.blocks[index].attn


def test_teacher_returns_online_patch_targets():
    model = FakeBackbone()
    teacher = FrozenTimmAttentionTeacher(
        model, FakeAdapter(model), source_layer=0, target_layer=1
    )
    source, target = teacher(torch.randn(3, 6, 8))
    assert source.shape == (3, 5, 8)
    assert target.shape == (3, 5)
    assert torch.allclose(target.sum(-1), torch.ones(3), atol=1e-6)
    assert not source.requires_grad
    probe = nn.Linear(8, 1)
    probe(source).sum().backward()
    assert probe.weight.grad is not None
    assert all(not parameter.requires_grad for parameter in model.parameters())
    teacher.close()


def test_rank_metrics_are_exact_for_identical_scores():
    scores = torch.tensor([[0.1, 0.3, 0.2, 0.4]])
    assert torch.allclose(spearman_correlation(scores, scores), torch.ones(1))
    assert torch.allclose(topk_recall(scores, scores, 0.5), torch.ones(1))
