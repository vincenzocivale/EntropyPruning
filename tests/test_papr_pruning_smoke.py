import torch
import torch.nn as nn

from src.models import (
    LinearProbingClassifier,
    PaPrPrunedClassifier,
    ThunderBackboneAdapter,
    apply_papr_to_tokens,
)
from tests._fakes import FakeBlock


def test_apply_papr_keeps_prefix_and_top_scoring_patches():
    tokens = torch.arange(5 * 3, dtype=torch.float32).view(1, 5, 3)
    proposal = torch.tensor([[[[0.1, 0.9], [0.4, 0.7]]]])

    pruned, indices = apply_papr_to_tokens(
        tokens,
        proposal,
        keep_ratio=0.5,
        num_prefix_tokens=1,
        grid_size=(2, 2),
        return_indices=True,
    )

    assert indices.tolist() == [[1, 3]]
    assert torch.equal(pruned[:, 0], tokens[:, 0])
    assert torch.equal(pruned[:, 1:], tokens[:, 1:][:, [1, 3]])


class FakeImagePatchEmbed(nn.Module):
    def __init__(self, embed_dim=8, patch_size=2, img_size=4):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        side = img_size // patch_size
        self.grid_size = (side, side)
        self.num_patches = side * side

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class FakeImageViT(nn.Module):
    def __init__(self, embed_dim=8, n_blocks=3, n_patches=4, num_prefix=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix
        self.patch_embed = FakeImagePatchEmbed(embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, num_prefix, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_prefix + n_patches, embed_dim))
        self.pos_drop = nn.Identity()
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList([FakeBlock(embed_dim) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(embed_dim)

    def _pos_embed(self, x):
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return self.pos_drop(torch.cat([cls, x], dim=1) + self.pos_embed)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x):
        return self.forward_features(x)[:, 0]


class FixedProposal(nn.Module):
    def forward(self, x):
        base = torch.tensor(
            [[[[0.1, 0.9], [0.4, 0.7]]]],
            dtype=x.dtype,
            device=x.device,
        )
        return base.expand(x.shape[0], -1, -1, -1)


def test_papr_pruned_classifier_forward_shape_and_token_count():
    backbone = FakeImageViT()
    adapter = ThunderBackboneAdapter(backbone)
    classifier = LinearProbingClassifier(backbone, adapter, n_classes=3)
    model = PaPrPrunedClassifier(
        classifier=classifier,
        adapter=adapter,
        proposal=FixedProposal(),
        keep_ratio=0.5,
    )

    x = torch.randn(2, 3, 4, 4)
    logits = model(x)
    info = model(x, return_tokens=True)

    assert logits.shape == (2, 3)
    assert info["logits"].shape == (2, 3)
    assert info["features"].shape[1] == 1 + 2
    assert info["kept_indices"].tolist() == [[1, 3], [1, 3]]
