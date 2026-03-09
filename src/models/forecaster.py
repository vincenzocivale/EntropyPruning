import torch
import torch.nn as nn


class AttentionForecaster(nn.Module):
    def __init__(self, embed_dim=1024, hidden=256,
                 n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden)
        self.cls_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

        self.self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=dropout, batch_first=True,
                norm_first=True,
            ) for _ in range(n_layers)
        ])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden, n_heads, dropout=dropout,
                                  batch_first=True)
            for _ in range(n_layers)
        ])
        self.cross_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(hidden)
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, patch_embeddings):
        B, N, D = patch_embeddings.shape
        x = self.input_proj(patch_embeddings)

        for sa in self.self_attn:
            x = sa(x)

        cls = self.cls_query.expand(B, -1, -1)
        for ca, norm in zip(self.cross_attn, self.cross_norms):
            cls_out, _ = ca(cls, x, x)
            cls = norm(cls + cls_out)

        x_norm = self.norm(x)
        cls_exp = cls.expand(-1, N, -1)
        scores = self.score_head(
            torch.cat([x_norm, cls_exp], dim=-1)
        ).squeeze(-1)
        return scores
