"""Gated attention Multiple-Instance Learning aggregator (Ilse et al., 2018).

Aggregates a variable-length bag of tile embeddings into a slide-level
prediction via learned per-tile attention. Used by Phase 3 WSI evaluation.
"""

import torch
import torch.nn as nn


class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, n_classes: int = 2,
                 dropout: float = 0.25):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(), nn.Dropout(dropout),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Sigmoid(), nn.Dropout(dropout),
        )
        self.attention_w = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        """Args: bag (N_tiles, D). Returns logits (n_classes,)."""
        V = self.attention_V(bag)
        U = self.attention_U(bag)
        a = self.attention_w(V * U)
        a = a.softmax(dim=0)
        z = (a * bag).sum(dim=0)
        return self.classifier(z)
