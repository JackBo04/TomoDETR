"""Gather-only sparse multi-head attention."""

from __future__ import annotations

import math

import torch


class SparseQueryAttention(torch.nn.Module):
    """Attend only to explicit K neighbors, with O(NK) tensors."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.head_dim = self.d_model // self.heads
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output = torch.nn.Linear(d_model, d_model)
        self.relative_bias = torch.nn.Sequential(
            torch.nn.Linear(4, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, heads),
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        relative_xyz: torch.Tensor,
        graph_radius: float,
    ) -> torch.Tensor:
        """Return sparse attention output without an N-by-N intermediate."""
        if nodes.ndim != 2 or neighbor_idx.ndim != 2:
            raise ValueError("nodes must be [N,D] and neighbor_idx must be [N,K]")
        if neighbor_idx.shape != neighbor_mask.shape:
            raise ValueError("neighbor_idx and neighbor_mask shapes differ")
        if relative_xyz.shape != (*neighbor_idx.shape, 3):
            raise ValueError("relative_xyz must be [N,K,3]")
        count, width = neighbor_idx.shape
        query = self.query(nodes).reshape(count, self.heads, self.head_dim)
        key = self.key(nodes).reshape(count, self.heads, self.head_dim)[neighbor_idx]
        value = self.value(nodes).reshape(count, self.heads, self.head_dim)[neighbor_idx]
        logits = (query[:, None] * key).sum(dim=-1) / math.sqrt(self.head_dim)
        normalized_delta = relative_xyz / max(float(graph_radius), 1e-6)
        distance = torch.linalg.vector_norm(normalized_delta, dim=-1, keepdim=True)
        relative_input = torch.cat(
            (
                normalized_delta,
                distance,
            ),
            dim=-1,
        )
        bias = self.relative_bias(relative_input)
        logits = logits + bias
        logits = logits.masked_fill(~neighbor_mask[..., None], torch.finfo(logits.dtype).min)
        weights = self.dropout(torch.softmax(logits, dim=1))
        attended = (weights[..., None] * value).sum(dim=1).reshape(count, self.d_model)
        return self.output(attended)
