"""Residual-logit sparse global query refiner."""

from __future__ import annotations

import torch

from .attention import SparseQueryAttention


class SparseTransformerBlock(torch.nn.Module):
    """Post-norm sparse attention and FFN block."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = SparseQueryAttention(
            d_model,
            heads,
            dropout,
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        relative_xyz: torch.Tensor,
        graph_radius: float,
    ) -> torch.Tensor:
        attended = self.attention(
            nodes, neighbor_idx, neighbor_mask, relative_xyz, graph_radius
        )
        nodes = self.norm1(nodes + self.dropout(attended))
        return self.norm2(nodes + self.dropout(self.ffn(nodes)))


def _position_mlp(input_dim: int, d_model: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, d_model),
        torch.nn.GELU(),
        torch.nn.Linear(d_model, d_model),
    )


class CWQR(torch.nn.Module):
    """Two sparse cross-window blocks and zero-initialized residual heads."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.query_projection = torch.nn.Linear(128, 128)
        self.global_position = _position_mlp(3, 128)
        self.local_position = _position_mlp(3, 128)
        self.boundary_position = _position_mlp(1, 128)
        self.blocks = torch.nn.ModuleList([
            SparseTransformerBlock(128, 8, 512, 0.1) for _ in range(2)
        ])
        self.class_delta = torch.nn.Linear(128, num_classes + 1)
        self.coordinate_delta = torch.nn.Linear(128, 3)
        for head in (self.class_delta, self.coordinate_delta):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def forward(self, query_feat, local_logits, xyz_global, xyz_local,
                boundary_dist, tomo_shape, patch_size, neighbor_idx,
                neighbor_mask, relative_xyz, graph_radius):
        global_position = 2 * xyz_global / (tomo_shape.float() - 1).clamp_min(1) - 1
        local_position = 2 * xyz_local / max(float(patch_size) - 1, 1) - 1
        boundary = (boundary_dist / (float(patch_size) / 2)).clamp(0, 1)
        nodes = (self.query_projection(query_feat)
                 + self.global_position(global_position)
                 + self.local_position(local_position))
        nodes = nodes + self.boundary_position(boundary[:, None])
        for block in self.blocks:
            nodes = block(nodes, neighbor_idx, neighbor_mask, relative_xyz, graph_radius)
        delta = 4.0 * torch.tanh(self.coordinate_delta(nodes))
        xyz = (xyz_global + delta).clamp(
            min=torch.zeros_like(tomo_shape).to(xyz_global),
            max=(tomo_shape.to(xyz_global) - 1).clamp_min(0),
        )
        return {"pred_logits": local_logits + self.class_delta(nodes),
                "pred_xyz": xyz, "coordinate_delta_voxels": delta}
