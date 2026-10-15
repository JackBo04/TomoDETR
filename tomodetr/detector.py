"""Residual U-Net and encoder-free multi-scale deformable 3D DETR."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as functional

from .geometry import bounded_sigmoid_refinement, inverse_sigmoid, normalized_zyx_to_grid_xyz


def _groups(channels: int) -> int:
    for value in (16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock3D(torch.nn.Module):
    """Two-convolution residual block with small-batch GroupNorm."""

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv3d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = torch.nn.GroupNorm(_groups(output_channels), output_channels)
        self.conv2 = torch.nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False)
        self.norm2 = torch.nn.GroupNorm(_groups(output_channels), output_channels)
        self.skip = (
            torch.nn.Identity()
            if input_channels == output_channels and stride == 1
            else torch.nn.Conv3d(input_channels, output_channels, 1, stride=stride, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        hidden = functional.gelu(self.norm1(self.conv1(inputs)))
        return functional.gelu(self.norm2(self.conv2(hidden)) + residual)


class ResidualUNetBackbone3D(torch.nn.Module):
    """Explicit light U-Net returning decoder features at 1/2, 1/4, and 1/8."""

    def __init__(self, channels: tuple[int, int, int, int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.stem = ResidualBlock3D(1, c0)
        self.down1 = ResidualBlock3D(c0, c1, stride=2)
        self.down2 = ResidualBlock3D(c1, c2, stride=2)
        self.down3 = ResidualBlock3D(c2, c3, stride=2)
        self.up2 = torch.nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.decode2 = ResidualBlock3D(c2 + c2, c2)
        self.up1 = torch.nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.decode1 = ResidualBlock3D(c1 + c1, c1)

    def forward_with_stem(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the full-resolution stem plus the 1/2, 1/4, and 1/8 pyramid."""
        x0 = self.stem(inputs)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        d2 = self.decode2(torch.cat((self.up2(x3), x2), dim=1))
        d1 = self.decode1(torch.cat((self.up1(d2), x1), dim=1))
        return x0, [d1, d2, x3]

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        _, pyramid = self.forward_with_stem(inputs)
        return pyramid


class FourierPosition3D(torch.nn.Module):
    """Learned projection of fixed 3D Fourier features."""

    def __init__(self, d_model: int, bands: int = 16) -> None:
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(bands, dtype=torch.float32))
        self.projection = torch.nn.Linear(3 * (1 + 2 * bands), d_model)

    def forward(self, points_zyx: torch.Tensor) -> torch.Tensor:
        angles = points_zyx[..., None] * self.frequencies * (2.0 * math.pi)
        features = torch.cat((points_zyx[..., None], angles.sin(), angles.cos()), dim=-1)
        return self.projection(features.flatten(-2))


def sample_features(features: torch.Tensor, points_zyx: torch.Tensor) -> torch.Tensor:
    """Trilinearly sample [B,C,D,H,W] at normalized [B,Q,...,3] z-y-x points."""
    batch = features.shape[0]
    prefix = points_zyx.shape[1:-1]
    flat = points_zyx.reshape(batch, -1, 3)
    grid = normalized_zyx_to_grid_xyz(flat).reshape(batch, -1, 1, 1, 3)
    sampled = functional.grid_sample(
        features, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled.reshape(batch, features.shape[1], *prefix).movedim(1, -1)


class DeformableCrossAttention3D(torch.nn.Module):
    """Pure-PyTorch 3D deformable cross-attention over projected feature levels."""

    def __init__(self, d_model: int, heads: int, levels: int, points: int) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.levels = levels
        self.points = points
        self.head_dim = d_model // heads
        self.offsets = torch.nn.Linear(d_model, heads * levels * points * 3)
        self.weights = torch.nn.Linear(d_model, heads * levels * points)
        self.value = torch.nn.ModuleList([torch.nn.Conv3d(d_model, d_model, 1) for _ in range(levels)])
        self.output = torch.nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.offsets.weight)
        directions = []
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for head in range(self.heads):
            z = 1.0 - 2.0 * (head + 0.5) / self.heads
            radius = math.sqrt(max(0.0, 1.0 - z * z))
            direction = (z, radius * math.sin(head * golden), radius * math.cos(head * golden))
            for _level in range(self.levels):
                for point in range(self.points):
                    directions.extend(value * (point + 1) / self.points for value in direction)
        with torch.no_grad():
            self.offsets.bias.copy_(torch.tensor(directions))
        torch.nn.init.zeros_(self.weights.weight)
        torch.nn.init.zeros_(self.weights.bias)

    def forward(
        self,
        query: torch.Tensor,
        reference_zyx: torch.Tensor,
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(features) != self.levels:
            raise ValueError(f"Expected {self.levels} feature levels, got {len(features)}")
        batch, queries, _ = query.shape
        offsets = self.offsets(query).reshape(
            batch, queries, self.heads, self.levels, self.points, 3
        )
        # Offset scale is one feature cell at each level, expressed in normalized patch units.
        level_scales = query.new_tensor(
            [[1.0 / max(feature.shape[-3 + axis] - 1, 1) for axis in range(3)] for feature in features]
        )
        local_offsets = offsets * level_scales[None, None, None, :, None, :]
        locations = (
            reference_zyx[:, :, None, None, None, :] + local_offsets
        ).clamp(0.0, 1.0)
        attention = functional.softmax(
            self.weights(query).reshape(batch, queries, self.heads, self.levels * self.points), dim=-1
        ).reshape(batch, queries, self.heads, self.levels, self.points)
        accumulated = query.new_zeros(batch, queries, self.heads, self.head_dim)
        for level, feature in enumerate(features):
            sampled = sample_features(self.value[level](feature), locations[:, :, :, level])
            # sampled: B,Q,H_location,P,H_channel*d; select matching location/channel head.
            sampled = sampled.reshape(batch, queries, self.heads, self.points, self.heads, self.head_dim)
            sampled = sampled.diagonal(dim1=2, dim2=4).permute(0, 1, 4, 2, 3)
            accumulated = accumulated + (
                sampled * attention[:, :, :, level, :, None]
            ).sum(dim=3)
        output = self.output(accumulated.reshape(batch, queries, self.d_model))
        return output, {"sampling_locations_zyx": locations, "attention_weights": attention}


class DecoderLayer3D(torch.nn.Module):
    """Self-attention, deformable cross-attention, FFN, and layer-specific heads."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        levels: int,
        points: int,
        ffn_dim: int,
        dropout: float,
        classes: int,
        center_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.self_attention = torch.nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_attention = DeformableCrossAttention3D(d_model, heads, levels, points)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn_dim), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.class_head = torch.nn.Linear(d_model, classes + 1)
        self.center_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model), torch.nn.GELU(), torch.nn.Linear(d_model, 3)
        )
        torch.nn.init.zeros_(self.center_head[-1].weight)
        torch.nn.init.zeros_(self.center_head[-1].bias)
        self.center_bounds = tuple(float(value) for value in center_bounds)

    def forward(
        self,
        query: torch.Tensor,
        centers: torch.Tensor,
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        attended, _ = self.self_attention(
            query, query, query, need_weights=False,
            key_padding_mask=torch.zeros(query.shape[:2], dtype=torch.bool, device=query.device),
        )
        query = self.norm1(query + self.dropout(attended))
        crossed, audit = self.cross_attention(
            query, centers, features
        )
        query = self.norm2(query + self.dropout(crossed))
        query = self.norm3(query + self.dropout(self.ffn(query)))
        centers = bounded_sigmoid_refinement(
            centers, self.center_head(query), *self.center_bounds
        )
        return query, self.class_head(query), centers, audit


class LocalDetector(torch.nn.Module):
    """Paper detector: 64 uniform references, three scales, four decoder layers."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ResidualUNetBackbone3D((32, 64, 128, 256))
        self.projections = torch.nn.ModuleList([
            torch.nn.Conv3d(c, 128, 1) for c in (64, 128, 256)
        ])
        self.position = FourierPosition3D(128)
        axis = (torch.arange(4, dtype=torch.float32) + 0.5) / 4
        self.register_buffer("initial_centers", torch.stack(
            torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
        ).reshape(64, 3))
        self.layers = torch.nn.ModuleList([
            DecoderLayer3D(128, 8, 3, 4, 512, 0.1, num_classes)
            for _ in range(4)
        ])

    def forward(self, inputs: torch.Tensor) -> dict[str, Any]:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError("Expected [B,1,D,H,W]")
        if any(n < 8 or n % 8 for n in inputs.shape[-3:]):
            raise ValueError("Spatial sizes must be positive multiples of 8")
        features = [p(f) for p, f in zip(self.projections, self.backbone(inputs))]
        centers = self.initial_centers[None].expand(inputs.shape[0], -1, -1)
        query = self.position(centers) + sample_features(features[0], centers)
        outputs = []
        for layer in self.layers:
            query, logits, centers, _ = layer(query, centers, features)
            outputs.append({"class_logits": logits, "centers": centers, "queries": query})
        return {"layers": outputs}
