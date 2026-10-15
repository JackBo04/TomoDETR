"""Voxel-center coordinates are always z,y,x."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import torch

def crop_local(points_zyx: np.ndarray, origin_zyx: Sequence[int], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Select GT in the closed voxel-center domain [0, size-1] on every axis.

    Image slices still contain ``size`` voxels. Continuous centers in
    (size-1, size) cannot be represented by the detector and become negative
    under reflection, so exclude them before spatial augmentation.
    """
    points = np.asarray(points_zyx, dtype=np.float32).reshape(-1, 3)
    origin = np.asarray(origin_zyx, dtype=np.float32)
    local = points - origin
    keep = np.all((local >= 0.0) & (local <= float(size - 1)), axis=1)
    return local[keep], keep

def rotate90_z(points_zyx: np.ndarray, size: int, k: int) -> np.ndarray:
    """Apply ``np.rot90(..., axes=(1,2))`` to z-y-x point coordinates."""
    result = np.asarray(points_zyx, dtype=np.float32).reshape(-1, 3).copy()
    for _ in range(int(k) % 4):
        z, y, x = result.T
        result = np.stack((z, float(size - 1) - x, y), axis=1)
    return result

def flip_points(points_zyx: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Flip z (0), y (1), or x (2) around voxel centers."""
    result = np.asarray(points_zyx, dtype=np.float32).reshape(-1, 3).copy()
    result[:, int(axis)] = float(size - 1) - result[:, int(axis)]
    return result

def normalized_zyx_to_grid_xyz(points_zyx: torch.Tensor) -> torch.Tensor:
    """Convert normalized [0,1] z-y-x to grid_sample [-1,1] x-y-z."""
    if points_zyx.shape[-1] != 3:
        raise ValueError("Expected final coordinate dimension z,y,x")
    return points_zyx[..., (2, 1, 0)] * 2.0 - 1.0

def inverse_sigmoid(values: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse sigmoid for iterative center refinement."""
    values = values.clamp(min=eps, max=1.0 - eps)
    return torch.log(values / (1.0 - values))

def bounded_sigmoid_refinement(
    centers: torch.Tensor,
    delta: torch.Tensor,
    low: float,
    high: float,
) -> torch.Tensor:
    """Refine normalized centers while keeping them strictly inside [low, high]."""
    span = float(high) - float(low)
    if not span > 0.0:
        raise ValueError("high must be greater than low")
    unit = (centers - float(low)) / span
    return float(low) + span * torch.sigmoid(inverse_sigmoid(unit) + delta)
