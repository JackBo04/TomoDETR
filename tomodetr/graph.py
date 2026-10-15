"""Build an explicit O(NK) cross-patch neighbor graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NeighborGraph:
    """Fixed-width sparse graph arrays."""

    neighbor_idx: np.ndarray
    neighbor_mask: np.ndarray
    relative_xyz: np.ndarray


def build_neighbor_graph(
    points_zyx: np.ndarray,
    patch_id: np.ndarray,
    radius: float,
    max_neighbors: int,
) -> NeighborGraph:
    """Connect self plus the nearest in-radius candidates from other patches.

    No dense N-by-N distance or attention mask is materialized.
    """
    points = np.asarray(points_zyx, dtype=np.float32).reshape(-1, 3)
    patches = np.asarray(patch_id, dtype=np.int64).reshape(-1)
    if len(points) != len(patches):
        raise ValueError("points_zyx and patch_id must have equal length")
    if radius <= 0.0 or max_neighbors <= 0:
        raise ValueError("radius and max_neighbors must be positive")
    count = len(points)
    indices = np.zeros((count, max_neighbors), dtype=np.int64)
    mask = np.zeros((count, max_neighbors), dtype=bool)
    relative = np.zeros((count, max_neighbors, 3), dtype=np.float32)
    if not count:
        return NeighborGraph(indices, mask, relative)
    tree = cKDTree(points)
    for row in range(count):
        candidates = tree.query_ball_point(points[row], float(radius), return_sorted=True)
        other = np.asarray(
            [value for value in candidates if value != row and patches[value] != patches[row]],
            dtype=np.int64,
        )
        if len(other):
            distance = np.linalg.norm(points[other] - points[row], axis=1)
            other = other[np.argsort(distance, kind="stable")]
        selected = np.concatenate((np.asarray([row], dtype=np.int64), other))[:max_neighbors]
        indices[row, : len(selected)] = selected
        mask[row, : len(selected)] = True
        relative[row, : len(selected)] = points[selected] - points[row]
    return NeighborGraph(indices, mask, relative)


def assert_sparse_graph(graph: NeighborGraph, node_count: int, max_neighbors: int) -> None:
    """Fail closed if a graph is not fixed-width O(NK)."""
    expected = (int(node_count), int(max_neighbors))
    if graph.neighbor_idx.shape != expected or graph.neighbor_mask.shape != expected:
        raise RuntimeError("Sparse graph index/mask shape mismatch")
    if graph.relative_xyz.shape != (*expected, 3):
        raise RuntimeError("Sparse graph relative geometry shape mismatch")
