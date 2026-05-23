"""Chamfer distance validation on skeleton edge maps."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ChamferScores:
    template_to_patch_distance: float
    patch_to_template_distance: float
    symmetric_distance: float
    similarity: float


def distance_transform_to_edges(edge_img: np.ndarray) -> np.ndarray:
    """Compute distance to nearest edge pixel using OpenCV distanceTransform semantics."""
    edge = np.where(edge_img > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(edge, cv2.DIST_L2, 3).astype(np.float32)


def one_way_chamfer_distance(
    source_edge: np.ndarray,
    target_edge: np.ndarray,
    *,
    percentile: float = 80.0,
) -> float:
    """Robust one-way Chamfer distance from source edges to target edges.

    Uses a high percentile instead of mean so missing structures are penalized harder.
    """
    source_mask = source_edge > 0
    if not np.any(source_mask):
        return float("inf")

    target_distance = distance_transform_to_edges(target_edge)
    distances = target_distance[source_mask]

    if distances.size == 0:
        return float("inf")

    return float(np.percentile(distances, percentile))


def template_to_patch_chamfer(template_edge: np.ndarray, patch_edge: np.ndarray) -> float:
    return one_way_chamfer_distance(template_edge, patch_edge, percentile=80.0)


def patch_to_template_chamfer(template_edge: np.ndarray, patch_edge: np.ndarray) -> float:
    return one_way_chamfer_distance(patch_edge, template_edge, percentile=80.0)


def symmetric_chamfer_scores(
    template_edge: np.ndarray,
    patch_edge: np.ndarray,
    *,
    sigma: float = 3.0,
) -> ChamferScores:
    """Return distances and similarity for symmetric Chamfer validation."""
    t2p = template_to_patch_chamfer(template_edge, patch_edge)
    p2t = patch_to_template_chamfer(template_edge, patch_edge)

    if not np.isfinite(t2p) or not np.isfinite(p2t):
        return ChamferScores(
            template_to_patch_distance=float("inf"),
            patch_to_template_distance=float("inf"),
            symmetric_distance=float("inf"),
            similarity=0.0,
        )

    symmetric_distance = max(t2p, p2t)
    similarity = float(np.exp(-symmetric_distance / max(sigma, 1e-6)))

    return ChamferScores(
        template_to_patch_distance=float(t2p),
        patch_to_template_distance=float(p2t),
        symmetric_distance=float(symmetric_distance),
        similarity=float(np.clip(similarity, 0.0, 1.0)),
    )


def symmetric_chamfer_similarity(
    template_edge: np.ndarray,
    patch_edge: np.ndarray,
    *,
    sigma: float = 3.0,
) -> float:
    """Backward-compatible wrapper."""
    return symmetric_chamfer_scores(template_edge, patch_edge, sigma=sigma).similarity