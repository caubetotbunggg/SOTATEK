"""Edge-only feature extraction for zero-shot geometric visual search."""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import skeletonize_binary as _skeletonize_binary


def skeletonize_binary(binary_img: np.ndarray) -> np.ndarray:
    """Return a 0/255 skeleton image from a 0/255 or boolean binary image."""
    return _skeletonize_binary(binary_img)


def compute_edge_orientation(edge_img: np.ndarray) -> np.ndarray:
    """Compute unsigned local edge tangent orientation in radians in [0, pi)."""
    edge = np.where(edge_img > 0, 255, 0).astype(np.uint8)
    smooth = cv2.GaussianBlur(edge, (3, 3), 0)
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)

    # Sobel gives normal direction; add pi/2 to approximate edge tangent direction.
    orientation = np.mod(np.arctan2(grad_y, grad_x) + (np.pi / 2.0), np.pi)
    orientation[edge == 0] = 0.0
    return orientation.astype(np.float32)


def extract_edge_density_grid(edge_img: np.ndarray, grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Summarize where edges are located using an HxW density grid."""
    edge = (edge_img > 0).astype(np.float32)
    grid_h, grid_w = grid_size
    h, w = edge.shape[:2]
    features: list[float] = []

    for gy in range(grid_h):
        y1 = int(round(gy * h / grid_h))
        y2 = int(round((gy + 1) * h / grid_h))
        for gx in range(grid_w):
            x1 = int(round(gx * w / grid_w))
            x2 = int(round((gx + 1) * w / grid_w))
            cell = edge[y1:y2, x1:x2]
            features.append(float(cell.mean()) if cell.size else 0.0)

    return np.asarray(features, dtype=np.float32)


def extract_edge_hog_descriptor(
    edge_img: np.ndarray,
    cell_grid: tuple[int, int] = (8, 8),
    num_bins: int = 8,
) -> np.ndarray:
    """Compute a compact HOG-like descriptor from skeleton edge orientation only."""
    edge = edge_img > 0
    orientation = compute_edge_orientation(edge_img)
    grid_h, grid_w = cell_grid
    h, w = edge.shape[:2]
    features: list[float] = []

    for gy in range(grid_h):
        y1 = int(round(gy * h / grid_h))
        y2 = int(round((gy + 1) * h / grid_h))
        for gx in range(grid_w):
            x1 = int(round(gx * w / grid_w))
            x2 = int(round((gx + 1) * w / grid_w))
            cell_edge = edge[y1:y2, x1:x2]
            cell_orientation = orientation[y1:y2, x1:x2]
            hist = _orientation_histogram(cell_orientation[cell_edge], num_bins)
            features.extend(hist.tolist())

    return np.asarray(features, dtype=np.float32)


def extract_combined_edge_descriptor(edge_img: np.ndarray) -> np.ndarray:
    """Combine edge density and orientation distribution into one normalized vector."""
    density = extract_edge_density_grid(edge_img, grid_size=(8, 8))
    hog = extract_edge_hog_descriptor(edge_img, cell_grid=(8, 8), num_bins=8)
    global_density = np.asarray([float(np.mean(edge_img > 0))], dtype=np.float32)
    descriptor = np.concatenate([density, hog, global_density]).astype(np.float32)
    return l2_normalize(descriptor)


def l2_normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _orientation_histogram(orientations: np.ndarray, num_bins: int) -> np.ndarray:
    if orientations.size == 0:
        return np.zeros(num_bins, dtype=np.float32)
    bins = np.floor((orientations / np.pi) * num_bins).astype(np.int32)
    bins = np.clip(bins, 0, num_bins - 1)
    hist = np.bincount(bins, minlength=num_bins).astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else hist
