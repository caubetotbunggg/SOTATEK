"""Sliding-window descriptor matching over skeletonized edge maps."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .edge_features import compute_edge_orientation, extract_combined_edge_descriptor, l2_normalize


@dataclass(frozen=True)
class EdgeTemplateVariant:
    edge: np.ndarray
    descriptor: np.ndarray
    scale: float
    rotation: float

    @property
    def h(self) -> int:
        return int(self.edge.shape[0])

    @property
    def w(self) -> int:
        return int(self.edge.shape[1])


@dataclass(frozen=True)
class DescriptorCandidate:
    x: int
    y: int
    w: int
    h: int
    descriptor_similarity: float
    scale: float
    rotation: float
    variant: EdgeTemplateVariant


@dataclass(frozen=True)
class DescriptorSearchDebug:
    heatmap: np.ndarray | None = None


def parse_rotations(rotations: str | list[float] | tuple[float, ...]) -> list[float]:
    """Parse comma-separated rotations while preserving order and removing duplicates."""
    if isinstance(rotations, str):
        raw_values = [value.strip() for value in rotations.split(",") if value.strip()]
        parsed = [float(value) for value in raw_values]
    else:
        parsed = [float(value) for value in rotations]
    if not parsed:
        parsed = [0.0, 90.0, 180.0, 270.0]

    unique: list[float] = []
    seen: set[float] = set()
    for value in parsed:
        normalized = float(value % 360.0)
        key = round(normalized, 6)
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def expand_rotations(
    base_rotations: list[float],
    *,
    fine_rotation_range: float = 0.0,
    fine_rotation_step: float = 5.0,
) -> list[float]:
    """Optionally add small offsets around each base rotation."""
    if fine_rotation_range <= 0:
        return parse_rotations(base_rotations)
    if fine_rotation_step <= 0:
        raise ValueError("fine_rotation_step must be positive")

    rotations: list[float] = []
    offset = -fine_rotation_range
    while offset <= fine_rotation_range + (fine_rotation_step * 0.5):
        for base in base_rotations:
            rotations.append((base + offset) % 360.0)
        offset += fine_rotation_step
    return parse_rotations(rotations)


def generate_edge_template_variants(
    pattern_edge: np.ndarray,
    *,
    min_scale: float,
    max_scale: float,
    scale_step: float,
    rotations: list[float],
    drawing_shape: tuple[int, int],
    min_template_size: int = 8,
) -> list[EdgeTemplateVariant]:
    """Generate transformed skeleton templates and their descriptors."""
    if scale_step <= 0:
        raise ValueError("scale_step must be positive")
    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("Scale values must be positive")
    if min_scale > max_scale:
        raise ValueError("min_scale must be <= max_scale")

    draw_h, draw_w = drawing_shape
    variants: list[EdgeTemplateVariant] = []
    seen: set[tuple[int, int, float, float]] = set()

    for scale in _float_range(min_scale, max_scale, scale_step):
        scaled = cv2.resize(
            np.where(pattern_edge > 0, 255, 0).astype(np.uint8),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
        for rotation in rotations:
            transformed = rotate_edge_image(scaled, rotation)
            if transformed.shape[0] < min_template_size or transformed.shape[1] < min_template_size:
                continue
            if transformed.shape[0] > draw_h or transformed.shape[1] > draw_w:
                continue
            if np.count_nonzero(transformed) < min_template_size:
                continue
            key = (transformed.shape[1], transformed.shape[0], round(scale, 4), round(rotation, 4))
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                EdgeTemplateVariant(
                    edge=transformed,
                    descriptor=extract_combined_edge_descriptor(transformed),
                    scale=float(scale),
                    rotation=float(rotation),
                )
            )

    return variants


def sliding_window_descriptor_search(
    drawing_edge: np.ndarray,
    variants: list[EdgeTemplateVariant],
    *,
    stride: int,
    top_k: int,
    heatmap_variant: tuple[float, float] | None = (1.0, 0.0),
) -> tuple[list[DescriptorCandidate], DescriptorSearchDebug]:
    """Search drawing windows using cosine similarity of edge descriptors."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    if top_k <= 0:
        return [], DescriptorSearchDebug()

    descriptor_index = _build_descriptor_integrals(drawing_edge)
    candidates: list[DescriptorCandidate] = []
    debug_heatmap: np.ndarray | None = None

    for variant in variants:
        variant_candidates, heatmap = _search_one_variant(
            descriptor_index,
            drawing_edge.shape[:2],
            variant,
            stride=stride,
            top_k=top_k,
        )
        candidates.extend(variant_candidates)
        if heatmap_variant and _same_variant(variant, heatmap_variant):
            debug_heatmap = heatmap

    return candidates, DescriptorSearchDebug(heatmap=debug_heatmap)


def rotate_edge_image(edge_img: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate a 0/255 edge image without clipping and crop empty margins."""
    edge = np.where(edge_img > 0, 255, 0).astype(np.uint8)
    normalized = angle_degrees % 360.0
    if abs(normalized) < 1e-9:
        return _crop_nonzero(edge)

    h, w = edge.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = max(1, int((h * sin) + (w * cos)))
    new_h = max(1, int((h * cos) + (w * sin)))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(edge, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return _crop_nonzero(np.where(rotated > 0, 255, 0).astype(np.uint8))


def _build_descriptor_integrals(edge_img: np.ndarray, num_bins: int = 8) -> dict[str, object]:
    edge = (edge_img > 0).astype(np.float32)
    orientation = compute_edge_orientation(edge_img)
    bin_maps = []
    for bin_idx in range(num_bins):
        lower = (bin_idx / num_bins) * np.pi
        upper = ((bin_idx + 1) / num_bins) * np.pi
        if bin_idx == num_bins - 1:
            mask = (edge > 0) & (orientation >= lower) & (orientation <= upper)
        else:
            mask = (edge > 0) & (orientation >= lower) & (orientation < upper)
        bin_maps.append(mask.astype(np.float32))

    return {
        "edge": edge,
        "edge_integral": cv2.integral(edge, sdepth=cv2.CV_64F),
        "bin_maps": bin_maps,
        "bin_integrals": [cv2.integral(m, sdepth=cv2.CV_64F) for m in bin_maps],
        "num_bins": num_bins,
    }


def _search_one_variant(
    descriptor_index: dict[str, object],
    drawing_shape: tuple[int, int],
    variant: EdgeTemplateVariant,
    *,
    stride: int,
    top_k: int,
) -> tuple[list[DescriptorCandidate], np.ndarray]:
    draw_h, draw_w = drawing_shape
    max_y = draw_h - variant.h
    max_x = draw_w - variant.w
    if max_y < 0 or max_x < 0:
        return [], np.zeros((0, 0), dtype=np.float32)

    score_map = _orientation_score_map(descriptor_index, variant)
    sampled = score_map[0 : max_y + 1 : stride, 0 : max_x + 1 : stride]
    if sampled.size == 0:
        return [], sampled.astype(np.float32)

    flat = sampled.ravel()
    keep = min(top_k, flat.size)
    if keep <= 0:
        return [], sampled.astype(np.float32)
    indices = np.argpartition(-flat, keep - 1)[:keep]
    indices = indices[np.argsort(-flat[indices])]

    candidates: list[DescriptorCandidate] = []
    for index in indices:
        yi, xi = np.unravel_index(int(index), sampled.shape)
        score = float(sampled[yi, xi])
        x = int(xi * stride)
        y = int(yi * stride)
        candidates.append(
            DescriptorCandidate(
            x=int(x),
            y=int(y),
            w=variant.w,
            h=variant.h,
            descriptor_similarity=float(score),
            scale=variant.scale,
            rotation=variant.rotation,
            variant=variant,
        )
        )
    return candidates, sampled.astype(np.float32)


def _orientation_score_map(descriptor_index: dict[str, object], variant: EdgeTemplateVariant) -> np.ndarray:
    """Fast cosine-like score map over orientation-bin edge channels."""
    drawing_bin_maps = descriptor_index["bin_maps"]
    drawing_edge = descriptor_index["edge"]
    num_bins = int(descriptor_index["num_bins"])
    template_bins = _edge_orientation_bin_maps(variant.edge, num_bins)

    score_sum: np.ndarray | None = None
    template_edge_count = 0.0
    for drawing_bin, template_bin in zip(drawing_bin_maps, template_bins):
        bin_count = float(template_bin.sum())
        template_edge_count += bin_count
        if bin_count <= 0:
            continue
        channel_score = cv2.matchTemplate(drawing_bin.astype(np.float32), template_bin.astype(np.float32), cv2.TM_CCORR)
        score_sum = channel_score if score_sum is None else score_sum + channel_score

    if score_sum is None or template_edge_count <= 0:
        return np.zeros(
            (drawing_edge.shape[0] - variant.h + 1, drawing_edge.shape[1] - variant.w + 1),
            dtype=np.float32,
        )

    template_edge = (variant.edge > 0).astype(np.float32)
    window_edge_count = cv2.matchTemplate(
        drawing_edge.astype(np.float32),
        np.ones((variant.h, variant.w), dtype=np.float32),
        cv2.TM_CCORR,
    )
    denom = np.sqrt(np.maximum(window_edge_count * template_edge_count, 1e-6))
    orientation_score = score_sum / denom

    edge_overlap = cv2.matchTemplate(drawing_edge.astype(np.float32), template_edge, cv2.TM_CCORR)
    edge_score = edge_overlap / denom
    """score = (0.65 * orientation_score) + (0.35 * edge_score)
    score = (0.35 * orientation_score) + (0.65 * edge_score)"""
    score = (0.20 * orientation_score) + (0.80 * edge_score)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _edge_orientation_bin_maps(edge_img: np.ndarray, num_bins: int) -> list[np.ndarray]:
    edge = edge_img > 0
    orientation = compute_edge_orientation(edge_img)
    bin_maps: list[np.ndarray] = []
    for bin_idx in range(num_bins):
        lower = (bin_idx / num_bins) * np.pi
        upper = ((bin_idx + 1) / num_bins) * np.pi
        if bin_idx == num_bins - 1:
            mask = edge & (orientation >= lower) & (orientation <= upper)
        else:
            mask = edge & (orientation >= lower) & (orientation < upper)
        bin_maps.append(mask.astype(np.float32))
    return bin_maps


def _window_descriptor(
    descriptor_index: dict[str, object],
    x: int,
    y: int,
    w: int,
    h: int,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    edge_integral = descriptor_index["edge_integral"]
    bin_integrals = descriptor_index["bin_integrals"]
    num_bins = int(descriptor_index["num_bins"])
    grid_h, grid_w = grid_size
    window_area = max(float(w * h), 1.0)

    density_features: list[float] = []
    hog_features: list[float] = []
    total_edges = _rect_sum(edge_integral, x, y, x + w, y + h)

    for gy in range(grid_h):
        y1 = y + int(round(gy * h / grid_h))
        y2 = y + int(round((gy + 1) * h / grid_h))
        for gx in range(grid_w):
            x1 = x + int(round(gx * w / grid_w))
            x2 = x + int(round((gx + 1) * w / grid_w))
            cell_area = max(float((x2 - x1) * (y2 - y1)), 1.0)
            cell_edges = _rect_sum(edge_integral, x1, y1, x2, y2)
            density_features.append(cell_edges / cell_area)

            hist = np.asarray([_rect_sum(integral, x1, y1, x2, y2) for integral in bin_integrals], dtype=np.float32)
            hist_sum = float(hist.sum())
            if hist_sum > 0:
                hist /= hist_sum
            hog_features.extend(hist.tolist())

    descriptor = np.concatenate(
        [
            np.asarray(density_features, dtype=np.float32),
            np.asarray(hog_features, dtype=np.float32),
            np.asarray([total_edges / window_area], dtype=np.float32),
        ]
    )
    return l2_normalize(descriptor)


def _rect_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def _same_variant(variant: EdgeTemplateVariant, target: tuple[float, float]) -> bool:
    scale, rotation = target
    return abs(variant.scale - scale) < 1e-6 and abs((variant.rotation - rotation) % 360.0) < 1e-6


def _crop_nonzero(edge: np.ndarray) -> np.ndarray:
    ys, xs = np.where(edge > 0)
    if len(xs) == 0 or len(ys) == 0:
        return edge
    return edge[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]


def _float_range(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    value = float(start)
    while value <= stop + (step * 0.5):
        values.append(round(value, 6))
        value += step
    return values
