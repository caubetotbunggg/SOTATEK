"""Direct template matching branches over skeleton and binary foreground maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .chamfer import symmetric_chamfer_scores


@dataclass(frozen=True)
class DirectMatchCandidate:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float
    rotation: float
    branch: str
    edge_iou: float = 0.0
    chamfer_similarity: float = 0.0
    xor_similarity: float = 0.0
    density_score: float = 0.0
    ncc_score: float = 0.0

    @property
    def confidence(self) -> float:
        return self.score


def dilate_binary(img: np.ndarray, iterations: int) -> np.ndarray:
    binary = np.where(img > 0, 255, 0).astype(np.uint8)
    if iterations <= 0:
        return binary
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(binary, kernel, iterations=int(iterations))


def safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray | None:
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    if y + h > img.shape[0] or x + w > img.shape[1]:
        return None
    return img[y : y + h, x : x + w]


def compute_edge_iou(template: np.ndarray, patch: np.ndarray, dilation: int = 1) -> float:
    template_mask = dilate_binary(template, dilation) > 0
    patch_mask = dilate_binary(patch, dilation) > 0
    intersection = float(np.logical_and(template_mask, patch_mask).sum())
    union = float(np.logical_or(template_mask, patch_mask).sum())
    return float(intersection / union) if union > 0 else 0.0


def compute_xor_similarity(template: np.ndarray, patch: np.ndarray, mask: np.ndarray | None = None) -> float:
    template_mask = template > 0
    patch_mask = patch > 0
    compare_mask = np.logical_or(template_mask, patch_mask) if mask is None else mask > 0
    total = float(compare_mask.sum())
    if total <= 0:
        return 0.0
    xor_count = float(np.logical_xor(template_mask, patch_mask)[compare_mask].sum())
    return float(np.clip(1.0 - (xor_count / total), 0.0, 1.0))


def compute_density_score(template: np.ndarray, patch: np.ndarray) -> float:
    template_density = float((template > 0).sum()) / max(float(template.size), 1.0)
    patch_density = float((patch > 0).sum()) / max(float(patch.size), 1.0)
    eps = 1e-8
    ratio = patch_density / max(template_density, eps)
    return float(np.clip(min(ratio, 1.0 / max(ratio, eps)), 0.0, 1.0))


def compute_masked_ncc(template: np.ndarray, patch: np.ndarray, mask: np.ndarray | None = None) -> float:
    template_f = (template > 0).astype(np.float32)
    patch_f = (patch > 0).astype(np.float32)
    compare_mask = np.ones(template_f.shape, dtype=bool) if mask is None else mask > 0
    if int(compare_mask.sum()) < 3:
        return 0.0
    t = template_f[compare_mask]
    p = patch_f[compare_mask]
    t = t - float(t.mean())
    p = p - float(p.mean())
    denom = float(np.linalg.norm(t) * np.linalg.norm(p))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(t, p) / denom, 0.0, 1.0))


def get_topk_candidates(score_items: list[DirectMatchCandidate], top_k: int) -> list[DirectMatchCandidate]:
    if top_k <= 0:
        return []
    return sorted(score_items, key=lambda candidate: candidate.score, reverse=True)[:top_k]


def run_skeleton_direct_branch(
    drawing_skeleton: np.ndarray,
    template_variants: list[Any],
    cfg: Any,
) -> list[DirectMatchCandidate]:
    candidates: list[DirectMatchCandidate] = []
    for variant in template_variants:
        candidates.extend(_run_skeleton_variant(drawing_skeleton, variant, cfg))
    return candidates


def run_binary_direct_branch(
    drawing_binary: np.ndarray,
    template_variants: list[Any],
    cfg: Any,
) -> list[DirectMatchCandidate]:
    candidates: list[DirectMatchCandidate] = []
    for variant in template_variants:
        candidates.extend(_run_binary_variant(drawing_binary, variant, cfg))
    return candidates


def _run_skeleton_variant(drawing_skeleton: np.ndarray, variant: Any, cfg: Any) -> list[DirectMatchCandidate]:
    drawing = np.where(drawing_skeleton > 0, 255, 0).astype(np.uint8)
    template = np.where(variant.edge > 0, 255, 0).astype(np.uint8)
    rough = _rough_top_locations(drawing, template, int(cfg.direct_match_stride), int(cfg.direct_match_top_k))
    scored: list[DirectMatchCandidate] = []
    for x, y in rough:
        patch = safe_crop(drawing, x, y, template.shape[1], template.shape[0])
        if patch is None:
            continue
        edge_iou = compute_edge_iou(template, patch, dilation=int(cfg.direct_match_dilation))
        chamfer_similarity = symmetric_chamfer_scores(template, patch, sigma=cfg.chamfer_sigma).similarity
        xor_similarity = compute_xor_similarity(template, patch)
        density_score = compute_density_score(template, patch)
        score = (
            cfg.skeleton_direct_weight_iou * edge_iou
            + cfg.skeleton_direct_weight_chamfer * chamfer_similarity
            + cfg.skeleton_direct_weight_xor * xor_similarity
            + cfg.skeleton_direct_weight_density * density_score
        )
        scored.append(
            DirectMatchCandidate(
                x=x,
                y=y,
                w=template.shape[1],
                h=template.shape[0],
                score=float(np.clip(score, 0.0, 1.0)),
                scale=float(variant.scale),
                rotation=float(variant.rotation),
                branch="skeleton_direct",
                edge_iou=edge_iou,
                chamfer_similarity=chamfer_similarity,
                xor_similarity=xor_similarity,
                density_score=density_score,
            )
        )
    return get_topk_candidates(scored, int(cfg.direct_match_top_k))


def _run_binary_variant(drawing_binary: np.ndarray, variant: Any, cfg: Any) -> list[DirectMatchCandidate]:
    drawing = np.where(drawing_binary > 0, 255, 0).astype(np.uint8)
    template = np.where(variant.edge > 0, 255, 0).astype(np.uint8)
    rough = _rough_top_locations(drawing, template, int(cfg.direct_match_stride), int(cfg.direct_match_top_k))
    scored: list[DirectMatchCandidate] = []
    for x, y in rough:
        patch = safe_crop(drawing, x, y, template.shape[1], template.shape[0])
        if patch is None:
            continue
        match_template = dilate_binary(template, int(cfg.direct_match_dilation)) if cfg.binary_direct_use_edges else template
        match_patch = dilate_binary(patch, int(cfg.direct_match_dilation)) if cfg.binary_direct_use_edges else patch
        mask = dilate_binary(template, max(1, int(cfg.direct_match_dilation))) if cfg.binary_direct_use_masked_ncc else None
        ncc_score = compute_masked_ncc(match_template, match_patch, mask=mask)
        edge_iou = compute_edge_iou(match_template, match_patch, dilation=int(cfg.direct_match_dilation))
        xor_similarity = compute_xor_similarity(match_template, match_patch, mask=mask)
        density_score = compute_density_score(match_template, match_patch)
        score = (
            cfg.binary_direct_weight_ncc * ncc_score
            + cfg.binary_direct_weight_iou * edge_iou
            + cfg.binary_direct_weight_xor * xor_similarity
            + cfg.binary_direct_weight_density * density_score
        )
        scored.append(
            DirectMatchCandidate(
                x=x,
                y=y,
                w=template.shape[1],
                h=template.shape[0],
                score=float(np.clip(score, 0.0, 1.0)),
                scale=float(variant.scale),
                rotation=float(variant.rotation),
                branch="binary_direct",
                edge_iou=edge_iou,
                xor_similarity=xor_similarity,
                density_score=density_score,
                ncc_score=ncc_score,
            )
        )
    return get_topk_candidates(scored, int(cfg.direct_match_top_k))


def _rough_top_locations(drawing: np.ndarray, template: np.ndarray, stride: int, top_k: int) -> list[tuple[int, int]]:
    if template.shape[0] > drawing.shape[0] or template.shape[1] > drawing.shape[1]:
        return []
    if template.shape[0] < 3 or template.shape[1] < 3 or np.count_nonzero(template) < 3:
        return []
    stride = max(1, stride)
    drawing_f = (drawing > 0).astype(np.float32)
    template_f = (template > 0).astype(np.float32)
    overlap = cv2.matchTemplate(drawing_f, template_f, cv2.TM_CCORR)
    template_count = float(template_f.sum())
    window_count = cv2.matchTemplate(
        drawing_f,
        np.ones(template_f.shape, dtype=np.float32),
        cv2.TM_CCORR,
    )
    score_map = np.clip(overlap / np.sqrt(np.maximum(template_count * window_count, 1e-6)), 0.0, 1.0)
    sampled = score_map[0 : score_map.shape[0] : stride, 0 : score_map.shape[1] : stride]
    if sampled.size == 0:
        return []
    flat = sampled.ravel()
    keep = min(max(0, top_k), flat.size)
    if keep <= 0:
        return []
    indices = np.argpartition(-flat, keep - 1)[:keep]
    indices = indices[np.argsort(-flat[indices])]
    locations: list[tuple[int, int]] = []
    for index in indices:
        sy, sx = np.unravel_index(int(index), sampled.shape)
        locations.append((int(sx * stride), int(sy * stride)))
    return locations
