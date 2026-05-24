"""Candidate validation metrics for edge-feature search."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ValidationScores:
    edge_f1: float
    density_score: float
    template_coverage: float
    patch_coverage: float
    extra_patch_ratio: float
    passed: bool


def validate_edge_candidate(
    template_edge: np.ndarray,
    patch_edge: np.ndarray,
    *,
    min_template_coverage: float = 0.60,
    min_patch_coverage: float = 0.0,
    max_extra_patch_ratio: float = 0.55,
    dilation_iterations: int = 0,
    center_weight: np.ndarray | None = None,
) -> ValidationScores:
    """Validate shape match between template and patch.

    template_coverage:
        Fraction of template edge pixels explained by nearby patch edges.
        Low value means the candidate is missing important template structure.

    patch_coverage:
        Fraction of patch edge pixels explained by nearby template edges.
        Low value means the patch contains many unrelated/extra edges.

    extra_patch_ratio:
        1 - patch_coverage.
    """
    template = _ensure_same_size_binary(template_edge, patch_edge.shape[:2])
    patch = np.where(patch_edge > 0, 255, 0).astype(np.uint8)

    if dilation_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        template_match_area = cv2.dilate(template, kernel, iterations=dilation_iterations) > 0
        patch_match_area = cv2.dilate(patch, kernel, iterations=dilation_iterations) > 0
    else:
        template_match_area = template > 0
        patch_match_area = patch > 0

    template_mask = template > 0
    patch_mask = patch > 0
    weights = _ensure_weight_mask(center_weight, template.shape[:2])

    total_template_edges = float(weights[template_mask].sum())
    total_patch_edges = float(weights[patch_mask].sum())
    eps = 1e-8

    if total_template_edges <= 0 or total_patch_edges <= 0:
        return ValidationScores(
            edge_f1=0.0,
            density_score=0.0,
            template_coverage=0.0,
            patch_coverage=0.0,
            extra_patch_ratio=1.0,
            passed=False,
        )

    matched_template_edges = float(weights[np.logical_and(template_mask, patch_match_area)].sum())
    matched_patch_edges = float(weights[np.logical_and(patch_mask, template_match_area)].sum())

    template_coverage = matched_template_edges / (total_template_edges + eps)
    patch_coverage = matched_patch_edges / (total_patch_edges + eps)
    extra_patch_ratio = 1.0 - patch_coverage

    edge_f1 = (2.0 * template_coverage * patch_coverage) / (
        template_coverage + patch_coverage + eps
    )

    template_density = total_template_edges / max(float(template.size), 1.0)
    patch_density = total_patch_edges / max(float(patch.size), 1.0)
    ratio = patch_density / max(template_density, eps)
    density_score = min(ratio, 1.0 / max(ratio, eps))
    density_score = float(np.clip(density_score, 0.0, 1.0))

    passed = (
        template_coverage >= min_template_coverage
        and patch_coverage >= min_patch_coverage
        and extra_patch_ratio <= max_extra_patch_ratio
    )

    if not passed:
        edge_f1 *= 0.25
        density_score *= 0.50

    return ValidationScores(
        edge_f1=float(np.clip(edge_f1, 0.0, 1.0)),
        density_score=density_score,
        template_coverage=float(np.clip(template_coverage, 0.0, 1.0)),
        patch_coverage=float(np.clip(patch_coverage, 0.0, 1.0)),
        extra_patch_ratio=float(np.clip(extra_patch_ratio, 0.0, 1.0)),
        passed=passed,
    )


def _ensure_same_size_binary(edge_img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    edge = np.where(edge_img > 0, 255, 0).astype(np.uint8)
    target_h, target_w = target_shape
    if edge.shape[:2] == (target_h, target_w):
        return edge
    return cv2.resize(edge, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _ensure_weight_mask(weight_mask: np.ndarray | None, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    if weight_mask is None:
        return np.ones((target_h, target_w), dtype=np.float32)

    weights = np.asarray(weight_mask, dtype=np.float32)
    if weights.shape[:2] != (target_h, target_w):
        weights = cv2.resize(weights, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(weights, 0.0, None).astype(np.float32)
