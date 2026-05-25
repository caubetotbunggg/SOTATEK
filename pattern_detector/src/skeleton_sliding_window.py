"""Experimental skeleton sliding-window template matching."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .chamfer import symmetric_chamfer_scores
from .nms import non_max_suppression
from .preprocessing import normalize_to_bgr, preprocess_image, preprocess_pattern
from .utils import ensure_parent_dir, round_float
from .visualization import draw_detections, save_visualization


@dataclass
class SkeletonSlidingWindowConfig:
    threshold: float = 0.35
    min_scale: float = 0.50
    max_scale: float = 1.50
    scale_step: float = 0.10
    stride: int = 4
    top_k: int = 200
    nms_iou_threshold: float = 0.30
    chamfer_sigma: float = 8.0
    dilation: int = 1
    max_detections: int = 200
    pattern_padding: int = 4
    enable_debug: bool = False
    debug_dir: str = "outputs/debug"


@dataclass
class SkeletonSlidingWindowDetection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    scale: float
    rotation: float = 0.0
    method: str = "skeleton_sliding_window"
    skeleton_iou: float = 0.0
    chamfer_similarity: float = 0.0
    xor_similarity: float = 0.0
    chamfer_distance: float = 0.0

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        for key in ("confidence", "scale", "rotation", "skeleton_iou", "chamfer_similarity", "xor_similarity", "chamfer_distance"):
            data[key] = round_float(float(data[key]))
        return data


def detect_skeleton_sliding_window(
    pattern_image: np.ndarray,
    drawing_image: np.ndarray,
    cfg: SkeletonSlidingWindowConfig,
) -> tuple[list[SkeletonSlidingWindowDetection], np.ndarray, dict[str, object]]:
    pattern_bgr = normalize_to_bgr(pattern_image)
    drawing_bgr = normalize_to_bgr(drawing_image)

    pattern_pre = preprocess_pattern(pattern_bgr, padding=cfg.pattern_padding).processed
    drawing_pre = preprocess_image(drawing_bgr, max_dim=None, compute_distance=False)
    pattern_skeleton = pattern_pre.skeleton
    drawing_skeleton = drawing_pre.skeleton

    all_candidates: list[SkeletonSlidingWindowDetection] = []
    num_scales = 0
    for scale in _float_range(cfg.min_scale, cfg.max_scale, cfg.scale_step):
        template = cv2.resize(
            pattern_skeleton,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
        template = np.where(template > 0, 255, 0).astype(np.uint8)
        if template.shape[0] < 3 or template.shape[1] < 3:
            continue
        if template.shape[0] > drawing_skeleton.shape[0] or template.shape[1] > drawing_skeleton.shape[1]:
            continue
        if np.count_nonzero(template) < 3:
            continue
        num_scales += 1
        all_candidates.extend(_score_scale(drawing_skeleton, template, scale, cfg))

    above_threshold = [candidate for candidate in all_candidates if candidate.confidence >= cfg.threshold]
    suppressed = non_max_suppression(above_threshold, cfg.nms_iou_threshold)[: cfg.max_detections]
    visualization = draw_detections(drawing_bgr, suppressed)
    summary = {
        "method": "skeleton_sliding_window",
        "num_scales": num_scales,
        "num_raw_candidates": len(all_candidates),
        "num_above_threshold": len(above_threshold),
        "num_after_nms": len(suppressed),
        "top_score": round_float(max((candidate.confidence for candidate in all_candidates), default=0.0)),
    }

    if cfg.enable_debug:
        _save_debug(pattern_skeleton, drawing_skeleton, above_threshold, visualization, summary, cfg.debug_dir)

    return suppressed, visualization, summary


def _score_scale(
    drawing_skeleton: np.ndarray,
    template: np.ndarray,
    scale: float,
    cfg: SkeletonSlidingWindowConfig,
) -> list[SkeletonSlidingWindowDetection]:
    rough_scores = _rough_score_map(drawing_skeleton, template)
    sampled = rough_scores[0 : rough_scores.shape[0] : cfg.stride, 0 : rough_scores.shape[1] : cfg.stride]
    if sampled.size == 0:
        return []

    flat = sampled.ravel()
    keep = min(cfg.top_k, flat.size)
    if keep <= 0:
        return []
    indices = np.argpartition(-flat, keep - 1)[:keep]
    indices = indices[np.argsort(-flat[indices])]

    candidates: list[SkeletonSlidingWindowDetection] = []
    for index in indices:
        sy, sx = np.unravel_index(int(index), sampled.shape)
        x = int(sx * cfg.stride)
        y = int(sy * cfg.stride)
        patch = drawing_skeleton[y : y + template.shape[0], x : x + template.shape[1]]
        skeleton_iou = _skeleton_iou(template, patch, dilation=cfg.dilation)
        chamfer = symmetric_chamfer_scores(template, patch, sigma=cfg.chamfer_sigma)
        xor_similarity = _xor_similarity(template, patch, dilation=cfg.dilation)
        confidence = (0.45 * skeleton_iou) + (0.35 * chamfer.similarity) + (0.20 * xor_similarity)
        candidates.append(
            SkeletonSlidingWindowDetection(
                x=x,
                y=y,
                w=template.shape[1],
                h=template.shape[0],
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                scale=float(scale),
                skeleton_iou=skeleton_iou,
                chamfer_similarity=chamfer.similarity,
                xor_similarity=xor_similarity,
                chamfer_distance=chamfer.symmetric_distance,
            )
        )
    return candidates


def _rough_score_map(drawing_skeleton: np.ndarray, template: np.ndarray) -> np.ndarray:
    drawing_f = (drawing_skeleton > 0).astype(np.float32)
    template_f = (template > 0).astype(np.float32)
    overlap = cv2.matchTemplate(drawing_f, template_f, cv2.TM_CCORR)
    template_count = float(template_f.sum())
    window_count = cv2.matchTemplate(
        drawing_f,
        np.ones(template_f.shape, dtype=np.float32),
        cv2.TM_CCORR,
    )
    denom = np.sqrt(np.maximum(template_count * window_count, 1e-6))
    return np.clip(overlap / denom, 0.0, 1.0).astype(np.float32)


def _skeleton_iou(template: np.ndarray, patch: np.ndarray, dilation: int) -> float:
    template_mask = _dilate(template, dilation) > 0
    patch_mask = _dilate(patch, dilation) > 0
    union = float(np.logical_or(template_mask, patch_mask).sum())
    if union <= 0:
        return 0.0
    return float(np.logical_and(template_mask, patch_mask).sum() / union)


def _xor_similarity(template: np.ndarray, patch: np.ndarray, dilation: int) -> float:
    template_mask = _dilate(template, dilation) > 0
    patch_mask = _dilate(patch, dilation) > 0
    compare_mask = np.logical_or(template_mask, patch_mask)
    total = float(compare_mask.sum())
    if total <= 0:
        return 0.0
    xor_count = float(np.logical_xor(template_mask, patch_mask)[compare_mask].sum())
    return float(np.clip(1.0 - (xor_count / total), 0.0, 1.0))


def _dilate(img: np.ndarray, iterations: int) -> np.ndarray:
    binary = np.where(img > 0, 255, 0).astype(np.uint8)
    if iterations <= 0:
        return binary
    return cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=int(iterations))


def _save_debug(
    pattern_skeleton: np.ndarray,
    drawing_skeleton: np.ndarray,
    before_nms: list[SkeletonSlidingWindowDetection],
    final_visualization: np.ndarray,
    summary: dict[str, object],
    debug_dir: str,
) -> None:
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    save_visualization(cv2.cvtColor(pattern_skeleton, cv2.COLOR_GRAY2BGR), str(debug_path / "pattern_processed.png"))
    save_visualization(cv2.cvtColor(drawing_skeleton, cv2.COLOR_GRAY2BGR), str(debug_path / "drawing_processed.png"))
    save_visualization(draw_detections(cv2.cvtColor(drawing_skeleton, cv2.COLOR_GRAY2BGR), before_nms), str(debug_path / "candidates_before_nms.png"))
    save_visualization(final_visualization, str(debug_path / "final_result.png"))
    ensure_parent_dir(str(debug_path / "score_summary.json"))
    with open(debug_path / "score_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _float_range(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    value = float(start)
    while value <= stop + (step * 0.5):
        values.append(round(value, 6))
        value += step
    return values
