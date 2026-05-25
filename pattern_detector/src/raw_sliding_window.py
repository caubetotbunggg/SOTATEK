"""Experimental raw/binary sliding-window template matching."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .nms import non_max_suppression
from .preprocessing import normalize_to_bgr, preprocess_image, preprocess_pattern
from .utils import ensure_parent_dir, round_float
from .visualization import draw_detections, save_visualization


@dataclass
class RawSlidingWindowConfig:
    threshold: float = 0.35
    min_scale: float = 0.50
    max_scale: float = 1.50
    scale_step: float = 0.10
    stride: int = 4
    top_k: int = 200
    nms_iou_threshold: float = 0.30
    max_detections: int = 200
    pattern_padding: int = 4
    enable_debug: bool = False
    debug_dir: str = "outputs/debug"


@dataclass
class RawSlidingWindowDetection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    scale: float
    rotation: float = 0.0
    method: str = "raw_sliding_window"
    ncc_score: float = 0.0
    foreground_iou: float = 0.0
    diff_similarity: float = 0.0

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        for key in ("confidence", "scale", "rotation", "ncc_score", "foreground_iou", "diff_similarity"):
            data[key] = round_float(float(data[key]))
        return data


def detect_raw_sliding_window(
    pattern_image: np.ndarray,
    drawing_image: np.ndarray,
    cfg: RawSlidingWindowConfig,
) -> tuple[list[RawSlidingWindowDetection], np.ndarray, dict[str, object]]:
    pattern_bgr = normalize_to_bgr(pattern_image)
    drawing_bgr = normalize_to_bgr(drawing_image)

    pattern_pre = preprocess_pattern(pattern_bgr, padding=cfg.pattern_padding).processed
    drawing_pre = preprocess_image(drawing_bgr, max_dim=None, compute_distance=False)
    pattern_binary = pattern_pre.binary
    drawing_binary = drawing_pre.binary

    all_candidates: list[RawSlidingWindowDetection] = []
    num_scales = 0
    for scale in _float_range(cfg.min_scale, cfg.max_scale, cfg.scale_step):
        template = cv2.resize(
            pattern_binary,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
        template = np.where(template > 0, 255, 0).astype(np.uint8)
        if template.shape[0] < 3 or template.shape[1] < 3:
            continue
        if template.shape[0] > drawing_binary.shape[0] or template.shape[1] > drawing_binary.shape[1]:
            continue
        if np.count_nonzero(template) < 3:
            continue
        num_scales += 1
        all_candidates.extend(_score_scale(drawing_binary, template, scale, cfg))

    above_threshold = [candidate for candidate in all_candidates if candidate.confidence >= cfg.threshold]
    suppressed = non_max_suppression(above_threshold, cfg.nms_iou_threshold)[: cfg.max_detections]
    visualization = draw_detections(drawing_bgr, suppressed)
    summary = {
        "method": "raw_sliding_window",
        "num_scales": num_scales,
        "num_raw_candidates": len(all_candidates),
        "num_above_threshold": len(above_threshold),
        "num_after_nms": len(suppressed),
        "top_score": round_float(max((candidate.confidence for candidate in all_candidates), default=0.0)),
    }

    if cfg.enable_debug:
        _save_debug(pattern_binary, drawing_binary, above_threshold, visualization, summary, cfg.debug_dir)

    return suppressed, visualization, summary


def _score_scale(
    drawing_binary: np.ndarray,
    template: np.ndarray,
    scale: float,
    cfg: RawSlidingWindowConfig,
) -> list[RawSlidingWindowDetection]:
    rough_scores = _rough_score_map(drawing_binary, template)
    sampled = rough_scores[0 : rough_scores.shape[0] : cfg.stride, 0 : rough_scores.shape[1] : cfg.stride]
    if sampled.size == 0:
        return []

    flat = sampled.ravel()
    keep = min(cfg.top_k, flat.size)
    if keep <= 0:
        return []
    indices = np.argpartition(-flat, keep - 1)[:keep]
    indices = indices[np.argsort(-flat[indices])]

    candidates: list[RawSlidingWindowDetection] = []
    for index in indices:
        sy, sx = np.unravel_index(int(index), sampled.shape)
        x = int(sx * cfg.stride)
        y = int(sy * cfg.stride)
        patch = drawing_binary[y : y + template.shape[0], x : x + template.shape[1]]
        ncc_score = _masked_ncc(template, patch)
        foreground_iou = _foreground_iou(template, patch)
        diff_similarity = _diff_similarity(template, patch)
        confidence = (0.50 * ncc_score) + (0.30 * foreground_iou) + (0.20 * diff_similarity)
        candidates.append(
            RawSlidingWindowDetection(
                x=x,
                y=y,
                w=template.shape[1],
                h=template.shape[0],
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                scale=float(scale),
                ncc_score=ncc_score,
                foreground_iou=foreground_iou,
                diff_similarity=diff_similarity,
            )
        )
    return candidates


def _rough_score_map(drawing_binary: np.ndarray, template: np.ndarray) -> np.ndarray:
    drawing_f = (drawing_binary > 0).astype(np.float32)
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


def _masked_ncc(template: np.ndarray, patch: np.ndarray) -> float:
    mask = cv2.dilate(np.where(template > 0, 255, 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    if int(mask.sum()) < 3:
        return 0.0
    t = (template > 0).astype(np.float32)[mask]
    p = (patch > 0).astype(np.float32)[mask]
    t = t - float(t.mean())
    p = p - float(p.mean())
    denom = float(np.linalg.norm(t) * np.linalg.norm(p))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(t, p) / denom, 0.0, 1.0))


def _foreground_iou(template: np.ndarray, patch: np.ndarray) -> float:
    template_mask = template > 0
    patch_mask = patch > 0
    union = float(np.logical_or(template_mask, patch_mask).sum())
    if union <= 0:
        return 0.0
    return float(np.logical_and(template_mask, patch_mask).sum() / union)


def _diff_similarity(template: np.ndarray, patch: np.ndarray) -> float:
    mask = cv2.dilate(np.where(template > 0, 255, 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    if int(mask.sum()) <= 0:
        return 0.0
    diff = np.abs(template.astype(np.float32) - patch.astype(np.float32))[mask] / 255.0
    return float(np.clip(1.0 - float(diff.mean()), 0.0, 1.0))


def _save_debug(
    pattern_binary: np.ndarray,
    drawing_binary: np.ndarray,
    before_nms: list[RawSlidingWindowDetection],
    final_visualization: np.ndarray,
    summary: dict[str, object],
    debug_dir: str,
) -> None:
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    save_visualization(cv2.cvtColor(pattern_binary, cv2.COLOR_GRAY2BGR), str(debug_path / "pattern_processed.png"))
    save_visualization(cv2.cvtColor(drawing_binary, cv2.COLOR_GRAY2BGR), str(debug_path / "drawing_processed.png"))
    save_visualization(draw_detections(cv2.cvtColor(drawing_binary, cv2.COLOR_GRAY2BGR), before_nms), str(debug_path / "candidates_before_nms.png"))
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
