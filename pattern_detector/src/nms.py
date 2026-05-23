"""Non-maximum suppression for detection boxes."""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np


class BoxLike(Protocol):
    x: int
    y: int
    w: int
    h: int
    confidence: float


T = TypeVar("T", bound=BoxLike)


def bbox_iou(a: BoxLike, b: BoxLike) -> float:
    """Compute IoU between two xywh boxes."""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = max(0, a.w) * max(0, a.h)
    area_b = max(0, b.w) * max(0, b.h)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def non_max_suppression(candidates: list[T], iou_threshold: float) -> list[T]:
    """Greedy NMS over arbitrary objects exposing x/y/w/h/confidence."""
    if not candidates:
        return []

    order = np.argsort([-c.confidence for c in candidates])
    sorted_candidates = [candidates[int(i)] for i in order]
    kept: list[T] = []

    for candidate in sorted_candidates:
        if all(bbox_iou(candidate, kept_box) <= iou_threshold for kept_box in kept):
            kept.append(candidate)

    return kept
