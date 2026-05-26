"""Template Matching baseline inspired by technical-drawings-detection.

This module is intentionally isolated from the advanced SOTATEK detector.
It keeps only the zero-shot raw Template Matching path that is working well
for the current experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_parent_dir


def to_gray(img: np.ndarray) -> np.ndarray:
    """Return a uint8 grayscale image from BGR, BGRA, or gray input."""
    if img is None:
        raise ValueError("Image is None")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return img.copy()
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def enhance(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE enhancement before template matching."""
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)


def nms_xywh(boxes: list[tuple[int, int, int, int]], scores: list[float], iou_thr: float = 0.35) -> list[int]:
    if not boxes:
        return []

    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    areas = np.maximum(0.0, b[:, 2]) * np.maximum(0.0, b[:, 3])
    order = s.argsort()[::-1]
    keep: list[int] = []

    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-6
        order = rest[(inter / union) <= iou_thr]

    return keep


def smart_cliff(scores: list[float], decay: float = 0.52, min_gap: float = 0.28) -> int:
    """Return how many descending scores to keep before a sharp drop."""
    if not scores:
        return 0
    if len(scores) == 1:
        return 1

    prev = max(float(scores[0]), 1e-6)
    for idx, score in enumerate(scores[1:], start=1):
        current = float(score)
        if current < prev * decay or (prev - current) >= min_gap:
            return idx
        prev = max(current, 1e-6)
    return len(scores)


def find_best_scale_tm(
    base_gray: np.ndarray,
    ref_gray: np.ndarray,
    steps: int = 80,
    min_scale: float = 0.05,
    max_scale: float = 0.85,
) -> tuple[float, float, int, int]:
    """Sweep scales and return the best rough TM scale."""
    best_score = -1.0
    best_scale = min_scale
    best_w = 0
    best_h = 0

    for scale in np.linspace(min_scale, max_scale, max(1, int(steps))):
        ref_scaled = _resize_ref(ref_gray, float(scale))
        if ref_scaled is None:
            continue
        h, w = ref_scaled.shape[:2]
        if h > base_gray.shape[0] or w > base_gray.shape[1]:
            continue

        result = cv2.matchTemplate(base_gray, ref_scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if float(max_val) > best_score:
            best_score = float(max_val)
            best_scale = float(scale)
            best_w = int(w)
            best_h = int(h)

    return best_scale, best_score, best_w, best_h


def run_tm_baseline(
    drawing_bgr: np.ndarray,
    pattern_bgr: np.ndarray,
    wide_thr: float = 0.25,
    nms_iou: float = 0.35,
    top_k: int = 15,
    min_scale: float = 0.05,
    max_scale: float = 0.85,
    scan_scales: int = 30,
    use_smart_cliff: bool = True,
    enable_debug: bool = False,
    debug_dir: str = "outputs/debug",
) -> list[dict[str, float | int | str]]:
    """Run foreground-aware raw Template Matching over a scale window."""
    base_gray = enhance(to_gray(drawing_bgr))
    ref_gray = enhance(to_gray(pattern_bgr))
    best_scale, _, _, _ = find_best_scale_tm(base_gray, ref_gray, min_scale=min_scale, max_scale=max_scale)
    low = max(min_scale, best_scale * 0.80)
    high = min(max_scale, best_scale * 1.20)

    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    details: list[dict[str, float | int | str]] = []

    for scale in np.linspace(low, high, max(1, int(scan_scales))):
        ref_scaled = _resize_ref(ref_gray, float(scale))
        if ref_scaled is None:
            continue
        h, w = ref_scaled.shape[:2]
        if h > base_gray.shape[0] or w > base_gray.shape[1]:
            continue

        foreground_mask = cv2.dilate(_foreground_mask(ref_scaled), np.ones((3, 3), np.uint8), iterations=1)
        score_map = _masked_ncc_map(base_gray, ref_scaled, foreground_mask)

        for x, y, ncc_score in _local_maxima_candidates(score_map, w, h, wide_thr):
            patch = base_gray[y : y + h, x : x + w]
            if patch.shape != ref_scaled.shape:
                continue

            foreground_iou = _foreground_iou(ref_scaled, patch, foreground_mask)
            diff_similarity = _masked_diff_similarity(ref_scaled, patch, foreground_mask)
            ncc_score = float(np.clip(max(0.0, ncc_score), 0.0, 1.0))
            confidence = 0.50 * ncc_score + 0.30 * foreground_iou + 0.20 * diff_similarity

            boxes.append((int(x), int(y), int(w), int(h)))
            scores.append(float(confidence))
            details.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "confidence": float(confidence),
                    "scale": float(scale),
                    "rotation": 0.0,
                    "method": "zet_tm",
                    "tm_score": ncc_score,
                    "foreground_iou": foreground_iou,
                    "diff_similarity": diff_similarity,
                }
            )

    detections = _finish_detections(details, boxes, scores, nms_iou, top_k, use_smart_cliff)
    if enable_debug:
        _save_tm_debug(
            drawing_bgr,
            detections,
            debug_dir,
            scale_range=(min_scale, max_scale),
            wide_thr=wide_thr,
            top_k=top_k,
            nms_iou=nms_iou,
        )
    return detections


def draw_baseline_detections(drawing_bgr: np.ndarray, detections: list[dict[str, float | int | str]]) -> np.ndarray:
    vis = drawing_bgr.copy()
    for det in detections:
        x, y, w, h = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
        conf = float(det["confidence"])
        color = (0, 180, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"zet_tm:{conf:.3f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_y = max(0, y - th - baseline - 3)
        cv2.rectangle(vis, (x, label_y), (x + tw + 6, label_y + th + baseline + 4), color, -1)
        cv2.putText(vis, label, (x + 3, label_y + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
    return vis


def _resize_ref(ref_gray: np.ndarray, scale: float) -> np.ndarray | None:
    h, w = ref_gray.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w < 4 or new_h < 4:
        return None
    return cv2.resize(ref_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(otsu > 0)) > 0.5:
        otsu = cv2.bitwise_not(otsu)
    return np.where(otsu > 0, 255, 0).astype(np.uint8)


def _foreground_iou(template_gray: np.ndarray, patch_gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    template_fg = _foreground_mask(template_gray) > 0
    patch_fg = _foreground_mask(patch_gray) > 0
    if mask is not None and np.any(mask > 0):
        valid = mask > 0
        template_fg = np.logical_and(template_fg, valid)
        patch_fg = np.logical_and(patch_fg, valid)

    intersection = np.logical_and(template_fg, patch_fg).sum()
    union = np.logical_or(template_fg, patch_fg).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _masked_diff_similarity(template: np.ndarray, patch: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if not np.any(valid):
        valid = np.ones_like(template, dtype=bool)
    diff = np.mean(np.abs(template.astype(np.float32)[valid] - patch.astype(np.float32)[valid])) / 255.0
    return float(np.clip(1.0 - diff, 0.0, 1.0))


def _masked_ncc_map(base_gray: np.ndarray, ref_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if np.any(mask > 0):
        return cv2.matchTemplate(base_gray, ref_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
    return cv2.matchTemplate(base_gray, ref_gray, cv2.TM_CCOEFF_NORMED)


def _local_maxima_candidates(
    score_map: np.ndarray,
    template_w: int,
    template_h: int,
    score_thr: float,
) -> list[tuple[int, int, float]]:
    kx = max(3, int(template_w * 0.5))
    ky = max(3, int(template_h * 0.5))
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1

    dilated = cv2.dilate(score_map.astype(np.float32), np.ones((ky, kx), np.uint8), iterations=1)
    keep = np.logical_and(score_map >= score_thr, score_map >= dilated - 1e-6)
    ys, xs = np.where(keep)
    items = [(int(x), int(y), float(score_map[y, x])) for y, x in zip(ys, xs)]
    items.sort(key=lambda item: item[2], reverse=True)
    return items


def _finish_detections(
    details: list[dict[str, float | int | str]],
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    nms_iou: float,
    top_k: int,
    use_smart: bool,
) -> list[dict[str, float | int | str]]:
    keep = nms_xywh(boxes, scores, nms_iou)
    detections = [details[i] for i in keep]
    detections.sort(key=lambda det: float(det["confidence"]), reverse=True)
    detections = detections[: max(0, int(top_k))]
    if use_smart:
        detections = detections[: smart_cliff([float(det["confidence"]) for det in detections])]
    return detections


def _save_tm_debug(
    drawing_bgr: np.ndarray,
    detections: list[dict[str, float | int | str]],
    debug_dir: str,
    *,
    scale_range: tuple[float, float],
    wide_thr: float,
    top_k: int,
    nms_iou: float,
) -> None:
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    result_path = debug_path / "zet_tm_result.png"
    ensure_parent_dir(result_path)
    cv2.imwrite(str(result_path), draw_baseline_detections(drawing_bgr, detections))

    summary = {
        "method": "zet_tm",
        "num_detections": len(detections),
        "top_score": float(detections[0]["confidence"]) if detections else 0.0,
        "scale_range": [float(scale_range[0]), float(scale_range[1])],
        "wide_thr": float(wide_thr),
        "top_k": int(top_k),
        "nms_iou": float(nms_iou),
    }
    with (debug_path / "zet_tm_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
