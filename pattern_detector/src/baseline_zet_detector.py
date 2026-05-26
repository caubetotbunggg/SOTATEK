"""Experimental baselines inspired by technical-drawings-detection.

These methods are intentionally isolated from the advanced SOTATEK detector.
They provide simple zero-shot Template Matching and HOG sliding-window
baselines for comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

try:
    from skimage.feature import hog as skimage_hog
except ImportError:  # Keep CLI/help usable until optional baseline dependency is installed.
    skimage_hog = None

from .utils import ensure_parent_dir


HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Return a uint8 grayscale image from BGR, BGRA, RGB-ish, or gray input."""
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
    """Apply the CLAHE enhancement used by the reference baseline."""
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)


def nms_xywh(boxes: list[tuple[int, int, int, int]], scores: list[float], iou_thr: float = 0.35) -> list[int]:
    if not boxes:
        return []

    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    areas = np.maximum(0, b[:, 2]) * np.maximum(0, b[:, 3])
    order = s.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou <= iou_thr]

    return keep


def smart_cliff(scores: list[float], decay: float = 0.52, min_gap: float = 0.28) -> int:
    """Return how many sorted scores to keep before a sharp confidence drop."""
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
    ratio = float(np.mean(otsu > 0))
    if ratio > 0.5:
        otsu = cv2.bitwise_not(otsu)
    return np.where(otsu > 0, 255, 0).astype(np.uint8)


def _masked_diff_similarity(template: np.ndarray, patch: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if not np.any(valid):
        valid = np.ones_like(template, dtype=bool)
    diff = np.mean(np.abs(template.astype(np.float32)[valid] - patch.astype(np.float32)[valid])) / 255.0
    return float(np.clip(1.0 - diff, 0.0, 1.0))


def _foreground_iou(template_gray: np.ndarray, patch_gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    t = _foreground_mask(template_gray) > 0
    p = _foreground_mask(patch_gray) > 0
    if mask is not None and np.any(mask > 0):
        valid = mask > 0
        t = np.logical_and(t, valid)
        p = np.logical_and(p, valid)
    inter = np.logical_and(t, p).sum()
    union = np.logical_or(t, p).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _masked_ncc_map(base_gray: np.ndarray, ref_gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None and np.any(mask > 0):
        return cv2.matchTemplate(base_gray, ref_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
    return cv2.matchTemplate(base_gray, ref_gray, cv2.TM_CCOEFF_NORMED)


def find_best_scale_tm(
    base_gray: np.ndarray,
    ref_gray: np.ndarray,
    steps: int = 80,
    min_scale: float = 0.05,
    max_scale: float = 0.85,
) -> tuple[float, float, int, int]:
    best = (-1.0, min_scale, 0, 0)
    for scale in np.linspace(min_scale, max_scale, max(1, int(steps))):
        ref_scaled = _resize_ref(ref_gray, float(scale))
        if ref_scaled is None:
            continue
        h, w = ref_scaled.shape[:2]
        if h > base_gray.shape[0] or w > base_gray.shape[1]:
            continue
        result = cv2.matchTemplate(base_gray, ref_scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if float(max_val) > best[0]:
            best = (float(max_val), float(scale), int(w), int(h))
    return best[1], best[0], best[2], best[3]


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
    mask = np.logical_and(score_map >= score_thr, score_map >= dilated - 1e-6)
    ys, xs = np.where(mask)
    items = [(int(x), int(y), float(score_map[y, x])) for y, x in zip(ys, xs)]
    items.sort(key=lambda item: item[2], reverse=True)
    return items


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

        mask = cv2.dilate(_foreground_mask(ref_scaled), np.ones((3, 3), np.uint8), iterations=1)
        score_map = _masked_ncc_map(base_gray, ref_scaled, mask)
        for x, y, ncc_score in _local_maxima_candidates(score_map, w, h, wide_thr):
            patch = base_gray[y : y + h, x : x + w]
            if patch.shape != ref_scaled.shape:
                continue
            fg_iou = _foreground_iou(ref_scaled, patch, mask)
            diff_similarity = _masked_diff_similarity(ref_scaled, patch, mask)
            ncc_score = float(np.clip(max(0.0, ncc_score), 0.0, 1.0))
            raw_score = 0.50 * ncc_score + 0.30 * fg_iou + 0.20 * diff_similarity
            boxes.append((int(x), int(y), int(w), int(h)))
            scores.append(float(raw_score))
            details.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "confidence": float(raw_score),
                    "scale": float(scale),
                    "rotation": 0.0,
                    "method": "zet_tm",
                    "tm_score": ncc_score,
                    "foreground_iou": fg_iou,
                    "diff_similarity": diff_similarity,
                }
            )

    detections = _finish_detections(details, boxes, scores, nms_iou, top_k, use_smart_cliff)
    if enable_debug:
        _save_baseline_debug(
            "zet_tm",
            drawing_bgr,
            detections,
            debug_dir,
            scale_range=(min_scale, max_scale),
            wide_thr=wide_thr,
            top_k=top_k,
            nms_iou=nms_iou,
        )
    return detections


def compute_hog(gray_img: np.ndarray) -> np.ndarray:
    if skimage_hog is None:
        raise ImportError("zet_hog requires scikit-image. Install with: pip install -r requirements.txt")
    return skimage_hog(gray_img, feature_vector=True, **HOG_PARAMS).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def find_best_scale_hog(
    base_gray: np.ndarray,
    ref_gray: np.ndarray,
    steps: int = 60,
    min_scale: float = 0.05,
    max_scale: float = 0.85,
) -> tuple[float, float, int, int]:
    best = (-1.0, min_scale, 0, 0)
    for scale in np.linspace(min_scale, max_scale, max(1, int(steps))):
        ref_scaled = _resize_ref(ref_gray, float(scale))
        if ref_scaled is None:
            continue
        h, w = ref_scaled.shape[:2]
        if h > base_gray.shape[0] or w > base_gray.shape[1] or h < 16 or w < 16:
            continue
        result = cv2.matchTemplate(base_gray, ref_scaled, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        x, y = max_loc
        patch = base_gray[y : y + h, x : x + w]
        try:
            sim = _cosine_similarity(compute_hog(ref_scaled), compute_hog(patch))
        except ValueError:
            continue
        if sim > best[0]:
            best = (float(sim), float(scale), int(w), int(h))
    return best[1], best[0], best[2], best[3]


def hog_sliding_window(
    base_gray: np.ndarray,
    ref_hog: np.ndarray,
    win_w: int,
    win_h: int,
    stride_ratio: float = 0.25,
    score_thr: float = 0.50,
) -> list[tuple[int, int, float]]:
    stride_x = max(4, int(win_w * stride_ratio))
    stride_y = max(4, int(win_h * stride_ratio))
    candidates: list[tuple[int, int, float]] = []
    max_y = base_gray.shape[0] - win_h
    max_x = base_gray.shape[1] - win_w
    if max_y < 0 or max_x < 0:
        return candidates

    for y in range(0, max_y + 1, stride_y):
        for x in range(0, max_x + 1, stride_x):
            patch = base_gray[y : y + win_h, x : x + win_w]
            try:
                patch_hog = compute_hog(patch)
            except ValueError:
                continue
            sim = _cosine_similarity(ref_hog, patch_hog)
            if sim >= score_thr:
                candidates.append((int(x), int(y), float(max(0.0, sim))))
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates


def run_hog_baseline(
    drawing_bgr: np.ndarray,
    pattern_bgr: np.ndarray,
    wide_thr: float = 0.50,
    nms_iou: float = 0.35,
    top_k: int = 30,
    stride_ratio: float = 0.25,
    min_scale: float = 0.05,
    max_scale: float = 0.85,
    n_scales: int = 8,
    use_smart_cliff: bool = True,
    enable_debug: bool = False,
    debug_dir: str = "outputs/debug",
) -> list[dict[str, float | int | str]]:
    base_gray = enhance(to_gray(drawing_bgr))
    ref_gray = enhance(to_gray(pattern_bgr))
    best_scale, _, _, _ = find_best_scale_hog(base_gray, ref_gray, min_scale=min_scale, max_scale=max_scale)
    low = max(min_scale, best_scale * 0.85)
    high = min(max_scale, best_scale * 1.15)

    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    details: list[dict[str, float | int | str]] = []

    for scale in np.linspace(low, high, max(1, int(n_scales))):
        ref_scaled = _resize_ref(ref_gray, float(scale))
        if ref_scaled is None:
            continue
        h, w = ref_scaled.shape[:2]
        if h > base_gray.shape[0] or w > base_gray.shape[1] or h < 16 or w < 16:
            continue
        try:
            ref_hog = compute_hog(ref_scaled)
        except ValueError:
            continue

        for x, y, hog_score in hog_sliding_window(base_gray, ref_hog, w, h, stride_ratio, wide_thr):
            boxes.append((int(x), int(y), int(w), int(h)))
            scores.append(float(hog_score))
            details.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "confidence": float(hog_score),
                    "scale": float(scale),
                    "rotation": 0.0,
                    "method": "zet_hog",
                    "hog_score": float(hog_score),
                }
            )

    detections = _finish_detections(details, boxes, scores, nms_iou, top_k, use_smart_cliff)
    if enable_debug:
        _save_baseline_debug(
            "zet_hog",
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
        method = str(det.get("method", "baseline"))
        color = (0, 180, 255) if method == "zet_tm" else (80, 220, 80)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{method}:{conf:.3f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_y = max(0, y - th - baseline - 3)
        cv2.rectangle(vis, (x, label_y), (x + tw + 6, label_y + th + baseline + 4), color, -1)
        cv2.putText(vis, label, (x + 3, label_y + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
    return vis


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
        count = smart_cliff([float(det["confidence"]) for det in detections])
        detections = detections[:count]
    return detections


def _save_baseline_debug(
    method: str,
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
    result_name = "zet_tm_result.png" if method == "zet_tm" else "zet_hog_result.png"
    result_path = debug_path / result_name
    ensure_parent_dir(result_path)
    cv2.imwrite(str(result_path), draw_baseline_detections(drawing_bgr, detections))

    summary = {
        "method": method,
        "num_detections": len(detections),
        "top_score": float(detections[0]["confidence"]) if detections else 0.0,
        "scale_range": [float(scale_range[0]), float(scale_range[1])],
        "wide_thr": float(wide_thr),
        "top_k": int(top_k),
        "nms_iou": float(nms_iou),
    }
    with (debug_path / "zet_baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
