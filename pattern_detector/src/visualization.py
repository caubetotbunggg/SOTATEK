"""Visualization helpers for drawing detections."""

from __future__ import annotations

import cv2
import numpy as np

from .utils import ensure_parent_dir


def draw_detections(image_bgr: np.ndarray, detections: list[object]) -> np.ndarray:
    """Draw boxes and confidence labels on a copy of the original drawing."""
    vis = image_bgr.copy()

    for det in detections:
        x, y, w, h = int(det.x), int(det.y), int(det.w), int(det.h)
        conf = float(det.confidence)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 70, 255), 2)

        label = f"{conf:.3f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(0, y - th - baseline - 3)
        cv2.rectangle(vis, (x, label_y), (x + tw + 6, label_y + th + baseline + 4), (0, 70, 255), -1)
        cv2.putText(
            vis,
            label,
            (x + 3, label_y + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


def save_visualization(image_bgr: np.ndarray, path: str) -> None:
    ensure_parent_dir(path)
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise OSError(f"Could not write visualization image: {path}")
