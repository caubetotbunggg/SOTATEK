"""Image loading and binarization helpers for black-and-white drawings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_parent_dir


@dataclass(frozen=True)
class ProcessedImage:
    original_bgr: np.ndarray
    gray: np.ndarray
    binary: np.ndarray
    edge: np.ndarray
    skeleton: np.ndarray
    dilated_edge: np.ndarray
    distance_transform: np.ndarray | None
    scale_to_original: float


@dataclass(frozen=True)
class CroppedPattern:
    processed: ProcessedImage
    crop_offset: tuple[int, int]


def load_image(path: str | Path) -> np.ndarray:
    """Load color, grayscale, or binary image as BGR for consistent downstream IO."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return normalize_to_bgr(image)


def normalize_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert common OpenCV/PIL image layouts to uint8 BGR."""
    if image is None:
        raise ValueError("Image is None")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()

    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_for_processing(image_bgr: np.ndarray, max_dim: int | None) -> tuple[np.ndarray, float]:
    """Resize large drawings while preserving a multiplier back to original coordinates."""
    if not max_dim or max_dim <= 0:
        return image_bgr, 1.0

    h, w = image_bgr.shape[:2]
    current_max = max(h, w)
    if current_max <= max_dim:
        return image_bgr, 1.0

    process_scale = max_dim / float(current_max)
    resized = cv2.resize(
        image_bgr,
        (max(1, int(round(w * process_scale))), max(1, int(round(h * process_scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, 1.0 / process_scale


def preprocess_image(
    image_bgr: np.ndarray,
    *,
    max_dim: int | None = None,
    compute_distance: bool = False,
) -> ProcessedImage:
    """Create grayscale, binary, edge, dilated-edge, and optional distance-transform maps."""
    original_bgr = normalize_to_bgr(image_bgr)
    work_bgr, scale_to_original = resize_for_processing(original_bgr, max_dim)

    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)

    binary = _binarize_foreground_white(gray)
    skeleton = skeletonize_binary(binary)

    edge = skeleton.copy()
    kernel = np.ones((3, 3), np.uint8)
    dilated_edge = cv2.dilate((edge > 0).astype(np.uint8) * 255, kernel, iterations=1).astype(np.uint8)

    distance_transform = None
    if compute_distance:
        # distanceTransform expects non-zero pixels as free space. Edges become zeros.
        edge_inverse = np.where(edge > 0, 0, 255).astype(np.uint8)
        distance_transform = cv2.distanceTransform(edge_inverse, cv2.DIST_L2, 3).astype(np.float32)

    return ProcessedImage(
        original_bgr=original_bgr,
        gray=gray,
        binary=binary,
        edge=edge,
        skeleton=skeleton,
        dilated_edge=dilated_edge,
        distance_transform=distance_transform,
        scale_to_original=scale_to_original,
    )


def preprocess_pattern(image_bgr: np.ndarray, *, padding: int = 4) -> CroppedPattern:
    """Preprocess and tightly crop the query pattern around foreground strokes."""
    processed = preprocess_image(image_bgr, max_dim=None, compute_distance=False)
    x, y, w, h = foreground_bbox(processed.binary, padding=padding)
    if w == 0 or h == 0:
        raise ValueError("Pattern image has no detectable foreground pixels")

    cropped_bgr = processed.original_bgr[y : y + h, x : x + w]
    cropped = preprocess_image(cropped_bgr, max_dim=None, compute_distance=False)
    return CroppedPattern(processed=cropped, crop_offset=(x, y))


def foreground_bbox(binary: np.ndarray, padding: int = 4) -> tuple[int, int, int, int]:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0

    h, w = binary.shape[:2]
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(w, int(xs.max()) + padding + 1)
    y2 = min(h, int(ys.max()) + padding + 1)
    return x1, y1, x2 - x1, y2 - y1


def _binarize_foreground_white(gray: np.ndarray) -> np.ndarray:
    """Threshold image and choose polarity so sparse strokes are foreground white."""
    if gray.size == 0:
        raise ValueError("Cannot preprocess an empty image")

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    # Technical drawings usually have dark strokes on light background. Pick the polarity
    # that makes foreground sparse rather than filling most of the canvas.
    combined = cv2.bitwise_and(otsu, adaptive)
    white_ratio = float(np.mean(combined > 0))
    if white_ratio > 0.5:
        combined = cv2.bitwise_not(combined)

    binary = np.where(combined > 0, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return binary.astype(np.uint8)


def skeletonize_binary(binary_img: np.ndarray) -> np.ndarray:
    """Normalize stroke thickness to 1-pixel-ish skeleton, preserving 0/255 convention."""
    binary = np.where(binary_img > 0, 255, 0).astype(np.uint8)
    try:
        from skimage.morphology import skeletonize as sk_skeletonize

        skeleton = sk_skeletonize(binary > 0)
        return np.where(skeleton, 255, 0).astype(np.uint8)
    except Exception:
        return _morphological_skeleton(binary)


def _morphological_skeleton(binary: np.ndarray) -> np.ndarray:
    """OpenCV-only skeleton fallback when scikit-image is unavailable."""
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return np.where(skel > 0, 255, 0).astype(np.uint8)


def save_preprocessing_debug(
    *,
    pattern_binary: np.ndarray,
    pattern_cropped: np.ndarray,
    pattern_skeleton: np.ndarray,
    drawing_binary: np.ndarray,
    drawing_skeleton: np.ndarray,
    debug_dir: str | Path,
) -> None:
    """Save canonical preprocessing debug images."""
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    for name, image in {
        "pattern_binary.png": pattern_binary,
        "pattern_cropped.png": pattern_cropped,
        "pattern_skeleton.png": pattern_skeleton,
        "drawing_binary.png": drawing_binary,
        "drawing_skeleton.png": drawing_skeleton,
    }.items():
        path = debug_path / name
        ensure_parent_dir(path)
        cv2.imwrite(str(path), image)
