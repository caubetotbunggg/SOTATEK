"""Main zero-shot edge-feature detector orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .chamfer import symmetric_chamfer_scores
from .descriptor_matching import (
    DescriptorCandidate,
    DescriptorSearchDebug,
    expand_rotations,
    generate_edge_template_variants,
    parse_rotations,
    sliding_window_descriptor_search,
)
from .nms import non_max_suppression
from .preprocessing import (
    load_image,
    normalize_to_bgr,
    preprocess_image,
    preprocess_pattern,
    save_preprocessing_debug,
)
from .utils import clamp, round_float
from .validation import validate_edge_candidate
from .visualization import draw_detections, save_visualization


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    scale: float
    rotation: float
    descriptor_similarity: float
    chamfer_similarity: float
    edge_f1: float
    density_score: float
    template_coverage: float
    patch_coverage: float
    extra_patch_ratio: float
    chamfer_distance: float

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "confidence",
            "scale",
            "rotation",
            "descriptor_similarity",
            "chamfer_similarity",
            "edge_f1",
            "density_score",
            "template_coverage",
            "patch_coverage",
            "extra_patch_ratio",
            "chamfer_distance",
        ):
            data[key] = round_float(float(data[key]))
        return data


@dataclass
class DetectorConfig:
    threshold: float = 0.35
    min_scale: float = 0.05
    max_scale: float = 0.30
    scale_step: float = 0.02
    rotations: str = "0,90,180,270"
    fine_rotation_range: float = 0.0
    fine_rotation_step: float = 5.0
    stride: int = 2
    top_k: int = 800
    nms_iou_threshold: float = 0.30
    max_processing_dim: int = 2500
    chamfer_sigma: float = 8.0
    max_chamfer_distance: float = 12.0
    min_template_coverage: float = 0.25
    max_extra_patch_ratio: float = 0.90
    validation_dilation_iterations: int = 1
    max_detections: int = 200
    pattern_padding: int = 4
    enable_debug: bool = False
    debug_dir: str = "outputs/debug"


class PatternDetector:
    """Zero-shot detector based on skeleton edge descriptors and geometric validation."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.last_debug_counts: dict[str, int] = {}

    def detect_from_paths(self, pattern_path: str, drawing_path: str) -> tuple[list[Detection], np.ndarray]:
        pattern = load_image(pattern_path)
        drawing = load_image(drawing_path)
        return self.detect(pattern, drawing)

    def detect(self, pattern_image: np.ndarray, drawing_image: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """Detect pattern occurrences and return detections in original drawing coordinates."""
        cfg = self.config

        pattern_bgr = normalize_to_bgr(pattern_image)
        drawing_bgr = normalize_to_bgr(drawing_image)

        pattern_pre_crop = preprocess_image(pattern_bgr, max_dim=None, compute_distance=False)
        pattern = preprocess_pattern(pattern_bgr, padding=cfg.pattern_padding).processed
        drawing = preprocess_image(
            drawing_bgr,
            max_dim=cfg.max_processing_dim,
            compute_distance=False,
        )

        if cfg.enable_debug:
            save_preprocessing_debug(
                pattern_binary=pattern_pre_crop.binary,
                pattern_cropped=pattern.binary,
                pattern_skeleton=pattern.skeleton,
                drawing_binary=drawing.binary,
                drawing_skeleton=drawing.skeleton,
                debug_dir=cfg.debug_dir,
            )

        base_rotations = parse_rotations(cfg.rotations)
        rotations = expand_rotations(
            base_rotations,
            fine_rotation_range=cfg.fine_rotation_range,
            fine_rotation_step=cfg.fine_rotation_step,
        )
        variants = generate_edge_template_variants(
            pattern.skeleton,
            min_scale=cfg.min_scale,
            max_scale=cfg.max_scale,
            scale_step=cfg.scale_step,
            rotations=rotations,
            drawing_shape=drawing.skeleton.shape,
        )

        descriptor_candidates, search_debug = sliding_window_descriptor_search(
            drawing.skeleton,
            variants,
            stride=cfg.stride,
            top_k=cfg.top_k,
            heatmap_variant=(1.0, 0.0),
        )

        scored: list[Detection] = []
        num_after_chamfer_extreme_filter = 0
        for candidate in descriptor_candidates:
            detection, survived_chamfer_extreme = self._validate_candidate(candidate, drawing.skeleton)
            if survived_chamfer_extreme:
                num_after_chamfer_extreme_filter += 1
            if detection is not None:
                scored.append(detection)

        validated = [det for det in scored if det.confidence >= cfg.threshold]
        before_validation = [
            self._map_to_original(det, drawing.scale_to_original, drawing.original_bgr.shape)
            for det in self._raw_candidates_to_detections(descriptor_candidates)
        ]
        before_nms = [
            self._map_to_original(det, drawing.scale_to_original, drawing.original_bgr.shape) for det in validated
        ]

        suppressed = non_max_suppression(validated, cfg.nms_iou_threshold)
        suppressed = suppressed[: cfg.max_detections]
        mapped = [self._map_to_original(det, drawing.scale_to_original, drawing.original_bgr.shape) for det in suppressed]
        visualization = draw_detections(drawing.original_bgr, mapped)

        self.last_debug_counts = {
            "num_template_variants": len(variants),
            "num_raw_candidates": len(descriptor_candidates),
            "num_after_chamfer_extreme_filter": num_after_chamfer_extreme_filter,
            "num_after_validation_scoring": len(scored),
            "num_above_threshold": len(validated),
            "num_after_nms": len(suppressed),
        }

        if cfg.enable_debug:
            self._print_debug_counts()
            self._save_debug_visuals(drawing.original_bgr, before_validation, before_nms, visualization, search_debug)

        return mapped, visualization

    
    def _validate_candidate(self, candidate: DescriptorCandidate, drawing_skeleton: np.ndarray) -> tuple[Detection | None, bool]:
        cfg = self.config
        w, h = candidate.w, candidate.h
        offsets = _refinement_offsets(cfg.stride)
        best: Detection | None = None
        survived_chamfer_extreme = False

        for dy in offsets:
            for dx in offsets:
                x = candidate.x + dx
                y = candidate.y + dy
                if x < 0 or y < 0 or y + h > drawing_skeleton.shape[0] or x + w > drawing_skeleton.shape[1]:
                    continue

                patch = drawing_skeleton[y : y + h, x : x + w]
                template = candidate.variant.edge

                chamfer = symmetric_chamfer_scores(template, patch, sigma=cfg.chamfer_sigma)
                if chamfer.symmetric_distance > cfg.max_chamfer_distance * 2.5:
                    continue
                survived_chamfer_extreme = True

                chamfer_penalty = 1.0
                if chamfer.symmetric_distance > cfg.max_chamfer_distance:
                    chamfer_penalty = 0.70

                validation = validate_edge_candidate(
                    template,
                    patch,
                    min_template_coverage=cfg.min_template_coverage,
                    max_extra_patch_ratio=cfg.max_extra_patch_ratio,
                    dilation_iterations=cfg.validation_dilation_iterations,
                )

                validation_penalty = 1.0 if validation.passed else 0.70

                confidence = (
                    0.05 * candidate.descriptor_similarity
                    + 0.45 * chamfer.similarity
                    + 0.45 * validation.edge_f1
                    + 0.05 * validation.density_score
                )
                confidence *= chamfer_penalty
                confidence *= validation_penalty
                confidence = float(np.clip(confidence, 0.0, 1.0))

                detection = Detection(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    confidence=confidence,
                    scale=candidate.scale,
                    rotation=candidate.rotation,
                    descriptor_similarity=candidate.descriptor_similarity,
                    chamfer_similarity=chamfer.similarity,
                    edge_f1=validation.edge_f1,
                    density_score=validation.density_score,
                    template_coverage=validation.template_coverage,
                    patch_coverage=validation.patch_coverage,
                    extra_patch_ratio=validation.extra_patch_ratio,
                    chamfer_distance=chamfer.symmetric_distance,
                )

                if best is None or detection.confidence > best.confidence:
                    best = detection

        return best, survived_chamfer_extreme

    def _save_debug_visuals(
        self,
        drawing_bgr: np.ndarray,
        before_validation: list[Detection],
        before_nms: list[Detection],
        final_visualization: np.ndarray,
        search_debug: DescriptorSearchDebug,
    ) -> None:
        debug_dir = Path(self.config.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        before_validation_vis = draw_detections(drawing_bgr, before_validation)
        before_nms_vis = draw_detections(drawing_bgr, before_nms)
        save_visualization(before_validation_vis, str(debug_dir / "candidates_before_validation.png"))
        save_visualization(before_nms_vis, str(debug_dir / "candidates_after_validation_before_nms.png"))
        save_visualization(final_visualization, str(debug_dir / "final_result.png"))

        if search_debug.heatmap is not None and search_debug.heatmap.size:
            heatmap = np.clip(search_debug.heatmap, 0.0, 1.0)
            heatmap_u8 = (heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_TURBO)
            save_visualization(heatmap_color, str(debug_dir / "descriptor_heatmap_s1_r0.png"))

    def _print_debug_counts(self) -> None:
        print("Detection debug counts:")
        for key, value in self.last_debug_counts.items():
            print(f"  {key}: {value}")

    @staticmethod
    def _raw_candidates_to_detections(candidates: list[DescriptorCandidate]) -> list[Detection]:
        return [
            Detection(
                x=candidate.x,
                y=candidate.y,
                w=candidate.w,
                h=candidate.h,
                confidence=candidate.descriptor_similarity,
                scale=candidate.scale,
                rotation=candidate.rotation,
                descriptor_similarity=candidate.descriptor_similarity,
                chamfer_similarity=0.0,
                edge_f1=0.0,
                density_score=0.0,
                template_coverage=0.0,
                patch_coverage=0.0,
                extra_patch_ratio=0.0,
                chamfer_distance=0.0,
            )
            for candidate in candidates
        ]

    @staticmethod
    def _map_to_original(det: Detection, scale_to_original: float, original_shape: tuple[int, ...]) -> Detection:
        if abs(scale_to_original - 1.0) < 1e-9:
            return det

        original_h, original_w = original_shape[:2]
        x1 = clamp(int(round(det.x * scale_to_original)), 0, original_w - 1)
        y1 = clamp(int(round(det.y * scale_to_original)), 0, original_h - 1)
        x2 = clamp(int(round((det.x + det.w) * scale_to_original)), x1 + 1, original_w)
        y2 = clamp(int(round((det.y + det.h) * scale_to_original)), y1 + 1, original_h)
        return Detection(
            x=x1,
            y=y1,
            w=x2 - x1,
            h=y2 - y1,
            confidence=det.confidence,
            scale=det.scale,
            rotation=det.rotation,
            descriptor_similarity=det.descriptor_similarity,
            chamfer_similarity=det.chamfer_similarity,
            edge_f1=det.edge_f1,
            density_score=det.density_score,
            template_coverage=det.template_coverage,
            patch_coverage=det.patch_coverage,
            extra_patch_ratio=det.extra_patch_ratio,
            chamfer_distance=det.chamfer_distance,
        )


def _refinement_offsets(stride: int) -> list[int]:
    radius = max(1, int(stride))
    half = max(1, radius // 2)
    offsets = [-radius, -half, 0, half, radius]
    unique: list[int] = []
    for offset in offsets:
        if offset not in unique:
            unique.append(offset)
    return unique
