"""Main zero-shot edge-feature detector orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .chamfer import ChamferScores, distance_transform_to_edges
from .descriptor_matching import (
    DescriptorCandidate,
    DescriptorSearchDebug,
    expand_rotations,
    generate_edge_template_variants,
    parse_rotations,
    sliding_window_descriptor_search,
)
from .direct_matcher import DirectMatchCandidate, run_binary_direct_branch, run_skeleton_direct_branch
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
    core_edge_f1: float = 0.0
    connector_edge_f1: float = 0.0
    core_chamfer_similarity: float = 0.0
    connector_chamfer_similarity: float = 0.0
    shape_score: float = 0.0
    branches: list[str] = field(default_factory=list)
    branch_scores: dict[str, float] = field(default_factory=dict)
    skeleton_direct_score: float = 0.0
    binary_direct_score: float = 0.0
    edge_iou: float = 0.0
    xor_similarity: float = 0.0
    ncc_score: float = 0.0

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
            "core_edge_f1",
            "connector_edge_f1",
            "core_chamfer_similarity",
            "connector_chamfer_similarity",
            "shape_score",
            "skeleton_direct_score",
            "binary_direct_score",
            "edge_iou",
            "xor_similarity",
            "ncc_score",
        ):
            data[key] = round_float(float(data[key]))
        data["branch_scores"] = {
            key: round_float(float(value)) for key, value in data["branch_scores"].items()
        }
        return data


@dataclass
class CandidateDebug:
    detection: Detection
    patch: np.ndarray
    core_edge: np.ndarray
    connector_edge: np.ndarray
    core_validation: Any
    connector_validation: Any
    coarse_box: tuple[int, int, int, int]
    refinement_shift: tuple[int, int]
    refinement_scale: float
    refine_score: float


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
    min_patch_coverage: float = 0.25
    max_extra_patch_ratio: float = 0.90
    validation_dilation_iterations: int = 2
    local_refinement_radius: int = 4
    validation_padding: int = 3
    enable_descriptor_branch: bool = True
    enable_skeleton_direct_branch: bool = True
    enable_binary_direct_branch: bool = True
    direct_match_top_k: int = 500
    direct_match_stride: int = 2
    direct_match_dilation: int = 1
    branch_support_iou: float = 0.50
    multi_branch_boost: float = 0.08
    max_branch_boost: float = 0.16
    skeleton_direct_weight_iou: float = 0.45
    skeleton_direct_weight_chamfer: float = 0.35
    skeleton_direct_weight_xor: float = 0.15
    skeleton_direct_weight_density: float = 0.05
    binary_direct_weight_ncc: float = 0.35
    binary_direct_weight_iou: float = 0.30
    binary_direct_weight_xor: float = 0.20
    binary_direct_weight_density: float = 0.15
    binary_direct_use_edges: bool = False
    binary_direct_use_masked_ncc: bool = True
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
            self._save_template_split_debug(pattern.skeleton)

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
        binary_variants = generate_edge_template_variants(
            pattern.binary,
            min_scale=cfg.min_scale,
            max_scale=cfg.max_scale,
            scale_step=cfg.scale_step,
            rotations=rotations,
            drawing_shape=drawing.binary.shape,
        )

        descriptor_candidates: list[DescriptorCandidate] = []
        search_debug = DescriptorSearchDebug()
        if cfg.enable_descriptor_branch:
            descriptor_candidates, search_debug = sliding_window_descriptor_search(
                drawing.skeleton,
                variants,
                stride=cfg.stride,
                top_k=cfg.top_k,
                heatmap_variant=(1.0, 0.0),
            )

        scored: list[Detection] = []
        debug_details: list[CandidateDebug] = []
        num_after_chamfer_extreme_filter = 0
        if cfg.enable_descriptor_branch:
            for candidate in descriptor_candidates:
                detection, survived_chamfer_extreme, debug_detail = self._validate_candidate(candidate, drawing.skeleton)
                if survived_chamfer_extreme:
                    num_after_chamfer_extreme_filter += 1
                if detection is not None:
                    detection.branches = ["descriptor"]
                    detection.branch_scores = {"descriptor": detection.confidence}
                    scored.append(detection)
                    if debug_detail is not None:
                        debug_details.append(debug_detail)

        skeleton_direct_candidates: list[DirectMatchCandidate] = []
        binary_direct_candidates: list[DirectMatchCandidate] = []
        if cfg.enable_skeleton_direct_branch:
            skeleton_direct_candidates = run_skeleton_direct_branch(drawing.skeleton, variants, cfg)
            scored.extend(_direct_candidates_to_detections(skeleton_direct_candidates))
        if cfg.enable_binary_direct_branch:
            binary_direct_candidates = run_binary_direct_branch(drawing.binary, binary_variants, cfg)
            scored.extend(_direct_candidates_to_detections(binary_direct_candidates))

        merged = merge_branch_candidates(scored, cfg)
        validated = [det for det in merged if det.confidence >= cfg.threshold]
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
            "num_merged_candidates": len(merged),
            "num_above_threshold": len(validated),
            "num_after_nms": len(suppressed),
        }
        branch_summary = _branch_summary(
            descriptor=scored,
            skeleton_direct=skeleton_direct_candidates,
            binary_direct=binary_direct_candidates,
            merged=merged,
            after_nms=suppressed,
        )

        if cfg.enable_debug:
            self._print_debug_counts()
            self._save_debug_visuals(
                drawing.original_bgr,
                before_validation,
                before_nms,
                visualization,
                search_debug,
                sorted(debug_details, key=lambda item: item.detection.confidence, reverse=True),
                skeleton_direct_candidates,
                binary_direct_candidates,
                branch_summary,
            )

        return mapped, visualization

    
    def _validate_candidate(
        self,
        candidate: DescriptorCandidate,
        drawing_skeleton: np.ndarray,
    ) -> tuple[Detection | None, bool, CandidateDebug | None]:
        cfg = self.config
        w, h = candidate.w, candidate.h
        offsets = _refinement_offsets(cfg.local_refinement_radius)
        best: Detection | None = None
        best_debug: CandidateDebug | None = None
        survived_chamfer_extreme = False
        template = candidate.variant.edge

        for scale_multiplier in (0.95, 1.0, 1.05):
            scaled_template = _resize_edge_template(template, scale_multiplier)
            if scaled_template.size == 0:
                continue
            scaled_h, scaled_w = scaled_template.shape[:2]
            center_adjust_x = int(round((scaled_w - w) / 2.0))
            center_adjust_y = int(round((scaled_h - h) / 2.0))

            for dy in offsets:
                for dx in offsets:
                    x = candidate.x + dx - center_adjust_x
                    y = candidate.y + dy - center_adjust_y
                    prepared = _prepare_refinement_patch(
                        scaled_template,
                        drawing_skeleton,
                        x=x,
                        y=y,
                        padding=cfg.validation_padding,
                    )
                    if prepared is None:
                        continue

                    detection, debug_detail, refine_score = self._score_refined_candidate(
                        candidate=candidate,
                        patch=prepared["patch"],
                        template=prepared["template"],
                        core_edge=prepared["core_edge"],
                        connector_edge=prepared["connector_edge"],
                        template_weight=prepared["template_weight"],
                        bbox=prepared["bbox"],
                        coarse_box=(candidate.x, candidate.y, w, h),
                        refinement_shift=(dx, dy),
                        refinement_scale=scale_multiplier,
                    )
                    if detection is None or debug_detail is None:
                        continue

                    survived_chamfer_extreme = True
                    if best is None or refine_score > best_debug.refine_score:
                        best = detection
                        best_debug = debug_detail

        return best, survived_chamfer_extreme, best_debug

    def _score_refined_candidate(
        self,
        *,
        candidate: DescriptorCandidate,
        patch: np.ndarray,
        template: np.ndarray,
        core_edge: np.ndarray,
        connector_edge: np.ndarray,
        template_weight: np.ndarray,
        bbox: tuple[int, int, int, int],
        coarse_box: tuple[int, int, int, int],
        refinement_shift: tuple[int, int],
        refinement_scale: float,
    ) -> tuple[Detection | None, CandidateDebug | None, float]:
        cfg = self.config
        x, y, w, h = bbox

        core_chamfer = _weighted_chamfer_scores(core_edge, patch, template_weight, sigma=cfg.chamfer_sigma)
        if core_chamfer.symmetric_distance > cfg.max_chamfer_distance * 2.5:
            return None, None, 0.0

        chamfer_penalty = 1.0
        if core_chamfer.symmetric_distance > cfg.max_chamfer_distance:
            chamfer_penalty = 0.70

        core_validation = validate_edge_candidate(
            core_edge,
            patch,
            min_template_coverage=cfg.min_template_coverage,
            min_patch_coverage=0.0,
            max_extra_patch_ratio=1.0,
            dilation_iterations=cfg.validation_dilation_iterations,
            center_weight=template_weight,
        )
        connector_validation = validate_edge_candidate(
            connector_edge,
            patch,
            min_template_coverage=0.0,
            min_patch_coverage=0.0,
            max_extra_patch_ratio=1.0,
            dilation_iterations=cfg.validation_dilation_iterations,
            center_weight=template_weight,
        )
        coverage_validation = validate_edge_candidate(
            template,
            patch,
            min_template_coverage=cfg.min_template_coverage,
            min_patch_coverage=cfg.min_patch_coverage,
            max_extra_patch_ratio=cfg.max_extra_patch_ratio,
            dilation_iterations=cfg.validation_dilation_iterations,
        )

        connector_chamfer = _weighted_chamfer_scores(
            connector_edge,
            patch,
            template_weight,
            sigma=cfg.chamfer_sigma,
        )
        core_edge_f1 = core_validation.edge_f1
        connector_edge_f1 = connector_validation.edge_f1
        refine_score = float(np.clip((0.60 * core_edge_f1) + (0.40 * core_chamfer.similarity), 0.0, 1.0))
        shape_score = (
            0.60 * core_edge_f1
            + 0.30 * core_chamfer.similarity
            + 0.10 * connector_edge_f1
        )
        if core_edge_f1 < 0.20:
            shape_score *= 0.40

        validation_penalty = 1.0 if coverage_validation.passed else 0.70
        coverage_penalty = 1.0
        if coverage_validation.patch_coverage < cfg.min_patch_coverage:
            coverage_penalty *= 0.65
        if coverage_validation.extra_patch_ratio > 0.80:
            coverage_penalty *= 0.75

        confidence = (
            0.05 * candidate.descriptor_similarity
            + 0.55 * core_edge_f1
            + 0.30 * core_chamfer.similarity
            + 0.10 * coverage_validation.density_score
        )
        confidence *= chamfer_penalty
        confidence *= validation_penalty
        confidence *= coverage_penalty
        confidence = float(np.clip(confidence, 0.0, 1.0))

        detection = Detection(
            x=x,
            y=y,
            w=w,
            h=h,
            confidence=confidence,
            scale=candidate.scale * refinement_scale,
            rotation=candidate.rotation,
            descriptor_similarity=candidate.descriptor_similarity,
            chamfer_similarity=core_chamfer.similarity,
            edge_f1=core_edge_f1,
            density_score=coverage_validation.density_score,
            template_coverage=coverage_validation.template_coverage,
            patch_coverage=coverage_validation.patch_coverage,
            extra_patch_ratio=coverage_validation.extra_patch_ratio,
            chamfer_distance=core_chamfer.symmetric_distance,
            core_edge_f1=core_edge_f1,
            connector_edge_f1=connector_edge_f1,
            core_chamfer_similarity=core_chamfer.similarity,
            connector_chamfer_similarity=connector_chamfer.similarity,
            shape_score=shape_score,
        )
        debug_detail = CandidateDebug(
            detection=detection,
            patch=patch.copy(),
            core_edge=core_edge.copy(),
            connector_edge=connector_edge.copy(),
            core_validation=core_validation,
            connector_validation=connector_validation,
            coarse_box=coarse_box,
            refinement_shift=refinement_shift,
            refinement_scale=refinement_scale,
            refine_score=refine_score,
        )
        return detection, debug_detail, refine_score

    def _save_debug_visuals(
        self,
        drawing_bgr: np.ndarray,
        before_validation: list[Detection],
        before_nms: list[Detection],
        final_visualization: np.ndarray,
        search_debug: DescriptorSearchDebug,
        debug_details: list[CandidateDebug],
        skeleton_direct_candidates: list[DirectMatchCandidate],
        binary_direct_candidates: list[DirectMatchCandidate],
        branch_summary: dict[str, Any],
    ) -> None:
        debug_dir = Path(self.config.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        before_validation_vis = draw_detections(drawing_bgr, before_validation)
        before_nms_vis = draw_detections(drawing_bgr, before_nms)
        save_visualization(before_validation_vis, str(debug_dir / "candidates_before_validation.png"))
        save_visualization(before_nms_vis, str(debug_dir / "candidates_after_validation_before_nms.png"))
        save_visualization(draw_detections(drawing_bgr, skeleton_direct_candidates), str(debug_dir / "skeleton_direct_candidates_before_nms.png"))
        save_visualization(draw_detections(drawing_bgr, binary_direct_candidates), str(debug_dir / "binary_direct_candidates_before_nms.png"))
        save_visualization(before_nms_vis, str(debug_dir / "merged_candidates_before_nms.png"))
        save_visualization(final_visualization, str(debug_dir / "final_result.png"))
        with open(debug_dir / "branch_summary.json", "w", encoding="utf-8") as f:
            json.dump(branch_summary, f, indent=2)

        if search_debug.heatmap is not None and search_debug.heatmap.size:
            heatmap = np.clip(search_debug.heatmap, 0.0, 1.0)
            heatmap_u8 = (heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_TURBO)
            save_visualization(heatmap_color, str(debug_dir / "descriptor_heatmap_s1_r0.png"))

        if debug_details:
            self._save_candidate_debug(debug_dir, debug_details[:50])

    def _print_debug_counts(self) -> None:
        print("Detection debug counts:")
        for key, value in self.last_debug_counts.items():
            print(f"  {key}: {value}")

    def _save_template_split_debug(self, template_edge: np.ndarray) -> None:
        debug_dir = Path(self.config.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        core_edge, connector_edge, weight_mask = split_template_core_and_connectors(template_edge)
        save_visualization(_edge_to_bgr(core_edge), str(debug_dir / "template_core_edge.png"))
        save_visualization(_edge_to_bgr(connector_edge), str(debug_dir / "template_connector_edge.png"))
        save_visualization(_gray_to_bgr((weight_mask * 255).astype(np.uint8)), str(debug_dir / "template_weight_mask.png"))
        save_visualization(_core_connector_overlay(core_edge, connector_edge), str(debug_dir / "template_core_connector_overlay.png"))

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
                core_edge_f1=0.0,
                connector_edge_f1=0.0,
                core_chamfer_similarity=0.0,
                connector_chamfer_similarity=0.0,
                shape_score=0.0,
                branches=[],
                branch_scores={},
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
            core_edge_f1=det.core_edge_f1,
            connector_edge_f1=det.connector_edge_f1,
            core_chamfer_similarity=det.core_chamfer_similarity,
            connector_chamfer_similarity=det.connector_chamfer_similarity,
            shape_score=det.shape_score,
            branches=list(det.branches),
            branch_scores=dict(det.branch_scores),
            skeleton_direct_score=det.skeleton_direct_score,
            binary_direct_score=det.binary_direct_score,
            edge_iou=det.edge_iou,
            xor_similarity=det.xor_similarity,
            ncc_score=det.ncc_score,
        )

    def _save_candidate_debug(self, debug_dir: Path, debug_details: list[CandidateDebug]) -> None:
        for idx, detail in enumerate(debug_details):
            prefix = debug_dir / f"det_{idx:03d}"
            save_visualization(_edge_to_bgr(detail.patch), str(prefix.with_name(f"{prefix.name}_patch_skeleton.png")))
            save_visualization(
                _overlap_visualization(detail.core_edge, detail.patch),
                str(prefix.with_name(f"{prefix.name}_core_overlap.png")),
            )
            save_visualization(
                _overlap_visualization(detail.connector_edge, detail.patch),
                str(prefix.with_name(f"{prefix.name}_connector_overlap.png")),
            )
            save_visualization(
                _refinement_overlay(detail.coarse_box, detail.detection),
                str(prefix.with_name(f"{prefix.name}_refinement_overlay.png")),
            )
            metrics = detail.detection.to_json()
            metrics.update(
                {
                    "core_template_coverage": round_float(detail.core_validation.template_coverage),
                    "core_patch_coverage": round_float(detail.core_validation.patch_coverage),
                    "connector_template_coverage": round_float(detail.connector_validation.template_coverage),
                    "connector_patch_coverage": round_float(detail.connector_validation.patch_coverage),
                    "refine_score": round_float(detail.refine_score),
                    "refinement_dx": detail.refinement_shift[0],
                    "refinement_dy": detail.refinement_shift[1],
                    "refinement_scale": round_float(detail.refinement_scale),
                    "coarse_x": detail.coarse_box[0],
                    "coarse_y": detail.coarse_box[1],
                    "coarse_w": detail.coarse_box[2],
                    "coarse_h": detail.coarse_box[3],
                    "final_confidence": round_float(detail.detection.confidence),
                }
            )
            with open(prefix.with_name(f"{prefix.name}_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)


def _refinement_offsets(radius: int) -> list[int]:
    radius = max(0, int(radius))
    return list(range(-radius, radius + 1))


def _direct_candidates_to_detections(candidates: list[DirectMatchCandidate]) -> list[Detection]:
    detections: list[Detection] = []
    for candidate in candidates:
        detection = Detection(
            x=candidate.x,
            y=candidate.y,
            w=candidate.w,
            h=candidate.h,
            confidence=candidate.score,
            scale=candidate.scale,
            rotation=candidate.rotation,
            descriptor_similarity=0.0,
            chamfer_similarity=candidate.chamfer_similarity,
            edge_f1=0.0,
            density_score=candidate.density_score,
            template_coverage=0.0,
            patch_coverage=0.0,
            extra_patch_ratio=0.0,
            chamfer_distance=0.0,
            branches=[candidate.branch],
            branch_scores={candidate.branch: candidate.score},
            skeleton_direct_score=candidate.score if candidate.branch == "skeleton_direct" else 0.0,
            binary_direct_score=candidate.score if candidate.branch == "binary_direct" else 0.0,
            edge_iou=candidate.edge_iou,
            xor_similarity=candidate.xor_similarity,
            ncc_score=candidate.ncc_score,
        )
        detections.append(detection)
    return detections


def merge_branch_candidates(candidates: list[Detection], cfg: DetectorConfig) -> list[Detection]:
    remaining = sorted(candidates, key=lambda det: det.confidence, reverse=True)
    merged: list[Detection] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        keep: list[Detection] = []
        for candidate in remaining:
            if _detection_iou(seed, candidate) >= cfg.branch_support_iou:
                group.append(candidate)
            else:
                keep.append(candidate)
        remaining = keep

        best = max(group, key=lambda det: det.confidence)
        branches = sorted({branch for det in group for branch in det.branches})
        branch_scores: dict[str, float] = {}
        for det in group:
            for branch, score in det.branch_scores.items():
                branch_scores[branch] = max(branch_scores.get(branch, 0.0), float(score))
        boost = min(max(0, len(branches) - 1) * cfg.multi_branch_boost, cfg.max_branch_boost)
        best.confidence = float(np.clip(max(det.confidence for det in group) + boost, 0.0, 1.0))
        best.branches = branches
        best.branch_scores = branch_scores
        best.skeleton_direct_score = branch_scores.get("skeleton_direct", best.skeleton_direct_score)
        best.binary_direct_score = branch_scores.get("binary_direct", best.binary_direct_score)
        best.edge_iou = max((det.edge_iou for det in group), default=best.edge_iou)
        best.xor_similarity = max((det.xor_similarity for det in group), default=best.xor_similarity)
        best.ncc_score = max((det.ncc_score for det in group), default=best.ncc_score)
        merged.append(best)
    return merged


def _detection_iou(a: Detection, b: Detection) -> float:
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = (a.w * a.h) + (b.w * b.h) - inter
    return float(inter / union) if union > 0 else 0.0


def _branch_summary(
    *,
    descriptor: list[Detection],
    skeleton_direct: list[DirectMatchCandidate],
    binary_direct: list[DirectMatchCandidate],
    merged: list[Detection],
    after_nms: list[Detection],
) -> dict[str, Any]:
    descriptor_only = [det for det in descriptor if "descriptor" in det.branch_scores]
    return {
        "descriptor": {
            "num_candidates": len(descriptor_only),
            "top_score": round_float(max((det.branch_scores.get("descriptor", 0.0) for det in descriptor_only), default=0.0)),
        },
        "skeleton_direct": {
            "num_candidates": len(skeleton_direct),
            "top_score": round_float(max((candidate.score for candidate in skeleton_direct), default=0.0)),
        },
        "binary_direct": {
            "num_candidates": len(binary_direct),
            "top_score": round_float(max((candidate.score for candidate in binary_direct), default=0.0)),
        },
        "merged": {
            "num_candidates": len(merged),
            "num_after_nms": len(after_nms),
        },
    }


def _resize_edge_template(template_edge: np.ndarray, scale_multiplier: float) -> np.ndarray:
    edge = np.where(template_edge > 0, 255, 0).astype(np.uint8)
    if abs(scale_multiplier - 1.0) < 1e-9:
        return edge

    h, w = edge.shape[:2]
    new_w = max(3, int(round(w * scale_multiplier)))
    new_h = max(3, int(round(h * scale_multiplier)))
    return cv2.resize(edge, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def _prepare_refinement_patch(
    scaled_template: np.ndarray,
    drawing_skeleton: np.ndarray,
    *,
    x: int,
    y: int,
    padding: int,
) -> dict[str, Any] | None:
    draw_h, draw_w = drawing_skeleton.shape[:2]
    template_h, template_w = scaled_template.shape[:2]
    if template_h <= 0 or template_w <= 0:
        return None

    x1 = x
    y1 = y
    x2 = x + template_w
    y2 = y + template_h
    if x2 <= 0 or y2 <= 0 or x1 >= draw_w or y1 >= draw_h:
        return None

    pad = max(0, int(padding))
    patch_x1 = clamp(x1 - pad, 0, draw_w - 1)
    patch_y1 = clamp(y1 - pad, 0, draw_h - 1)
    patch_x2 = clamp(x2 + pad, patch_x1 + 1, draw_w)
    patch_y2 = clamp(y2 + pad, patch_y1 + 1, draw_h)
    patch = drawing_skeleton[patch_y1:patch_y2, patch_x1:patch_x2]
    if patch.size == 0:
        return None

    template_canvas = np.zeros(patch.shape[:2], dtype=np.uint8)
    dst_x1 = max(0, x1 - patch_x1)
    dst_y1 = max(0, y1 - patch_y1)
    src_x1 = max(0, patch_x1 - x1)
    src_y1 = max(0, patch_y1 - y1)
    copy_w = min(template_w - src_x1, template_canvas.shape[1] - dst_x1)
    copy_h = min(template_h - src_y1, template_canvas.shape[0] - dst_y1)
    if copy_w <= 0 or copy_h <= 0:
        return None
    template_canvas[dst_y1 : dst_y1 + copy_h, dst_x1 : dst_x1 + copy_w] = scaled_template[
        src_y1 : src_y1 + copy_h,
        src_x1 : src_x1 + copy_w,
    ]

    core_edge, connector_edge, weight_mask = split_template_core_and_connectors(template_canvas)
    template_weight = np.where(template_canvas > 0, weight_mask, 0.0).astype(np.float32)
    return {
        "patch": patch,
        "template": template_canvas,
        "core_edge": core_edge,
        "connector_edge": connector_edge,
        "template_weight": template_weight,
        "bbox": (patch_x1, patch_y1, patch_x2 - patch_x1, patch_y2 - patch_y1),
    }


def split_template_core_and_connectors(template_edge: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    template = np.where(template_edge > 0, 255, 0).astype(np.uint8)
    h, w = template.shape[:2]
    edge_mask = template > 0

    connector_line_mask = _dominant_connector_line_mask(template)
    connector_mask = edge_mask & connector_line_mask
    core_mask = edge_mask & ~connector_mask

    core_edge = np.where(core_mask, 255, 0).astype(np.uint8)
    connector_edge = np.where(connector_mask, 255, 0).astype(np.uint8)
    weight_mask = np.zeros((h, w), dtype=np.float32)
    weight_mask[core_mask] = 1.0
    weight_mask[connector_mask] = 0.20
    return core_edge, connector_edge, weight_mask


def _dominant_connector_line_mask(template: np.ndarray) -> np.ndarray:
    h, w = template.shape[:2]
    line_mask = np.zeros((h, w), dtype=np.uint8)
    max_dim = max(h, w)
    threshold = max(4, int(round(max_dim * 0.10)))
    min_line_length = max(6, int(round(max_dim * 0.45)))
    max_line_gap = max(2, int(round(max_dim * 0.08)))

    lines = cv2.HoughLinesP(
        template,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            if _is_dominant_connector_line(x1, y1, x2, y2, h, w):
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

    projection_mask = _projection_connector_line_mask(template)
    line_mask = cv2.bitwise_or(line_mask, projection_mask)

    if not np.any(line_mask):
        return np.zeros((h, w), dtype=bool)

    kernel = np.ones((3, 3), np.uint8)
    line_mask = cv2.dilate(line_mask, kernel, iterations=1)
    return line_mask > 0


def _is_dominant_connector_line(x1: int, y1: int, x2: int, y2: int, h: int, w: int) -> bool:
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = float(np.hypot(dx, dy))
    if length <= 0:
        return False

    angle = abs(np.degrees(np.arctan2(dy, dx))) % 180.0
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    x_mid = (x1 + x2) / 2.0
    y_mid = (y1 + y2) / 2.0

    is_horizontal = angle <= 12.0 or angle >= 168.0
    is_vertical = 78.0 <= angle <= 102.0
    horizontal_through_center = (
        is_horizontal
        and length >= 0.45 * w
        and abs(y_mid - (h / 2.0)) <= 0.18 * h
        and x_min <= 0.22 * w
        and x_max >= 0.78 * w
    )
    vertical_through_center = (
        is_vertical
        and length >= 0.45 * h
        and abs(x_mid - (w / 2.0)) <= 0.18 * w
        and y_min <= 0.22 * h
        and y_max >= 0.78 * h
    )
    return horizontal_through_center or vertical_through_center


def _projection_connector_line_mask(template: np.ndarray) -> np.ndarray:
    h, w = template.shape[:2]
    edge = template > 0
    line_mask = np.zeros((h, w), dtype=np.uint8)

    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0
    horizontal_band = max(1, int(round(0.12 * h)))
    vertical_band = max(1, int(round(0.12 * w)))

    for y in range(h):
        if abs(y - center_y) > horizontal_band:
            continue
        xs = np.flatnonzero(edge[y, :])
        if xs.size < max(3, int(round(0.20 * w))):
            continue
        if xs[0] <= 0.22 * w and xs[-1] >= 0.78 * w and (xs[-1] - xs[0]) >= 0.45 * w:
            cv2.line(line_mask, (int(xs[0]), y), (int(xs[-1]), y), 255, 1)

    for x in range(w):
        if abs(x - center_x) > vertical_band:
            continue
        ys = np.flatnonzero(edge[:, x])
        if ys.size < max(3, int(round(0.20 * h))):
            continue
        if ys[0] <= 0.22 * h and ys[-1] >= 0.78 * h and (ys[-1] - ys[0]) >= 0.45 * h:
            cv2.line(line_mask, (x, int(ys[0])), (x, int(ys[-1])), 255, 1)

    return line_mask


def _weighted_chamfer_scores(
    template_edge: np.ndarray,
    patch_edge: np.ndarray,
    weight_mask: np.ndarray,
    *,
    sigma: float,
) -> ChamferScores:
    template = np.where(template_edge > 0, 255, 0).astype(np.uint8)
    patch = np.where(patch_edge > 0, 255, 0).astype(np.uint8)
    weights = np.asarray(weight_mask, dtype=np.float32)
    if weights.shape[:2] != template.shape[:2]:
        weights = cv2.resize(weights, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_LINEAR)

    t2p = _weighted_one_way_chamfer(template, patch, weights)
    p2t = _weighted_one_way_chamfer(patch, template, weights)

    if not np.isfinite(t2p) or not np.isfinite(p2t):
        return ChamferScores(
            template_to_patch_distance=float("inf"),
            patch_to_template_distance=float("inf"),
            symmetric_distance=float("inf"),
            similarity=0.0,
        )

    symmetric_distance = max(t2p, p2t)
    similarity = float(np.exp(-symmetric_distance / max(sigma, 1e-6)))
    return ChamferScores(
        template_to_patch_distance=float(t2p),
        patch_to_template_distance=float(p2t),
        symmetric_distance=float(symmetric_distance),
        similarity=float(np.clip(similarity, 0.0, 1.0)),
    )


def _weighted_one_way_chamfer(source_edge: np.ndarray, target_edge: np.ndarray, weights: np.ndarray) -> float:
    source_mask = source_edge > 0
    if not np.any(source_mask):
        return float("inf")

    target_distance = distance_transform_to_edges(target_edge)
    distances = target_distance[source_mask]
    source_weights = weights[source_mask]
    if distances.size == 0 or source_weights.size == 0:
        return float("inf")

    return _weighted_percentile(distances.astype(np.float32), source_weights.astype(np.float32), percentile=80.0)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, *, percentile: float) -> float:
    positive = weights > 0
    if not np.any(positive):
        return float("inf")

    values = values[positive]
    weights = weights[positive]
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = (percentile / 100.0) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = min(max(index, 0), sorted_values.size - 1)
    return float(sorted_values[index])


def _edge_to_bgr(edge_img: np.ndarray) -> np.ndarray:
    edge = np.where(edge_img > 0, 255, 0).astype(np.uint8)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)


def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _overlap_visualization(template_edge: np.ndarray, patch_edge: np.ndarray) -> np.ndarray:
    template = template_edge > 0
    patch = patch_edge > 0
    vis = np.zeros((*template.shape[:2], 3), dtype=np.uint8)
    vis[np.logical_and(template, patch)] = (0, 255, 0)
    vis[np.logical_and(template, ~patch)] = (0, 0, 255)
    vis[np.logical_and(~template, patch)] = (255, 0, 0)
    return vis


def _core_connector_overlay(core_edge: np.ndarray, connector_edge: np.ndarray) -> np.ndarray:
    core = core_edge > 0
    connector = connector_edge > 0
    vis = np.zeros((*core.shape[:2], 3), dtype=np.uint8)
    vis[core] = (0, 255, 0)
    vis[connector] = (0, 80, 255)
    vis[np.logical_and(core, connector)] = (255, 255, 255)
    return vis


def _refinement_overlay(coarse_box: tuple[int, int, int, int], refined: Detection) -> np.ndarray:
    cx, cy, cw, ch = coarse_box
    rx, ry, rw, rh = refined.x, refined.y, refined.w, refined.h
    min_x = min(cx, rx)
    min_y = min(cy, ry)
    max_x = max(cx + cw, rx + rw)
    max_y = max(cy + ch, ry + rh)
    margin = 8
    width = max(32, max_x - min_x + (2 * margin))
    height = max(32, max_y - min_y + (2 * margin))
    vis = np.zeros((height, width, 3), dtype=np.uint8)

    coarse_pt1 = (cx - min_x + margin, cy - min_y + margin)
    coarse_pt2 = (cx + cw - min_x + margin, cy + ch - min_y + margin)
    refined_pt1 = (rx - min_x + margin, ry - min_y + margin)
    refined_pt2 = (rx + rw - min_x + margin, ry + rh - min_y + margin)
    cv2.rectangle(vis, coarse_pt1, coarse_pt2, (255, 80, 0), 1)
    cv2.rectangle(vis, refined_pt1, refined_pt2, (0, 180, 255), 1)
    cv2.arrowedLine(
        vis,
        ((coarse_pt1[0] + coarse_pt2[0]) // 2, (coarse_pt1[1] + coarse_pt2[1]) // 2),
        ((refined_pt1[0] + refined_pt2[0]) // 2, (refined_pt1[1] + refined_pt2[1]) // 2),
        (0, 255, 255),
        1,
        tipLength=0.25,
    )
    return vis
