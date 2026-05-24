from __future__ import annotations

import argparse
import time

from src.detector import DetectorConfig, PatternDetector
from src.utils import save_json
from src.visualization import save_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot pattern detection for technical drawings.")
    parser.add_argument("--pattern", required=True, help="Path to query pattern image")
    parser.add_argument("--drawing", required=True, help="Path to drawing image")
    parser.add_argument("--output", default="outputs/result.png", help="Path for visualization image")
    parser.add_argument("--json", default="outputs/result.json", help="Path for JSON detections")
    parser.add_argument("--threshold", type=float, default=0.35, help="Final confidence threshold")
    parser.add_argument("--min-scale", type=float, default=0.05, help="Minimum template scale")
    parser.add_argument("--max-scale", type=float, default=0.30, help="Maximum template scale")
    parser.add_argument("--scale-step", type=float, default=0.02, help="Template scale step")
    parser.add_argument("--rotations", default="0,90,180,270", help="Comma-separated base rotations")
    parser.add_argument("--fine-rotation-range", type=float, default=0.0, help="Optional +/- rotation offsets around base rotations")
    parser.add_argument("--fine-rotation-step", type=float, default=5.0, help="Fine rotation step in degrees")
    parser.add_argument("--stride", type=int, default=2, help="Sliding-window stride in pixels")
    parser.add_argument("--top-k", type=int, default=800, help="Top candidates to keep per scale/rotation")
    parser.add_argument("--nms-iou", "--nms-iou-threshold", dest="nms_iou", type=float, default=0.30, help="NMS IoU threshold")
    parser.add_argument("--max-processing-dim", type=int, default=2500, help="Resize drawing if max side exceeds this")
    parser.add_argument("--chamfer-sigma", type=float, default=8.0, help="Distance-to-similarity sigma")
    parser.add_argument("--max-chamfer-distance", type=float, default=12.0, help="Soft Chamfer distance threshold")
    parser.add_argument("--min-template-coverage", type=float, default=0.25, help="Minimum template edge coverage")
    parser.add_argument("--max-extra-patch-ratio", type=float, default=0.90, help="Maximum extra patch edge ratio")
    parser.add_argument("--validation-dilation-iterations", type=int, default=1, help="Edge validation dilation iterations")
    parser.add_argument("--max-detections", type=int, default=200, help="Maximum boxes after NMS")
    parser.add_argument("--pattern-padding", type=int, default=4, help="Foreground crop padding for pattern")
    parser.add_argument("--enable-debug", action="store_true", help="Save preprocessing and debug visualization images")
    parser.add_argument("--debug-dir", default="outputs/debug", help="Directory for debug images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectorConfig(
        threshold=args.threshold,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        scale_step=args.scale_step,
        rotations=args.rotations,
        fine_rotation_range=args.fine_rotation_range,
        fine_rotation_step=args.fine_rotation_step,
        stride=args.stride,
        top_k=args.top_k,
        nms_iou_threshold=args.nms_iou,
        max_processing_dim=args.max_processing_dim,
        chamfer_sigma=args.chamfer_sigma,
        max_chamfer_distance=args.max_chamfer_distance,
        min_template_coverage=args.min_template_coverage,
        max_extra_patch_ratio=args.max_extra_patch_ratio,
        validation_dilation_iterations=args.validation_dilation_iterations,
        max_detections=args.max_detections,
        pattern_padding=args.pattern_padding,
        enable_debug=args.enable_debug,
        debug_dir=args.debug_dir,
    )

    start = time.perf_counter()
    detector = PatternDetector(config)
    detections, visualization = detector.detect_from_paths(args.pattern, args.drawing)
    elapsed = time.perf_counter() - start

    result = [det.to_json() for det in detections]
    save_visualization(visualization, args.output)
    save_json(result, args.json)

    print(f"Found {len(result)} detections in {elapsed:.2f}s")
    print(f"Debug counts: {detector.last_debug_counts}")
    print(f"Visualization: {args.output}")
    print(f"JSON: {args.json}")


if __name__ == "__main__":
    main()
