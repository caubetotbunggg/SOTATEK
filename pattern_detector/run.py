from __future__ import annotations

import argparse
import time

from src.baseline_zet_detector import draw_baseline_detections, run_tm_baseline
from src.preprocessing import load_image
from src.utils import save_json
from src.visualization import save_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZET-style Template Matching baseline for technical drawings.")
    parser.add_argument("--pattern", required=True, help="Path to query pattern image")
    parser.add_argument("--drawing", required=True, help="Path to drawing image")
    parser.add_argument("--output", default="outputs/zet_tm_result.png", help="Path for visualization image")
    parser.add_argument("--json", default="outputs/zet_tm_result.json", help="Path for JSON detections")
    parser.add_argument("--wide-thr", type=float, default=0.25, help="Template Matching candidate threshold")
    parser.add_argument("--baseline-min-scale", type=float, default=0.05, help="Minimum template scale")
    parser.add_argument("--baseline-max-scale", type=float, default=0.85, help="Maximum template scale")
    parser.add_argument("--baseline-scan-scales", type=int, default=30, help="Number of scales to scan near best scale")
    parser.add_argument("--top-k", type=int, default=15, help="Maximum detections after NMS")
    parser.add_argument("--nms-iou", "--nms-iou-threshold", dest="nms_iou", type=float, default=0.35, help="NMS IoU threshold")
    parser.add_argument("--use-smart-cliff", dest="use_smart_cliff", action="store_true", default=True)
    parser.add_argument("--no-smart-cliff", dest="use_smart_cliff", action="store_false")
    parser.add_argument("--enable-debug", action="store_true", help="Save debug visualization and summary")
    parser.add_argument("--debug-dir", default="outputs/debug", help="Directory for debug output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    pattern_bgr = load_image(args.pattern)
    drawing_bgr = load_image(args.drawing)
    detections = run_tm_baseline(
        drawing_bgr,
        pattern_bgr,
        wide_thr=args.wide_thr,
        nms_iou=args.nms_iou,
        top_k=args.top_k,
        min_scale=args.baseline_min_scale,
        max_scale=args.baseline_max_scale,
        scan_scales=args.baseline_scan_scales,
        use_smart_cliff=args.use_smart_cliff,
        enable_debug=args.enable_debug,
        debug_dir=args.debug_dir,
    )

    visualization = draw_baseline_detections(drawing_bgr, detections)
    elapsed = time.perf_counter() - start
    save_visualization(visualization, args.output)
    save_json(detections, args.json)

    print(f"Found {len(detections)} detections in {elapsed:.2f}s")
    print("idx\tmethod\tconfidence\tx\ty\tw\th\tscale")
    for idx, det in enumerate(detections, start=1):
        print(
            f"{idx}\t{det.get('method', 'zet_tm')}\t{float(det['confidence']):.3f}\t"
            f"{det['x']}\t{det['y']}\t{det['w']}\t{det['h']}\t{float(det['scale']):.3f}"
        )
    print(f"Visualization: {args.output}")
    print(f"JSON: {args.json}")


if __name__ == "__main__":
    main()
