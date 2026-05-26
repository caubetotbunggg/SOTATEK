from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np

from src.baseline_zet_detector import draw_baseline_detections, run_tm_baseline


# Load example images
EXAMPLES_DIR = Path(__file__).parent / "examples"
PATTERN_DIR = EXAMPLES_DIR / "pattern"
DRAWING_DIR = EXAMPLES_DIR / "drawing"

# Available examples with specific scale ranges
EXAMPLES = []
if DRAWING_DIR.exists() and PATTERN_DIR.exists():
    drawing_path = DRAWING_DIR / "example1.png"
    
    # Define 3 examples using the same drawing with different scale ranges
    example_configs = [
        {
            "name": "Example 1",
            "pattern_name": "example1_pattern.png",
            "min_scale": 0.08,
            "max_scale": 0.09,
        },
        {
            "name": "Example 2",
            "pattern_name": "example2_pattern.png",
            "min_scale": 0.10,
            "max_scale": 0.14,
        },
        {
            "name": "Example 3",
            "pattern_name": "example3_pattern.png",
            "min_scale": 0.04,
            "max_scale": 1.0,
        },
    ]
    
    if drawing_path.exists():
        for config in example_configs:
            pattern_path = PATTERN_DIR / config["pattern_name"]
            if pattern_path.exists():
                EXAMPLES.append({
                    "name": config["name"],
                    "drawing": str(drawing_path),
                    "pattern": str(pattern_path),
                    "min_scale": config["min_scale"],
                    "max_scale": config["max_scale"],
                })


def _rgb_to_bgr(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        raise gr.Error("Please upload both a pattern image and a drawing image.")
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    return image[:, :, ::-1].copy()


def run_detection(
    pattern_image: np.ndarray | None,
    drawing_image: np.ndarray | None,
    wide_thr: float,
    min_scale: float,
    max_scale: float,
    scale_step: float,
    top_k: int,
    nms_iou: float,
    use_smart_cliff: bool,
    enable_debug: bool,
) -> tuple[np.ndarray | None, str, str]:
    start = time.perf_counter()
    pattern_bgr = _rgb_to_bgr(pattern_image)
    drawing_bgr = _rgb_to_bgr(drawing_image)

    detections = run_tm_baseline(
        drawing_bgr,
        pattern_bgr,
        wide_thr=wide_thr,
        nms_iou=nms_iou,
        top_k=int(top_k),
        min_scale=min_scale,
        max_scale=max_scale,
        scale_step=scale_step,
        use_smart_cliff=use_smart_cliff,
        enable_debug=enable_debug,
    )

    elapsed = time.perf_counter() - start
    visualization_rgb = draw_baseline_detections(drawing_bgr, detections)[:, :, ::-1]
    result = {
        "method": "zet_tm",
        "num_detections": len(detections),
        "detections": detections,
    }
    return visualization_rgb, json.dumps(result, indent=2), f"{elapsed:.2f} seconds"


def launch_demo() -> None:
    configured_port = os.getenv("GRADIO_SERVER_PORT")
    if configured_port:
        demo.launch(server_port=int(configured_port))
        return

    try:
        demo.launch()
    except OSError as first_error:
        for port in range(7960, 8060):
            try:
                demo.launch(server_port=port)
                return
            except OSError:
                continue
        raise first_error


with gr.Blocks(title="ZET Template Matching Detector") as demo:
    gr.Markdown("# ZET Template Matching Detector")

    with gr.Row():
        pattern_input = gr.Image(label="Pattern image", type="numpy")
        drawing_input = gr.Image(label="Drawing image", type="numpy")

    with gr.Row():
        wide_thr = gr.Slider(0.0, 1.0, value=0.4, step=0.01, label="Candidate threshold")
        nms_iou = gr.Slider(0.05, 0.9, value=0.07, step=0.01, label="NMS IoU")
        top_k = gr.Slider(1, 100, value=10, step=1, label="Top K")

    with gr.Row():
        min_scale = gr.Slider(0.01, 1.5, value=0.08, step=0.01, label="Min scale")
        max_scale = gr.Slider(0.05, 2.0, value=0.14, step=0.01, label="Max scale")
        scale_step = gr.Slider(0.001, 0.20, value=0.01, step=0.001, label="Scale step")

    with gr.Row():
        use_smart_cliff = gr.Checkbox(value=True, label="Use smart cliff")
        enable_debug = gr.Checkbox(value=False, label="Save debug output")

    run_button = gr.Button("Detect", variant="primary")

    with gr.Row():
        output_image = gr.Image(label="Detections", type="numpy")
        with gr.Column():
            output_json = gr.Code(label="JSON output", language="json")
            runtime = gr.Textbox(label="Runtime")

    run_button.click(
        fn=run_detection,
        inputs=[
            pattern_input,
            drawing_input,
            wide_thr,
            min_scale,
            max_scale,
            scale_step,
            top_k,
            nms_iou,
            use_smart_cliff,
            enable_debug,
        ],
        outputs=[output_image, output_json, runtime],
    )
    
    # Add preset examples
    if EXAMPLES:
        gr.Markdown("## Preset Examples")
        with gr.Row():
            for example in EXAMPLES:
                with gr.Column():
                    gr.Markdown(f"### {example['name']}")
                    example_button = gr.Button(f"Load {example['name']}")
                    
                    def make_load_example(ex):
                        def load_example():
                            from PIL import Image
                            pattern = np.array(Image.open(ex["pattern"]))
                            drawing = np.array(Image.open(ex["drawing"]))
                            return [
                                pattern,
                                drawing,
                                ex.get("min_scale", 0.08),
                                ex.get("max_scale", 0.14),
                            ]
                        return load_example
                    
                    example_button.click(
                        fn=make_load_example(example),
                        outputs=[pattern_input, drawing_input, min_scale, max_scale],
                    )


if __name__ == "__main__":
    launch_demo()
