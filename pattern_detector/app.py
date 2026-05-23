from __future__ import annotations

import json
import os
import time
from typing import Any

import gradio as gr
import numpy as np

from src.detector import DetectorConfig, PatternDetector


def _rgb_to_bgr(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        raise gr.Error("Please upload both a pattern image and a drawing image.")
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    return image[:, :, ::-1].copy()


def run_detection(
    pattern_image: np.ndarray | None,
    drawing_image: np.ndarray | None,
    threshold: float,
    min_scale: float,
    max_scale: float,
    scale_step: float,
    rotations: str,
    fine_rotation_range: float,
    fine_rotation_step: float,
    stride: int,
    top_k: int,
    nms_iou: float,
    max_processing_dim: int,
    enable_debug: bool,
) -> tuple[np.ndarray | None, str, str]:
    start = time.perf_counter()
    pattern_bgr = _rgb_to_bgr(pattern_image)
    drawing_bgr = _rgb_to_bgr(drawing_image)

    config = DetectorConfig(
        threshold=threshold,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_step=scale_step,
        rotations=rotations,
        fine_rotation_range=fine_rotation_range,
        fine_rotation_step=fine_rotation_step,
        stride=int(stride),
        top_k=int(top_k),
        nms_iou_threshold=nms_iou,
        max_processing_dim=int(max_processing_dim),
        enable_debug=enable_debug,
    )
    detector = PatternDetector(config)
    detections, visualization_bgr = detector.detect(pattern_bgr, drawing_bgr)
    elapsed = time.perf_counter() - start

    result: list[dict[str, Any]] = [det.to_json() for det in detections]
    visualization_rgb = visualization_bgr[:, :, ::-1]
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


with gr.Blocks(title="Zero-shot BOM Pattern Detector") as demo:
    gr.Markdown("# Zero-shot BOM Pattern Detector")
    with gr.Row():
        pattern_input = gr.Image(label="Pattern image", type="numpy")
        drawing_input = gr.Image(label="Drawing image", type="numpy")

    with gr.Row():
        threshold = gr.Slider(0.1, 0.95, value=0.60, step=0.01, label="Confidence threshold")
        nms_iou = gr.Slider(0.05, 0.9, value=0.30, step=0.01, label="NMS IoU threshold")
        max_processing_dim = gr.Slider(800, 4000, value=2500, step=100, label="Max processing dimension")

    with gr.Row():
        min_scale = gr.Slider(0.02, 1.0, value=0.8, step=0.01, label="Min scale")
        max_scale = gr.Slider(0.1, 2.5, value=1.2, step=0.01, label="Max scale")
        scale_step = gr.Slider(0.05, 0.5, value=0.1, step=0.01, label="Scale step")

    with gr.Row():
        rotations = gr.Textbox(value="0,90,180,270", label="Rotations")
        fine_rotation_range = gr.Slider(0, 20, value=0, step=1, label="Fine rotation range")
        fine_rotation_step = gr.Slider(1, 20, value=5, step=1, label="Fine rotation step")

    with gr.Row():
        stride = gr.Slider(2, 32, value=4, step=1, label="Stride")
        top_k = gr.Slider(20, 1000, value=200, step=20, label="Top K per variant")
        enable_debug = gr.Checkbox(value=False, label="Enable debug images")

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
            threshold,
            min_scale,
            max_scale,
            scale_step,
            rotations,
            fine_rotation_range,
            fine_rotation_step,
            stride,
            top_k,
            nms_iou,
            max_processing_dim,
            enable_debug,
        ],
        outputs=[output_image, output_json, runtime],
    )


if __name__ == "__main__":
    launch_demo()
