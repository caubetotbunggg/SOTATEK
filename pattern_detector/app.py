from __future__ import annotations

import json
import os
import time

import gradio as gr
import numpy as np

from src.baseline_zet_detector import draw_baseline_detections, run_tm_baseline


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
        wide_thr = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Candidate threshold")
        nms_iou = gr.Slider(0.05, 0.9, value=0.35, step=0.01, label="NMS IoU")
        top_k = gr.Slider(1, 100, value=15, step=1, label="Top K")

    with gr.Row():
        min_scale = gr.Slider(0.01, 1.5, value=0.05, step=0.01, label="Min scale")
        max_scale = gr.Slider(0.05, 2.0, value=0.85, step=0.01, label="Max scale")
        scale_step = gr.Slider(0.001, 0.20, value=0.02, step=0.001, label="Scale step")

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


if __name__ == "__main__":
    launch_demo()
