"""Gradio web interface for ZET Template Matching Detector.

Provides an interactive UI for:
- Uploading pattern and drawing images
- Adjusting detection parameters via sliders
- Loading preset examples with optimal scale ranges
- Visualizing detections in real-time
- Exporting results as JSON

Usage:
    python app.py

Then open the URL printed in console (typically http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np

from src.baseline_zet_detector import draw_baseline_detections, run_tm_baseline


# ============================================================================
# Configuration
# ============================================================================

# Load example images
EXAMPLES_DIR = Path(__file__).parent / "examples"
PATTERN_DIR = EXAMPLES_DIR / "pattern"
DRAWING_DIR = EXAMPLES_DIR / "drawing"

# Available examples with specific scale ranges for optimal detection
EXAMPLES = []
if DRAWING_DIR.exists() and PATTERN_DIR.exists():
    drawing_path = DRAWING_DIR / "example1.png"
    
    # Define 3 preset examples using the same drawing with different patterns
    # Each has optimized scale ranges for its specific pattern size
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
    """Convert RGB image to BGR or handle grayscale images.
    
    Args:
        image: Input image (RGB or grayscale)
    
    Returns:
        BGR image suitable for processing by detector
    
    Raises:
        gr.Error: If image is None
    """
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
    """Execute template matching detection with given parameters.
    
    Args:
        pattern_image: Pattern/template to find (RGB numpy array)
        drawing_image: Drawing/scene to search in (RGB numpy array)
        wide_thr: Local maxima threshold for candidate extraction (0-1)
        min_scale: Minimum scale factor to test
        max_scale: Maximum scale factor to test
        scale_step: Granularity of scale search
        top_k: Maximum number of detections to return
        nms_iou: Non-maximum suppression IoU threshold
        use_smart_cliff: Auto-trim low-confidence tail
        enable_debug: Save intermediate debug images
    
    Returns:
        Tuple of:
        - Visualization image (RGB numpy array with bounding boxes)
        - JSON string with detection details
        - Runtime in seconds
    """
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
    """Launch the Gradio web interface.

    On Hugging Face Spaces (SPACE_ID set), uses default Gradio launch settings.
    Locally, honors GRADIO_SERVER_PORT or scans for a free port from 7960.
    """
    if os.getenv("SPACE_ID"):
        # HF Spaces bind to a container port and proxy requests externally.
        # Explicitly bind to 0.0.0.0 and disable `share` to avoid the "localhost not accessible" error.
        port = int(os.getenv("PORT", "7860"))
        demo.launch(server_name="0.0.0.0", server_port=port, share=False)
        return

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


def _gradio_example_rows() -> list[list]:
    """Build gr.Examples rows: pattern, drawing, then all slider/checkbox inputs."""
    rows: list[list] = []
    for ex in EXAMPLES:
        rows.append(
            [
                ex["pattern"],
                ex["drawing"],
                0.4,
                ex["min_scale"],
                ex["max_scale"],
                0.01,
                10,
                0.07,
                True,
                False,
            ]
        )
    return rows


with gr.Blocks(title="SOTATEK — ZET Pattern Detector") as demo:
    gr.Markdown("# SOTATEK — ZET Template Matching Detector")
    gr.Markdown(
        """
    Zero-shot pattern detection for technical drawings.
    Upload a **pattern** (template) and a **drawing**, or pick a quick example below, then click **Detect**.
    Results show bounding boxes on the drawing and JSON metadata.
    """
    )

    # ====================================================================
    # Input Section
    # ====================================================================
    with gr.Row():
        pattern_input = gr.Image(label="Pattern image", type="numpy")
        drawing_input = gr.Image(label="Drawing image", type="numpy")

    # ====================================================================
    # Parameter Section
    # ====================================================================
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

    # ====================================================================
    # Action Button
    # ====================================================================
    run_button = gr.Button("Detect", variant="primary")

    # ====================================================================
    # Output Section
    # ====================================================================
    with gr.Row():
        output_image = gr.Image(label="Detections", type="numpy")
        with gr.Column():
            output_json = gr.Code(label="JSON output", language="json")
            runtime = gr.Textbox(label="Runtime")

    detection_inputs = [
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
    ]

    run_button.click(
        fn=run_detection,
        inputs=detection_inputs,
        outputs=[output_image, output_json, runtime],
    )

    # ====================================================================
    # Quick examples (reviewer-friendly one-click tests)
    # ====================================================================
    if EXAMPLES:
        gr.Markdown("## Quick test examples")
        gr.Markdown(
            "Three presets with tuned scale ranges. Select an example to load images "
            "and parameters, then run detection."
        )
        gr.Examples(
            examples=_gradio_example_rows(),
            inputs=detection_inputs,
            outputs=[output_image, output_json, runtime],
            fn=run_detection,
            cache_examples=False,
            label="Examples",
            examples_per_page=3,
        )

        gr.Markdown("### Load example images only (without running)")
        with gr.Row():
            for example in EXAMPLES:
                with gr.Column():
                    gr.Markdown(f"**{example['name']}**")
                    gr.Markdown(
                        f"Scale: {example['min_scale']:.3f} – {example['max_scale']:.3f}"
                    )
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
