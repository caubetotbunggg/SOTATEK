# Zero-shot BOM Edge Feature Detector

Local Python project for zero-shot pattern detection in black-and-white technical BOM drawings. It takes a query pattern image and a larger drawing image, then returns bounding boxes, confidence scores, scale/rotation metadata, a visualization image, and JSON output.

The pipeline uses classical edge/geometric features only. It does not train, fine-tune, or load semantic/deep learning models.

## How It Works

1. Convert pattern and drawing to grayscale.
2. Threshold with Otsu plus adaptive thresholding.
3. Normalize internal binary convention:
   - foreground strokes = `255`
   - background = `0`
4. Auto-fix polarity so sparse drawing strokes become foreground.
5. Crop the pattern tightly around foreground pixels with configurable padding.
6. Skeletonize pattern and drawing to normalize stroke thickness.
7. Generate scaled and rotated pattern variants.
8. Extract an edge descriptor for each pattern variant:
   - edge density grid
   - HOG-like orientation histogram over skeleton edges
9. Slide a same-size window across the skeletonized drawing.
10. Extract the same descriptor for each window and compute cosine similarity.
11. Keep top-K descriptor candidates per scale/rotation.
12. Validate candidates using:
   - symmetric Chamfer similarity
   - dilated edge F1
   - foreground density score
13. Compute final confidence:

```text
confidence =
  0.35 * descriptor_similarity
  + 0.30 * symmetric_chamfer_similarity
  + 0.25 * edge_f1
  + 0.10 * density_score
```

14. Apply NMS and map boxes back to original drawing coordinates.

## Installation

```bash
cd pattern_detector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI Usage

```bash
python run.py \
  --pattern examples/example1_pattern.png \
  --drawing examples/example1_drawing.png \
  --output outputs/result.png \
  --json outputs/result.json \
  --threshold 0.60 \
  --min-scale 0.8 \
  --max-scale 1.2 \
  --scale-step 0.1 \
  --rotations 0,90,180,270 \
  --stride 4 \
  --top-k 200 \
  --nms-iou 0.30 \
  --enable-debug
```

Useful tuning options:

- `--threshold`: lower for more recall, higher for fewer false positives.
- `--min-scale`, `--max-scale`, `--scale-step`: controls scale search.
- `--rotations`: comma-separated base rotations. Defaults to `0,90,180,270`.
- `--fine-rotation-range`, `--fine-rotation-step`: optional small offsets around base rotations.
- `--stride`: sliding-window stride. Smaller is more accurate but slower.
- `--top-k`: number of descriptor candidates kept per scale/rotation.
- `--max-processing-dim`: resizes very large drawings for CPU runtime while returning original-coordinate boxes.
- `--nms-iou`: suppresses overlapping detections.
- `--enable-debug`: saves preprocessing and debug images.

## Debug Outputs

When `--enable-debug` is set, images are saved to `outputs/debug/`:

- `pattern_binary.png`
- `pattern_cropped.png`
- `pattern_skeleton.png`
- `drawing_binary.png`
- `drawing_skeleton.png`
- `candidates_before_nms.png`
- `final_result.png`
- `descriptor_heatmap_s1_r0.png` when scale `1.0` and rotation `0` are searched

## Gradio Usage

```bash
cd pattern_detector
python app.py
```

Open the local URL printed by Gradio, upload a pattern image and drawing image, tune the sliders, then click **Detect**.

If the default port range is busy:

```bash
GRADIO_SERVER_PORT=8960 python app.py
```

## JSON Output

```json
[
  {
    "x": 120,
    "y": 340,
    "w": 48,
    "h": 32,
    "confidence": 0.873,
    "scale": 1.0,
    "rotation": 0.0,
    "descriptor_similarity": 0.91,
    "chamfer_similarity": 0.84,
    "edge_f1": 0.78,
    "density_score": 0.95
  }
]
```

Coordinates are `x, y, w, h` in the original drawing image coordinate system.

## Project Structure

```text
pattern_detector/
  app.py
  run.py
  requirements.txt
  README.md
  src/
    __init__.py
    detector.py
    preprocessing.py
    edge_features.py
    descriptor_matching.py
    chamfer.py
    validation.py
    nms.py
    visualization.py
    utils.py
  examples/
    .gitkeep
  outputs/
    .gitkeep
```

## Known Limitations

- Very simple patterns may produce many false positives.
- The query should include enough geometric context, not just a tiny line segment or corner.
- Large rotation/scale search increases runtime and false positives.
- Legend symbols may also be detected unless ROI is restricted.
- Heavy occlusion or topology changes can still fail because this is geometric matching, not semantic recognition.

## Recommended Tuning Flow

1. Put `example1_pattern.png`, `example1_drawing.png`, etc. under `examples/`.
2. Start with scale `0.8-1.2`, rotations `0,90,180,270`, stride `4`, threshold `0.60`.
3. If matches are missed, lower threshold, lower stride, or widen scale search.
4. If false positives appear, raise threshold, reduce rotations/scales, or increase query context.
