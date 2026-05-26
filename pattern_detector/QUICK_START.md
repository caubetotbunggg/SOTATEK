# Quick Reference Guide

## Installation (1 minute)

### Using pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Using conda
```bash
conda env create -f environment.yml
conda activate pattern-detector
```

## Usage (3 different ways)

### 1️⃣ Web UI (Easiest)
```bash
python app.py
```
- Open browser to http://localhost:7860
- Upload images via GUI
- Adjust sliders in real-time
- Click "Detect" to run
- Download results as JSON

### 2️⃣ Command Line
```bash
python run.py \
  --pattern examples/pattern/example1_pattern.png \
  --drawing examples/drawing/example1.png \
  --output outputs/result.png \
  --json outputs/result.json
```

### 3️⃣ Python Script
```python
import cv2
from src.baseline_zet_detector import run_tm_baseline, draw_baseline_detections

# Load images
drawing = cv2.imread("drawing.png")
pattern = cv2.imread("pattern.png")

# Run detection
detections = run_tm_baseline(
    drawing, pattern,
    wide_thr=0.4,
    min_scale=0.08,
    max_scale=0.14,
    scale_step=0.01,
    top_k=10,
    nms_iou=0.07,
)

# Visualize
result = draw_baseline_detections(drawing, detections)
cv2.imwrite("result.png", result)

# Print results
for det in detections:
    print(f"Found at ({det['x']}, {det['y']}), "
          f"confidence: {det['confidence']:.3f}")
```

## Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `wide_thr` | 0.4 | 0.0-1.0 | Candidate threshold (↓ = more candidates) |
| `min_scale` | 0.08 | 0.01-1.5 | Minimum pattern size |
| `max_scale` | 0.14 | 0.05-2.0 | Maximum pattern size |
| `scale_step` | 0.01 | 0.001-0.20 | Scale search granularity |
| `top_k` | 10 | 1-100 | Max detections to return |
| `nms_iou` | 0.07 | 0.05-0.9 | NMS suppression threshold |

## Tuning Guide

### Too many false positives?
```python
wide_thr=0.5,  # Raise threshold
top_k=5,       # Fewer detections
nms_iou=0.1,   # Stricter NMS
```

### Missing detections?
```python
wide_thr=0.2,  # Lower threshold
top_k=20,      # More detections
min_scale=0.05, max_scale=1.0,  # Wider search
```

### Slow performance?
```python
scale_step=0.05,  # Coarser search
min_scale=0.08, max_scale=0.14,  # Narrow range
top_k=5,  # Fewer detections
```

## Output Format

JSON output includes:
```json
{
  "method": "zet_tm",
  "num_detections": 3,
  "detections": [
    {
      "x": 120,              // Left position
      "y": 85,               // Top position
      "w": 32,               // Width
      "h": 28,               // Height
      "confidence": 0.873,   // Overall score (0-1)
      "scale": 0.09,         // Applied scale factor
      "rotation": 0.0,       // Rotation angle (0/45/90/135)
      "tm_score": 0.91,      // NCC correlation
      "foreground_iou": 0.84,// Foreground overlap
      "diff_similarity": 0.78 // Pixel similarity
    }
  ]
}
```

## Common Issues

### "Image is None"
→ Make sure both images are uploaded

### No detections found
→ Lower `wide_thr` or expand `min_scale`/`max_scale`

### Too slow
→ Increase `scale_step` or narrow scale range

### Port already in use
```bash
GRADIO_SERVER_PORT=8000 python app.py
```

## Debug Mode

Enable debug outputs:
```python
detections = run_tm_baseline(
    drawing, pattern,
    enable_debug=True,
    debug_dir="outputs/debug"
)
```

This saves intermediate images to `outputs/debug/`:
- Pattern preprocessing
- Mask generation
- Score maps
- Candidates before/after NMS
- Final detections

## File Structure
```
pattern_detector/
├── app.py              # Web UI
├── run.py              # CLI
├── src/
│   └── baseline_zet_detector.py  # Core algorithm
├── examples/
│   ├── drawing/
│   └── pattern/
└── outputs/
    └── debug/
```

## Next Steps

1. **Quick Test**: Run `python app.py`, load an example
2. **Fine-tune**: Adjust parameters based on results
3. **Batch Process**: Use Python API in a loop
4. **Deploy**: Use CLI in production scripts
5. **Debug**: Enable `enable_debug=True` if issues arise

## Need Help?

- See `README.md` for detailed documentation
- Check `src/baseline_zet_detector.py` for algorithm details
- Review examples in `examples/` folder
- Run with `enable_debug=True` to visualize internals
