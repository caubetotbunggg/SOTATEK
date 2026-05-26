# ZET Template Matching Detector

A zero-shot template matching detector for finding patterns in technical drawings using classical computer vision techniques (no deep learning required).

## Overview

This project detects pattern occurrences in large technical drawing images using:
- **Template Matching**: Normalized Cross-Correlation (NCC) with multi-scale search
- **Rotation Support**: Detects patterns at 0°, 45°, 90°, 135° rotations
- **Foreground Aware**: Uses binary foreground masks for robust matching
- **Multi-metric Scoring**: Combines NCC score, foreground IoU, and pixel difference similarity

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Gradio Web UI](#gradio-web-ui)
  - [Command Line](#command-line)
  - [Python API](#python-api)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Performance Tuning](#performance-tuning)

## Features

✅ **Zero-shot Detection**: No training required  
✅ **Multi-scale Search**: Detects patterns at different sizes  
✅ **Rotation Invariance**: Supports pattern rotations (0°, 45°, 90°, 135°)  
✅ **Binary Drawing Handling**: Works with black-and-white technical drawings  
✅ **Confidence Scoring**: Combines multiple similarity metrics  
✅ **NMS Filtering**: Removes duplicate/overlapping detections  
✅ **Web UI**: Gradio interface for interactive testing  
✅ **Debug Mode**: Saves intermediate visualization images  

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone or download the repository:**

```bash
cd pattern_detector
```

2. **Create virtual environment (recommended):**

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n pattern-detector python=3.10
conda activate pattern-detector
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Web UI (Recommended for Beginners)

```bash
python app.py
```

Then open the URL printed in console (usually `http://localhost:7860`).

**Features:**
- Upload or paste pattern and drawing images
- Adjust detection parameters with sliders in real-time
- View results with bounding boxes
- Export detections as JSON
- Load preset examples with one click

### 2. Command Line

```bash
python run.py \
  --pattern examples/pattern/example1_pattern.png \
  --drawing examples/drawing/example1.png \
  --output outputs/result.png \
  --json outputs/result.json \
  --wide-thr 0.4 \
  --min-scale 0.08 \
  --max-scale 0.14 \
  --scale-step 0.01 \
  --top-k 10 \
  --nms-iou 0.07
```

### 3. Python API

```python
import cv2
from src.baseline_zet_detector import run_tm_baseline, draw_baseline_detections

# Load images
drawing = cv2.imread("drawing.png")
pattern = cv2.imread("pattern.png")

# Run detection
detections = run_tm_baseline(
    drawing_bgr=drawing,
    pattern_bgr=pattern,
    wide_thr=0.4,
    min_scale=0.08,
    max_scale=0.14,
    scale_step=0.01,
    top_k=10,
    nms_iou=0.07,
    use_smart_cliff=True,
    enable_debug=False,
)

# Visualize
result = draw_baseline_detections(drawing, detections)
cv2.imwrite("result.png", result)

# Access results
for det in detections:
    print(f"Found at ({det['x']}, {det['y']}), "
          f"size {det['w']}x{det['h']}, "
          f"confidence: {det['confidence']:.3f}, "
          f"rotation: {det['rotation']}°")
```

## Usage

### Gradio Web UI

#### Basic Usage

1. **Upload Images:**
   - Pattern: The template/symbol to find
   - Drawing: The larger document/image to search in

2. **Adjust Parameters:**
   - **Candidate threshold**: Local maxima threshold (0.0-1.0)
     - Lower = more candidates, higher sensitivity
     - Higher = fewer false positives
   - **NMS IoU**: Non-maximum suppression threshold (0.05-0.9)
     - Higher = keep more overlapping detections
   - **Top K**: Maximum detections to return (1-100)
   - **Min/Max scale**: Scale range relative to pattern size
     - Min scale: smallest pattern size to detect
     - Max scale: largest pattern size to detect
   - **Scale step**: Granularity of scale search
     - Smaller = finer search, slower
     - Larger = coarser search, faster

3. **Advanced:**
   - **Use smart cliff**: Automatically trim low-confidence detections
   - **Save debug output**: Generate intermediate visualization images

#### Preset Examples

Click any preset example button to:
- Load pre-configured images
- Auto-adjust scale ranges for optimal detection

Current presets (all use same drawing):
- **Example 1**: Pattern 1, scale 0.08-0.09
- **Example 2**: Pattern 2, scale 0.10-0.14
- **Example 3**: Pattern 3, scale 0.04-1.0

### Command Line

See `python run.py --help` for all options.

Key parameters:

- `--pattern`: Path to pattern image
- `--drawing`: Path to drawing image
- `--output`: Output visualization image path
- `--json`: Output JSON detections path
- `--wide-thr`: Candidate threshold (default: 0.4)
- `--min-scale`, `--max-scale`: Scale range (defaults: 0.05-0.85)
- `--scale-step`: Scale search granularity (default: None = linspace)
- `--top-k`: Keep top K detections (default: 15)
- `--nms-iou`: NMS threshold (default: 0.35)
- `--enable-debug`: Save debug images

### Python API

```python
from src.baseline_zet_detector import run_tm_baseline

detections = run_tm_baseline(
    drawing_bgr,           # BGR image (numpy array)
    pattern_bgr,           # BGR image (numpy array)
    wide_thr=0.25,         # Candidate threshold
    nms_iou=0.35,          # NMS IoU threshold
    top_k=15,              # Max detections
    min_scale=0.05,        # Min scale factor
    max_scale=0.85,        # Max scale factor
    scale_step=None,       # None = linspace, or fixed step
    use_smart_cliff=True,  # Auto-trim low scores
    enable_debug=False,    # Save debug outputs
    debug_dir="outputs/debug",
    rotation_angles=[0.0, 45.0, 90.0, 135.0],  # Rotation angles in degrees
)
```

## Configuration

### Default Parameters

```
Candidate threshold (wide_thr): 0.4
NMS IoU (nms_iou): 0.07
Top K: 10
Min scale: 0.08
Max scale: 0.14
Scale step: 0.01
Use smart cliff: True
```

### Tuning Guide

**Increase Recall (find more patterns):**
- ↓ Lower `wide_thr` (candidate threshold)
- ↓ Lower `min_scale` to detect smaller instances
- ↑ Increase `max_scale` to detect larger instances
- ↓ Lower `scale_step` for finer granularity
- ↓ Lower `nms_iou` threshold

**Decrease False Positives:**
- ↑ Raise `wide_thr` (candidate threshold)
- ↑ Increase `top_k` and rely on smart_cliff
- ↑ Narrow scale range (tighter min/max)
- ↑ Increase `nms_iou` threshold

**Speed Optimization:**
- ↑ Increase `scale_step` (coarser search)
- ↓ Reduce `top_k`
- ↑ Raise `wide_thr` (fewer candidates)
- Use narrower scale range

## Output Format

### JSON Output

```json
{
  "method": "zet_tm",
  "num_detections": 3,
  "detections": [
    {
      "x": 120,
      "y": 85,
      "w": 32,
      "h": 28,
      "confidence": 0.873,
      "scale": 0.09,
      "rotation": 0.0,
      "tm_score": 0.91,
      "foreground_iou": 0.84,
      "diff_similarity": 0.78
    },
    {
      "x": 450,
      "y": 200,
      "w": 32,
      "h": 28,
      "confidence": 0.782,
      "scale": 0.09,
      "rotation": 90.0,
      "tm_score": 0.88,
      "foreground_iou": 0.79,
      "diff_similarity": 0.71
    }
  ]
}
```

### Field Descriptions

- **x, y**: Top-left corner of bounding box (pixels)
- **w, h**: Width and height of bounding box (pixels)
- **confidence**: Overall confidence score (0.0-1.0)
  - = 0.50 × tm_score + 0.30 × foreground_iou + 0.20 × diff_similarity
- **scale**: Scale factor applied to pattern
- **rotation**: Detected rotation angle (0, 45, 90, or 135 degrees)
- **tm_score**: Template matching correlation score (0.0-1.0)
- **foreground_iou**: IoU of foreground regions (0.0-1.0)
- **diff_similarity**: Pixel difference similarity (0.0-1.0)

### Visualization Output

Generated images show:
- Original drawing with detected patterns
- Red bounding boxes around detections
- Confidence scores as labels

## Project Structure

```
pattern_detector/
├── app.py                          # Gradio web interface
├── run.py                          # Command-line interface
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── baseline_zet_detector.py   # Core template matching algorithm
│   ├── preprocessing.py            # Image preprocessing utilities
│   ├── visualization.py            # Visualization functions
│   ├── utils.py                    # General utilities
│   ├── detector.py                 # Advanced detector (not used in baseline)
│   ├── edge_features.py            # Edge feature extraction
│   ├── descriptor_matching.py      # Descriptor matching (advanced)
│   ├── chamfer.py                  # Chamfer distance (advanced)
│   ├── validation.py               # Validation logic (advanced)
│   └── nms.py                      # NMS implementation
│
├── examples/
│   ├── drawing/
│   │   └── example1.png            # Sample drawing image
│   └── pattern/
│       ├── example1_pattern.png    # Sample pattern 1
│       ├── example2_pattern.png    # Sample pattern 2
│       └── example3_pattern.png    # Sample pattern 3
│
└── outputs/                        # Generated outputs
    ├── result.png                  # Visualization
    ├── result.json                 # Detection JSON
    └── debug/                      # Debug outputs (if enabled)
```

## Examples

### Example 1: Basic Detection

```python
import cv2
from src.baseline_zet_detector import run_tm_baseline, draw_baseline_detections

# Load images
drawing = cv2.imread("examples/drawing/example1.png")
pattern = cv2.imread("examples/pattern/example1_pattern.png")

# Simple detection with defaults
detections = run_tm_baseline(drawing, pattern)

# Save result
result = draw_baseline_detections(drawing, detections)
cv2.imwrite("outputs/example1_result.png", result)

print(f"Found {len(detections)} patterns")
for i, det in enumerate(detections):
    print(f"  {i+1}. At ({det['x']}, {det['y']}), "
          f"confidence: {det['confidence']:.3f}")
```

### Example 2: Fine-tuned Detection

```python
import cv2
from src.baseline_zet_detector import run_tm_baseline

drawing = cv2.imread("examples/drawing/example1.png")
pattern = cv2.imread("examples/pattern/example2_pattern.png")

# Fine-tuned for this specific pattern
detections = run_tm_baseline(
    drawing,
    pattern,
    wide_thr=0.35,          # Lower threshold for more candidates
    min_scale=0.10,         # This pattern is medium-sized
    max_scale=0.14,
    scale_step=0.005,       # Fine granularity
    top_k=20,               # Allow more detections
    nms_iou=0.05,           # Strict NMS
    use_smart_cliff=True,   # Auto-trim low scores
)

print(f"Found {len(detections)} patterns with fine tuning")
```

### Example 3: Batch Processing

```python
import cv2
from pathlib import Path
from src.baseline_zet_detector import run_tm_baseline, draw_baseline_detections

# Process multiple patterns on same drawing
drawing = cv2.imread("examples/drawing/example1.png")
pattern_dir = Path("examples/pattern")

for pattern_file in sorted(pattern_dir.glob("*.png")):
    print(f"\nProcessing {pattern_file.name}...")
    pattern = cv2.imread(str(pattern_file))
    
    detections = run_tm_baseline(
        drawing, pattern,
        min_scale=0.05,
        max_scale=1.0,
        top_k=15,
    )
    
    result = draw_baseline_detections(drawing, detections)
    output_path = f"outputs/{pattern_file.stem}_result.png"
    cv2.imwrite(output_path, result)
    print(f"  Found {len(detections)} matches → {output_path}")
```

## Performance Tuning

### Algorithm Overview

1. **Preprocessing**: Convert to grayscale, enhance with CLAHE
2. **Scale Sweep**: Generate pattern at multiple scales
3. **Rotation Variants**: Create rotated versions (0°, 45°, 90°, 135°)
4. **Template Matching**: 
   - Compute foreground mask from template
   - Compute NCC score map using mask
   - Extract local maxima above threshold
5. **Multi-metric Scoring**:
   - Template Matching (NCC) score: 50% weight
   - Foreground IoU: 30% weight
   - Pixel difference similarity: 20% weight
6. **NMS**: Remove overlapping detections
7. **Smart Cliff**: Auto-trim low-confidence tail

### Speed Considerations

- **Scale search dominates runtime**
  - Fewer scales = faster
  - Use narrow scale range if size is known
  
- **Rotation search**
  - 4 rotations (0°, 45°, 90°, 135°) is balanced
  - Remove rotations if pattern has no symmetry
  
- **Image resolution**
  - Larger images = slower
  - Consider downsampling very large inputs
  
- **Pattern size**
  - Larger patterns = faster matching
  - Tiny patterns (<10px) may be slow due to interpolation

### Memory Considerations

- Typical case (drawing ~1000x1000, pattern ~50x50): ~50MB RAM
- Very large drawings or patterns may need > 500MB
- Use appropriate data types (uint8 for binary images)

## Troubleshooting

### Patterns not detected?

1. **Lower the threshold**
   - Try `wide_thr=0.2` or lower
   - Check if pattern exists in drawing

2. **Expand scale range**
   - If pattern size unknown, use `min_scale=0.05, max_scale=1.0`
   - But this is slower

3. **Increase top_k**
   - Allow more candidates before NMS
   - Try `top_k=50` or higher

4. **Disable smart_cliff**
   - Try `use_smart_cliff=False`
   - See all candidates

### Too many false positives?

1. **Raise the threshold**
   - Try `wide_thr=0.6` or higher

2. **Narrow scale range**
   - If you know approximate pattern size

3. **Increase nms_iou**
   - Stricter duplicate suppression
   - Try `nms_iou=0.1` or higher

4. **Add more context to pattern**
   - Include surrounding symbols/lines
   - Avoid isolated line segments

### Slow performance?

1. **Use coarser scale step**
   - Try `scale_step=0.05` instead of 0.01

2. **Narrow scale range**
   - Use exact or near-exact size if known

3. **Reduce top_k**
   - Try `top_k=5` instead of 15

4. **Disable debug**
   - `enable_debug=False` skips visualization generation

## Requirements

See `requirements.txt`:

```
opencv-python>=4.5.0
numpy>=1.21.0
scipy>=1.7.0
scikit-image>=0.18.0
gradio>=3.0.0
pillow>=8.0.0
```

## License

This project is part of the SOTATEK pattern detection research.

## Citation

If you use this work, please cite:

```bibtex
@software{zet_tm_detector,
  title={ZET Template Matching Detector},
  author={SOTATEK},
  year={2024},
  note={Zero-shot pattern detection in technical drawings}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Add tests/documentation
5. Submit a pull request

## Support

For issues, questions, or suggestions:
- Check existing issues on GitHub
- Create a new issue with minimal reproducible example
- Include parameter values and image samples if possible
