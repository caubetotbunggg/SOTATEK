"""Hugging Face Space entry point for the ZET pattern detector demo."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_DETECTOR_DIR = _PROJECT_ROOT / "pattern_detector"

os.chdir(_DETECTOR_DIR)
if str(_DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_DIR))

_spec = importlib.util.spec_from_file_location(
    "pattern_detector_gradio_app",
    _DETECTOR_DIR / "app.py",
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

demo = _module.demo


def launch() -> None:
    _module.launch_demo()


if __name__ == "__main__":
    launch()
