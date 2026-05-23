"""Small shared utilities for file IO and coordinate handling."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for a file path when it is missing."""
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: str | Path) -> None:
    """Write pretty JSON with stable UTF-8 encoding."""
    ensure_parent_dir(path)

    def default(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def round_float(value: float, digits: int = 3) -> float:
    return round(float(value), digits)
