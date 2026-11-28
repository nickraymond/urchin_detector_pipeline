"""macOS "._" sidecar files: skip them in every listing of data_dir.

On a disk that is not a Mac disk (Nick's data SSD is exFAT), macOS writes a "._<name>" file (AppleDouble:
the file's extra attributes) beside every file it writes. A glob such as "*_full_integer_quant.tflite" then
matches both the model and its sidecar, so a step that wants one file sees two (seen 2026-11-27, convert job 8:
every 8-bit step reported "does not convert" though onnx2tf had made the model).

Standard library only: app/fit/to_int8.py imports this inside the tool environments (D51).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

PREFIX = "._"


def is_sidecar(name: str) -> bool:
    return name.startswith(PREFIX)


def real(paths: Iterable[Path]) -> list[Path]:
    """`paths` without the "._" sidecars, in their given order."""
    return [p for p in paths if not is_sidecar(p.name)]
