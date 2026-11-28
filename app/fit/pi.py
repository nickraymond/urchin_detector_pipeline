"""Pi Zero 2W step (S6 bite 3c, D52). The Pi has no NPU and no compiler: its file is the 8-bit TFLite
itself, run on the CPU by a small runtime. On the laptop this step can only show three things: the file
loads and runs under the runtime (LiteRT, the first candidate), which operators it uses and whether any
is not a built-in one, and the file's size. Speed and memory on the Pi are S7's, on the bench Pi.

Limit (source): 512 MB of memory on the Pi Zero 2W (`docs/source/build2_spec.md`, stage 11's table).
There is no model-area limit: the file sits on the SD card. The laptop's time and peak memory are
recorded as laptop figures, never as estimates for the Pi.

Not done here: ONNX Runtime and ncnn (D52 names them as the other candidates). Each needs its own
8-bit route (ONNX QDQ quantisation; pnnx), which is a different model file from the one the two NPUs
get; they are compared on the bench Pi in S7.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from app.fit import reports
from app.fit.pi_run import TOOL_FAULT
from app import ORT_OFF

MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
RUNTIME_ENV = "convert-venv"
REPO = Path(__file__).resolve().parents[2]


QDQ_NAME = "model_qdq.onnx"


def qdq(python: Path, onnx_file: Path, out: Path) -> dict:
    """The Pi's second file (S7 bite 3c; Lead call, lead 2): the 8-bit QDQ ONNX file for ONNX Runtime, from the
    candidate's own `model.onnx`. Its failure is recorded here and never changes the LiteRT result."""
    if not onnx_file.is_file():
        return {"file": None, "runtime": "ONNX Runtime", "error": f"no {onnx_file.name} beside the 8-bit file to make it from"}
    made = out / QDQ_NAME
    made.unlink(missing_ok=True)
    done = subprocess.run([str(python), "-m", "app.fit.to_qdq", str(onnx_file), str(made)], capture_output=True,
                          text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1", **ORT_OFF})
    (out / "qdq.log").write_text(done.stdout + done.stderr)
    try:
        ran = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 and made.is_file() else None
    except (ValueError, IndexError):
        ran = None
    if ran is None:
        return {"file": None, "runtime": "ONNX Runtime",
                "error": (f"the QDQ step ended with exit code {done.returncode}.\n"
                          + reports.error_lines(done.stdout + done.stderr)).strip()}
    return {"file": QDQ_NAME, "file_bytes": made.stat().st_size, "runtime": ran["runtime"], "laptop_ms": ran["laptop_ms"],
            "runs": ran["runs"], "calibration": ran["calibration"], "quantization": ran["quantization"],
            "operator_types": ran["operator_types"],
            "note": "for ONNX Runtime on the Pi (D85); laptop time, one thread, says nothing about the Pi"}


def compile(tflite: Path, out: Path, job: dict) -> dict:
    from app.fit.convert_worker import ToolsMissing
    python = Path(job["tools_dir"]) / RUNTIME_ENV / "bin" / "python"
    if not python.is_file():
        raise ToolsMissing(f"Tool environment not found: {python}. Fix: set the tools up (DESIGN D51).")
    done = subprocess.run([str(python), "-m", "app.fit.pi_run", str(tflite)], capture_output=True, text=True,
                          cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1", **ORT_OFF})
    (out / "compile.log").write_text(done.stdout + done.stderr)
    if done.returncode == TOOL_FAULT:                # the runtime could not be used: a job error, not a result
        raise ToolsMissing("LiteRT could not be used in " + RUNTIME_ENV + ": "
                           + (done.stderr.strip().splitlines() or ["no message"])[-1]
                           + ". Fix: set the tools up again (DESIGN D51); this says nothing about the model.")
    base = {"tool": "LiteRT on the laptop (no compiler: the Pi runs the 8-bit file on its CPU)",
            "limit_bytes": None, "limit_source": "no model-area limit: the file sits on the SD card",
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "memory_limit_source": "Pi Zero 2W, 512 MB (build2_spec.md, stage 11)", "estimate_ms": None}
    try:
        ran = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else None
    except (ValueError, IndexError):
        ran = None
    if ran is None:
        # LiteRT can end the process without a Python error (an abort while it prepares the model): say so
        how = (f"LiteRT ended the run with exit code {done.returncode} and no result"
               + (" (the process was aborted while preparing or running the model)" if done.returncode in (-6, 134) else ""))
        return {**base, "converts": False, "failing_layers": [],
                "error": (how + ".\n" + reports.error_lines(done.stdout + done.stderr)).strip()}
    shutil.copyfile(tflite, out / tflite.name)
    return {**base, "converts": True, "second_file": qdq(python, tflite.parent / "model.onnx", out), "tool": ran["runtime"] + " on the laptop (the Pi runs the 8-bit file on its CPU)",
            "file": tflite.name, "file_bytes": tflite.stat().st_size, "memory_bytes": None,
            "placement": {"npu": 0, "hybrid": 0, "cpu": ran["operators"], "total": ran["operators"],
                          "unit": "operators"},
            "failing_layers": [{"name": n, "type": n, "why": "not a built-in LiteRT operator: needs an extra "
                                "delegate or library on the Pi"} for n in ran["not_builtin"]],
            "laptop": {"ms": ran["laptop_ms"], "runs": ran["runs"], "peak_bytes": ran["laptop_peak_bytes"],
                       "note": "measured on the laptop, one thread; peak memory is the whole Python process; says "
                               "nothing about the Pi"},
            "operator_types": ran["operator_types"], "io": {"input": ran["input"], "outputs": ran["outputs"]}}
