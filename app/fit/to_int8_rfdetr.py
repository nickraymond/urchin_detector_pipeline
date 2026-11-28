"""The 8-bit step for the RF-DETR family (S6 bite 2e): `python -m app.fit.to_int8_rfdetr <recipe.json> <int8 dir>`.
Runs INSIDE the family's tool environment (D51), like app/fit/to_int8.py, and leaves the same files:
model_int8.tflite (named last), int8.json, and the ONNX file RF-DETR wrote.

Why its own route. With the known-good route (our ONNX export, onnxsim, onnx2tf 1.28.8 `-oiqt`) Atto, Femto and
Pico do not convert, and Nano's 8-bit file needs operators plain LiteRT does not have (DESIGN §Bench results).
RF-DETR ships its own export: it freezes the backbone's position embeddings to the export size before the ONNX
trace, rewrites GridSample, and calls onnx2tf 2.4 with its own settings (`rfdetr/export/_tflite/exporter.py`).
With it all six candidates convert (measured, bite 2e).

What RF-DETR's exporter does NOT make is the file the cameras' compilers need. Its "int8" is dynamic-range (8-bit
weights, float activations); its docstring: "Static (full-integer) INT8 is not supported and raises ValueError".
So this step takes the SavedModel the exporter leaves and runs TensorFlow's own converter in full-integer mode
on it, with the seeded noise calibration of D55. That last step is ours, not Roboflow's, and nobody has measured
what it costs in accuracy (S11's question for a trained model).

Every network connection and name lookup is refused while this runs (the Roboflow-licensed sizes: D59; the
others: nothing to fetch for an untrained model). No weights, no frames.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import sys
import time
from importlib import metadata
from pathlib import Path

from app import sidecars
from app.fit.to_int8 import CALIB_IMAGES, SEED, code_version, repo_version, sha256, write_licence

ROUTE = "rfdetr-own"


def tf_files(tf_dir: Path, pattern: str) -> list[Path]:
    """Files in RF-DETR's export folder, without macOS "._" sidecars (exFAT data disk)."""
    return sorted(sidecars.real(tf_dir.glob(pattern)))


def refuse_network() -> list[str]:
    """Make every connection and name lookup fail; returns the list the attempts are written to."""
    attempts: list[str] = []

    def refuse(*a, **k):
        attempts.append(str(a[:2])[:200])
        raise OSError("network refused: the RF-DETR 8-bit step runs offline")

    socket.socket.connect = socket.socket.connect_ex = refuse
    socket.getaddrinfo = socket.create_connection = refuse
    return attempts


def main(argv: list[str]) -> None:
    recipe = json.loads(Path(argv[1]).read_text())
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    arch = recipe["arch"]
    t0 = time.time()
    write_licence(arch, out.parent)
    if recipe.get("weights"):
        raise SystemExit("Trained weights are not handled yet (S11).")
    attempts = refuse_network()
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # TensorFlow before anything loads ONNX's C extension, or the conversion deadlocks (rfdetr/export/_backend.py)
    from rfdetr.export._backend import preload_tensorflow_before_onnx
    preload_tensorflow_before_onnx()
    import numpy as np
    import tensorflow as tf
    import torch
    from app import modeldefs

    version = code_version(arch)
    torch.manual_seed(SEED)
    h, w = arch["input"]
    problems = modeldefs.check(arch)
    if problems:
        raise SystemExit("Cannot build this model: " + "; ".join(problems) + ".")
    family = importlib.import_module(modeldefs.FAMILIES[arch["family"]])
    tf_dir = out / "tf"
    shutil.rmtree(tf_dir, ignore_errors=True)
    print(f"[to_int8_rfdetr] RF-DETR's own export to TFLite ({version}, onnx2tf {metadata.version('onnx2tf')})",
          flush=True)
    family.build_model(arch).export(output_dir=str(tf_dir), format="tflite", quantization="int8", verbose=False,
                                    shape=(h, w))
    made = [p.name for p in tf_files(tf_dir, "*.tflite")]
    if not (tf_dir / "saved_model.pb").is_file():
        raise SystemExit(f"RF-DETR's export left no SavedModel to make the 8-bit file from (it wrote: {made}).")

    print("[to_int8_rfdetr] full-integer 8-bit file from the SavedModel (TensorFlow's converter)", flush=True)
    calib = np.random.default_rng(SEED).uniform(0, 255, (CALIB_IMAGES, h, w, 3)).astype(np.float32)   # D55
    conv = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([calib[i:i + 1]] for i in range(len(calib)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    part = out / "model_int8.tflite.part"            # the final name appears last and whole (D33; QE PR 76)
    part.write_bytes(conv.convert())

    onnx_files = sorted(tf_files(tf_dir, "*.onnx"), key=lambda p: p.stat().st_size)
    if onnx_files:
        shutil.copyfile(onnx_files[-1], out / "model.onnx")
    (out / "int8.json").write_text(json.dumps({
        "arch_key": modeldefs.arch_key(arch), "input": [h, w], "outputs": arch["outputs"]["names"],
        "weights": "untrained, seeded", "seed": SEED,
        "calibration": f"{CALIB_IMAGES} seeded noise images, 0-255 RGB (D55)",
        "export": modeldefs.export_settings(arch), "code": version, "repo": repo_version(),
        "route": "RF-DETR's own export (ONNX, GridSample rewrite, onnx2tf) to a SavedModel; then TensorFlow's "
                 "converter in full-integer mode (our step: RF-DETR's exporter refuses full-integer)",
        "exporter_files": {p.name: p.stat().st_size for p in tf_files(tf_dir, "*.tflite")},
        "onnx_bytes": (out / "model.onnx").stat().st_size if onnx_files else None,
        "tflite_bytes": part.stat().st_size, "tflite_sha256": sha256(part),
        "network_attempts": len(attempts),
        "versions": {name: metadata.version(name) for name in ("torch", "onnx", "onnx2tf", "tensorflow", "rfdetr")},
        "seconds": round(time.time() - t0, 1),
    }, indent=1))
    os.replace(part, out / "model_int8.tflite")
    shutil.rmtree(tf_dir, ignore_errors=True)        # float, half-float and SavedModel copies: several times the model
    print(f"[to_int8_rfdetr] done: model_int8.tflite, {(out / 'model_int8.tflite').stat().st_size} bytes; "
          f"network attempts refused: {len(attempts)}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
