"""PyTorch -> ONNX -> 8-bit TFLite for one recipe. Runs INSIDE a tool environment (D51), started by the
convert worker as `<tools_dir>/<env>/bin/python -m app.fit.to_int8 <recipe.json> <int8 dir>` with the repo
on PYTHONPATH. Imports only `app.modeldefs` from this repo.

The route is the earlier work's (`ADIN_SPI_OpenMV/ml/yolox_urchin/export.py`): classic ONNX exporter,
then `onnx2tf -oiqt` with a calibration file. Changes: seeded weights and seeded noise calibration for
untrained models (D55); an optional onnxsim pass (D56); every step leaves its file, and int8.json says
how the result was made. A failure exits non-zero after printing the reason; the worker turns that into
a fit report ("does not convert"), not a job error.

Writes into <int8 dir>: model.onnx, calib.npy, model_int8.tflite, int8.json (onnx2tf's other outputs are removed); onnx2tf's own log goes to
this process's output (the worker saves it as convert.log).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

CALIB_IMAGES = 20
SEED = 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_version(arch: dict) -> str:
    """What model code is really on this machine, checked against the recipe's `code.commit`."""
    from app import modeldefs
    want = arch["code"]["commit"]
    if want.startswith("pip:"):
        name, _, pin = want[4:].partition("==")
        have = metadata.version(name)
        if pin and have != pin:
            raise SystemExit(f"{name} {have} is installed in this tool environment, the recipe asks for {pin}. "
                             "Fix: set the tools up again.")
        return f"pip:{name}=={have}"
    family = sys.modules.get(modeldefs.FAMILIES[arch["family"]])
    checkout = getattr(family, "CHECKOUT", None)
    if not checkout:
        return want                                   # this repository's own code
    head = (modeldefs.tools_dir() / "third_party" / checkout / ".git" / "HEAD").read_text().strip()
    if head != want:
        raise SystemExit(f"The {checkout} checkout is at {head[:12]}, the recipe asks for {want[:12]}. "
                         "Fix: set the tools up again.")
    return head


def main(argv: list[str]) -> None:
    recipe = json.loads(Path(argv[1]).read_text())
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    arch = recipe["arch"]
    t0 = time.time()

    import numpy as np
    import torch
    from app import modeldefs

    settings = modeldefs.export_settings(arch)
    torch.manual_seed(SEED)
    model = modeldefs.build(arch)
    version = code_version(arch)
    weights = recipe.get("weights")                   # a trained model's state_dict file (S11); none in S6
    if weights:
        raise SystemExit("Trained weights are not handled yet (S11).")
    h, w = arch["input"]
    names = arch["outputs"]["names"]

    onnx_path = out / "model.onnx"
    print(f"[to_int8] ONNX export, opset {settings['opset']}", flush=True)
    torch.onnx.export(model, torch.randn(1, 3, h, w), str(onnx_path), opset_version=settings["opset"],
                      dynamo=False, input_names=["images"], output_names=names)
    source = onnx_path
    if settings["simplify"]:
        import onnx
        import onnxsim                                 # its command line crashes printing a summary (DESIGN)
        print("[to_int8] onnxsim", flush=True)
        simplified, ok = onnxsim.simplify(onnx.load(str(onnx_path)))
        if not ok:
            raise SystemExit("onnxsim could not check its simplified model against the original.")
        source = out / "model_sim.onnx"
        onnx.save(simplified, str(source))

    # Calibration (D55): seeded noise, raw 0-255 RGB as in the earlier work (training used unnormalised pixels).
    calib = np.random.default_rng(SEED).uniform(0, 255, (CALIB_IMAGES, h, w, 3)).astype(np.float32)
    np.save(out / "calib.npy", calib)
    # onnx2tf fetches a sample-data file from the internet unless it finds it in its working folder
    # (`ml/compile_gate_report.md`, "potholes"); nothing is downloaded here.
    np.save(out / "calibration_image_sample_data_20x128x128x3_float32.npy", calib[:, :128, :128] / 255.0)

    tf_dir = out / "tf"
    cmd = [str(Path(sys.executable).parent / "onnx2tf"), "-i", str(source), "-o", str(tf_dir), "-oiqt",
           "--non_verbose", "-cind", "images", str(out / "calib.npy"),
           "[[[[0.0,0.0,0.0]]]]", "[[[[1.0,1.0,1.0]]]]"]
    print("[to_int8] " + " ".join(cmd), flush=True)
    done = subprocess.run(cmd, cwd=out, stdout=sys.stdout, stderr=subprocess.STDOUT)
    quant = sorted(tf_dir.glob("*_full_integer_quant.tflite"))
    if done.returncode != 0 or len(quant) != 1:
        raise SystemExit(f"onnx2tf did not produce an 8-bit model (exit code {done.returncode}, "
                         f"{len(quant)} file(s)); its last lines are above.")
    tflite = out / "model_int8.tflite"
    tflite.write_bytes(quant[0].read_bytes())
    shutil.rmtree(tf_dir, ignore_errors=True)        # float and half-quantised copies: several times the model

    (out / "int8.json").write_text(json.dumps({
        "arch_key": modeldefs.arch_key(arch), "input": [h, w], "outputs": names,
        "weights": "untrained, seeded", "seed": SEED,
        "calibration": f"{CALIB_IMAGES} seeded noise images, 0-255 RGB (D55)",
        "export": settings, "code": version,
        "onnx_bytes": onnx_path.stat().st_size, "tflite_bytes": tflite.stat().st_size,
        "tflite_sha256": sha256(tflite),
        "versions": {name: metadata.version(name) for name in ("torch", "onnx", "onnx2tf", "tensorflow")},
        "seconds": round(time.time() - t0, 1),
    }, indent=1))
    print(f"[to_int8] done: {tflite.name}, {tflite.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main(sys.argv)
