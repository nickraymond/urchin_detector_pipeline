"""The Pi's second 8-bit file (S7 bite 3c, D85; Lead call, lead 2: S6's Pi step writes it): an 8-bit QDQ ONNX file
for ONNX Runtime, made from the candidate's own `int8/model.onnx` with onnxruntime's `quantize_static`, calibrated on
D55's seeded noise (CALIB_IMAGES images, uniform 0-255, seed 0, as `to_int8.py` makes for onnx2tf, here NCHW, the
layout of our own export). Then it runs once under ONNX Runtime on the laptop, one thread, and prints one JSON line.

Runs INSIDE the conversion environment (D51), started by app/fit/pi.py: `python -m app.fit.to_qdq <model.onnx>
<out.onnx>`. Imports nothing from the rest of the app. Everything measured here is on the laptop, not on a Pi.
"""

from __future__ import annotations

import json
import os
import sys
import time

CALIB_IMAGES, SEED, RUNS = 20, 0, 20     # D55's calibration; RUNS as pi_run.py
TOOL_FAULT = 3                          # the tool itself could not be used (as pi_run.py)


def main(argv: list[str]) -> None:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as e:
        print(f"tool fault: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(TOOL_FAULT) from None
    src, out = argv[1], argv[2]
    inp = onnx.load(src).graph.input[0]
    dims = [d.dim_value or 1 for d in inp.type.tensor_type.shape.dim]           # a free dimension: batch 1
    calib = np.random.default_rng(SEED).uniform(0, 255, (CALIB_IMAGES, *dims[1:])).astype(np.float32)

    class Noise(CalibrationDataReader):
        def __init__(self):
            self.it = iter({inp.name: calib[i:i + 1]} for i in range(CALIB_IMAGES))

        def get_next(self):
            return next(self.it, None)

    prep = out + ".prep.onnx"                    # onnxruntime's own pre-processing (shape inference, folding) first
    try:
        quant_pre_process(src, prep)
        quantize_static(prep, out, Noise(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8, per_channel=True)
    finally:
        if os.path.exists(prep):
            os.remove(prep)
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
    sess = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])
    feed = {inp.name: calib[:1]}
    sess.run(None, feed)                                                         # first run not timed
    t = time.perf_counter()
    for _ in range(RUNS):
        sess.run(None, feed)
    ms = (time.perf_counter() - t) / RUNS * 1000
    ops = {}
    for n in onnx.load(out).graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(json.dumps({"runtime": "onnxruntime " + ort.__version__, "input": {"name": inp.name, "shape": dims},
                      "calibration": f"{CALIB_IMAGES} seeded noise images, 0-255, NCHW (D55)",
                      "quantization": "onnxruntime quant_pre_process, then QDQ: int8 activations and weights, per-channel weights",
                      "laptop_ms": round(ms, 2), "runs": RUNS, "operator_types": dict(sorted(ops.items()))}))


if __name__ == "__main__":
    main(sys.argv)
