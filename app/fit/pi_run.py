"""Run an 8-bit TFLite file once under LiteRT and print one JSON line. Runs INSIDE the conversion
environment (D51), started by app/fit/pi.py: `python -m app.fit.pi_run <model.tflite>`. Imports nothing
from the rest of the app. Everything measured here is on the laptop, not on a Pi."""

from __future__ import annotations

import collections
import json
import resource
import sys
import time
from importlib import metadata

RUNS = 20


def main(argv: list[str]) -> None:
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=argv[1], num_threads=1)
    ops = collections.Counter(o["op_name"] for o in it._get_ops_details())
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], np.zeros(inp["shape"], inp["dtype"]))
    it.invoke()                                   # first run not timed
    t = time.perf_counter()
    for _ in range(RUNS):
        it.invoke()
    ms = (time.perf_counter() - t) / RUNS * 1000
    odd = sorted(n for n in ops if n.startswith("Flex") or n in ("CUSTOM", "DELEGATE"))
    print(json.dumps({
        "runtime": "LiteRT (ai-edge-litert) " + metadata.version("ai-edge-litert"),
        "operators": sum(ops.values()), "operator_types": dict(sorted(ops.items())), "not_builtin": odd,
        "input": {"shape": [int(v) for v in inp["shape"]], "dtype": np.dtype(inp["dtype"]).name},
        "outputs": [{"shape": [int(v) for v in o["shape"]], "dtype": np.dtype(o["dtype"]).name}
                    for o in it.get_output_details()],
        "laptop_ms": round(ms, 2), "runs": RUNS,
        # macOS reports ru_maxrss in bytes, Linux in KiB
        "laptop_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
    }))


if __name__ == "__main__":
    main(sys.argv)
