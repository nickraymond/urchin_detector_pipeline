"""A fake 8-bit step for convert-job tests (no torch): `python -m tests.fake_int8 <recipe.json> <int8 dir>`.
The recipe's arch may carry "fake": {"int8_fails": true} or {"int8_seconds": 5}."""

import json
import sys
import time
from pathlib import Path


def main() -> None:
    arch = json.loads(Path(sys.argv[1]).read_text())["arch"]
    out, fake = Path(sys.argv[2]), arch.get("fake") or {}
    with open(out / "calls.log", "a") as f:
        f.write("int8\n")
    print("[fake to_int8] start", flush=True)
    time.sleep(fake.get("int8_seconds", 0))
    if fake.get("int8_fails"):
        print("onnx2tf: Expand is not supported in this fake", flush=True)
        sys.exit(1)
    (out / "int8.json").write_text(json.dumps({"tflite_bytes": 4, "seed": 0}))
    (out / "model_int8.tflite").write_bytes(b"TFL3")


if __name__ == "__main__":
    main()
