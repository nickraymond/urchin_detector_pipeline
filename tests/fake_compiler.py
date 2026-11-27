"""A fake camera compiler for convert-job tests: places everything on the NPU unless the recipe's arch
carries "fake": {"cpu_layers": [...]} (runs, partly on the CPU) or {"compile_fails": "reason"}."""

from pathlib import Path


def compile(tflite: Path, out: Path, job: dict) -> dict:
    fake = job["arch"].get("fake") or {}
    (out / "compile.log").write_text(f"fake compiler read {tflite.stat().st_size} bytes\n")
    base = {"tool": "fake compiler 1.0", "limit_bytes": 1000, "limit_source": "a made-up board"}
    if fake.get("compile_fails"):
        return {**base, "converts": False, "error": fake["compile_fails"],
                "failing_layers": [{"name": "transpose_7", "type": "TRANSPOSE", "why": fake["compile_fails"]}]}
    cpu = fake.get("cpu_layers") or []
    (out / "model.bin").write_bytes(b"x" * 10)
    return {**base, "converts": True, "file_bytes": 10, "memory_bytes": 512,
            "placement": {"npu": 9 - len(cpu), "hybrid": 0, "cpu": len(cpu), "total": 9, "unit": "operators"},
            "failing_layers": [{"name": n, "type": "TRANSPOSE", "why": "runs on the CPU"} for n in cpu],
            "estimate_ms": 12.5}
