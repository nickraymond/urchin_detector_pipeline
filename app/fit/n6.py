"""N6 compiler step (S6 bite 3b, D52): ST Edge AI Core 4.0.0 through OpenMV's `tools/modelc.py`, inside the
`firmware-builder` container, as the earlier `ml/compile_model.sh` did; here the SDK, the OpenMV tree and
the input are mounted read-only. Docker must be running. Reproduces the earlier output byte for byte.

The output file keeps the name modelc gives it (`model_int8.tflite`), but it is not a TFLite file: it is
ST's relocatable Neural-ART binary for the N6's read-only model area (`compile_model.sh`, header).

Limit (source): `OMV_ROMFS_PART0_LENGTH 0x01800000` (24 MiB) in `boards/OPENMV_N6/board_config.h`. The
whole area is written as one image that must also hold the vendor's models (`ml/build_romfs_n6.sh`); how
much those take is bite 4's "stored together" check. ST Edge AI Core gives no speed estimate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.fit import reports

MODEL_AREA_BYTES = 0x01800000
LIMIT_SOURCE = "OpenMV boards/OPENMV_N6/board_config.h, OMV_ROMFS_PART0_LENGTH (24 MiB model area)"
# /sdk/gcc/bin is needed: ST's relocatable step links with arm-none-eabi-gcc (compile_model.sh)
INSIDE = ("export PATH=/sdk/python/bin:/sdk/stedgeai/Utilities/linux:/sdk/gcc/bin:$PATH; "
          "export PYTHONDONTWRITEBYTECODE=1; cd /omv; stedgeai --version | head -1; "
          'python3 tools/modelc.py --input /in/{name} --build-dir /out --stedge-args "--target stm32n6"')


def docker_command(tflite: Path, out: Path, fit: dict) -> list[str]:
    return ["docker", "run", "--rm", "--platform", "linux/amd64",
            "-v", f"{fit['sdk_dir']}:/sdk:ro", "-v", f"{fit['openmv_dir']}:/omv:ro",
            "-v", f"{tflite.parent}:/in:ro", "-v", f"{out}:/out",
            fit["docker_image"], "bash", "-lc", INSIDE.format(name=tflite.name)]


def compile(tflite: Path, out: Path, job: dict) -> dict:
    from app.fit.convert_worker import ToolsMissing
    fit = job["fit"]
    for key, what in (("sdk_dir", "OpenMV SDK"), ("openmv_dir", "OpenMV firmware tree")):
        if not Path(fit[key]).is_dir():
            raise ToolsMissing(f"{what} not found: {fit[key]}. Fix: set [fit] {key} in config.local.toml.")
    try:
        up = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ToolsMissing(f"Docker cannot be run ({e}). Fix: install and start Docker Desktop.") from None
    if up.returncode != 0:
        raise ToolsMissing("Docker is not running. Fix: start Docker Desktop, then run this again.")
    done = subprocess.run(docker_command(tflite, out, fit), capture_output=True, text=True)
    log = done.stdout + done.stderr
    (out / "compile.log").write_text(log)
    if "Unable to find image" in log or "No such image" in log:
        raise ToolsMissing(f"Container image {fit['docker_image']} not found. Fix: build it as the earlier "
                           "work did (ADIN_SPI_OpenMV/firmware/openmv_build/), or set [fit] docker_image.")
    version = next((l.strip() for l in log.splitlines() if l.startswith("ST Edge AI Core")), "ST Edge AI Core")
    base = {"tool": version, "tool_args": "--target stm32n6 --relocatable, OpenMV modelc.py default profile",
            "limit_bytes": MODEL_AREA_BYTES, "limit_source": LIMIT_SOURCE, "estimate_ms": None}
    made = out / tflite.name
    report = out / tflite.stem / "gen" / "network_generate_report.txt"
    if done.returncode != 0 or not made.is_file() or not report.is_file():
        return {**base, "converts": False, "error": reports.error_lines(log), "failing_layers": []}
    placed, sizes = reports.stedgeai_report(report.read_text(errors="replace")), reports.stedgeai_sizes(log)
    return {**base, "converts": True, "file": made.name, "file_bytes": made.stat().st_size,
            "params_bytes": sizes["params_bytes"], "memory_bytes": sizes["memory_bytes"],
            "placement": {"npu": placed["npu"], "hybrid": placed["hybrid"], "cpu": placed["cpu"],
                          "total": placed["total"], "unit": "epochs"},
            "failing_layers": placed["failing_layers"]}
