"""AE3 compiler step (S6 bite 3b, D52): Arm Vela 5.0.0, run natively from `<tools_dir>/vela-5.0.0-venv`
with OpenMV's `tools/vela.ini` and the arguments of the earlier `ml/compile_model.sh` (the ones OpenMV
compiles its own shipped models with). Reproduces the earlier container output byte for byte (DESIGN,
Bench results).

Limits (sources): the model area is the board's ROMFS partition 0, `OMV_ROMFS_PART0_LENGTH 0x01800000`
(24 MiB) in `boards/OPENMV_AE3/board_config.h` of the OpenMV tree: the HP core's partition (`#if CORE_M55_HP`), which
is the core Vela's RTSS_HP configuration targets; the HE core's is 1 MiB. Free memory on the board is not a
build-time fact: the earlier work measured 4.09 MB free before a model loads (`STAGE1.md`, 2026-08-23),
kept here as a reference until S7 measures it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from app import sidecars
from app.fit import reports

VELA_ENV = "vela-5.0.0-venv"
ARGS = ["--system-config", "RTSS_HP_DTCM_MRAM", "--accelerator-config", "ethos-u55-256",
        "--memory-mode", "Shared_Sram", "--optimise", "Performance"]
MODEL_AREA_BYTES = 0x01800000
LIMIT_SOURCE = "OpenMV boards/OPENMV_AE3/board_config.h, OMV_ROMFS_PART0_LENGTH for the HP core (24 MiB model area)"
MEMORY_REFERENCE = {"bytes": 4_090_000, "source": "free memory measured on the AE3 in the earlier work "
                    "(STAGE1.md, 2026-08-23); not re-measured"}


def compile(tflite: Path, out: Path, job: dict) -> dict:
    from app.fit.convert_worker import ToolsMissing
    vela = Path(job["tools_dir"]) / VELA_ENV / "bin" / "vela"
    ini = Path(job["fit"]["openmv_dir"]) / "tools" / "vela.ini"
    for path, what in ((vela, "Arm Vela"), (ini, "OpenMV's vela.ini")):
        if not path.is_file():
            raise ToolsMissing(f"{what} not found: {path}. Fix: set the tools up (DESIGN D52).")
    version = subprocess.run([str(vela), "--version"], capture_output=True, text=True).stdout.strip()
    done = subprocess.run([str(vela), *ARGS, "--output-dir", str(out), "--config", str(ini), str(tflite)],
                          capture_output=True, text=True)
    log = done.stdout + done.stderr
    (out / "compile.log").write_text(log)
    base = {"tool": f"Arm Vela {version}", "tool_args": " ".join(ARGS), "limit_bytes": MODEL_AREA_BYTES,
            "limit_source": LIMIT_SOURCE, "memory_reference": MEMORY_REFERENCE}
    made = out / f"{tflite.stem}_vela.tflite"
    summaries = sorted(sidecars.real(out.glob(f"{tflite.stem}_summary_*.csv")))
    if done.returncode != 0 or not made.is_file() or not summaries:
        return {**base, "converts": False, "error": reports.error_lines(log),
                "failing_layers": reports.vela_log(log)["failing_layers"]}
    summary, placed = reports.vela_summary(summaries[0].read_text()), reports.vela_log(log)
    return {**base, "converts": True, "file": made.name, "file_bytes": made.stat().st_size,
            "memory_bytes": summary["memory_bytes"], "estimate_ms": summary["estimate_ms"],
            "placement": {"npu": placed["npu"], "hybrid": 0, "cpu": placed["cpu"],
                          "total": (placed["npu"] or 0) + (placed["cpu"] or 0), "unit": "operators"},
            "failing_layers": placed["failing_layers"]}
