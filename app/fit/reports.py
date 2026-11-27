"""Read the two compilers' own reports into fit.json fields (S6 bite 3b). Text in, plain data out; the
formats are those of Vela 5.0.0 and ST Edge AI Core 4.0.0, tested on the real reports of the known-good
run (tests/fixtures/fit/). Method: `ADIN_SPI_OpenMV/ml/compile_gate_report.md`.
"""

from __future__ import annotations

import csv
import io
import re

MIB = 1024 * 1024


def vela_summary(csv_text: str) -> dict:
    """Vela's `<model>_summary_<system config>.csv`: one row; sizes are in KiB, time in seconds."""
    row = next(csv.DictReader(io.StringIO(csv_text)))
    return {"estimate_ms": round(float(row["inference_time"]) * 1000, 2),
            "memory_bytes": round(float(row["sram_memory_used"]) * 1024),
            "flash_bytes": round(float(row["off_chip_flash_memory_used"]) * 1024),
            "accelerator": row["accelerator_configuration"], "system_config": row["system_config"]}


def vela_log(text: str) -> dict:
    """Vela's console output: operator counts, and one entry per operator it leaves on the CPU.
    A warning block is "Warning (supported operators) operator:<TYPE> ofm:<name>" then "Reason: <why>"."""
    def count(label: str) -> int | None:
        m = re.search(rf"^{label} operators = (\d+)", text, re.M)
        return int(m.group(1)) if m else None

    layers = []
    for m in re.finditer(r"^Warning \(supported operators\) operator:(\S+) ofm:(\S+)\s*\n((?:(?!Warning |\n).*\n?)*)",
                         text, re.M):
        reason = " ".join(l.strip() for l in m.group(3).splitlines()
                          if l.strip() and not l.startswith("TfLiteReader"))
        layers.append({"name": m.group(2), "type": m.group(1), "why": reason.removeprefix("Reason: ")})
    return {"npu": count("NPU"), "cpu": count("CPU"), "failing_layers": layers}


def stedgeai_report(text: str) -> dict:
    """ST's `network_generate_report.txt`, section "Epochs details": counts of hardware, hybrid and
    software steps ("epochs"), and each hybrid or software step with the layer type ST names for it."""
    def count(pattern: str) -> int | None:
        m = re.search(pattern + r"\s+(\d+)", text)
        return int(m.group(1)) if m else None

    total = count(r"Total number of epochs")
    if total is None:
        raise ValueError('no "Total number of epochs" line: not an ST Edge AI Core generate report')
    layers = [{"name": m.group(1), "type": m.group(3).strip() or "?",
               "why": "runs partly on the CPU (hybrid)" if m.group(2) == "Hybrid" else "runs on the CPU (software)"}
              for m in re.finditer(r"^\|\s*(epoch_\d+)\s*\|\s*(Hybrid|SW)\s*\|([^|]*)\|", text, re.M)]
    version = re.search(r"ST Edge AI Core (v[\w.\-]+)", text)
    return {"npu": count(r">> pure hardware \(HW or EC\) epochs"),
            "hybrid": count(r">> hybrid epochs \(using both software and hardware\)"),
            "cpu": count(r">> pure software \(SW\) epochs"), "total": total,
            "failing_layers": layers, "version": version.group(1) if version else None}


def stedgeai_sizes(log_text: str) -> dict:
    """The size table OpenMV's modelc.py prints from ST's output (bytes)."""
    def size(label: str) -> int | None:
        m = re.search(rf"{label}\s*=\s*([\d,]+)", log_text)
        return int(m.group(1).replace(",", "")) if m else None
    return {"file_bytes": size("binary file size"), "params_bytes": size("params size"),
            "memory_bytes": size("acts size")}


def error_lines(log_text: str, keep: int = 6) -> str:
    """The lines of a failed compile that say why: ST's and Vela's error lines, else the log's end.
    Colour codes are removed."""
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", l).rstrip() for l in log_text.splitlines()]
    lines = [l for l in lines if l.strip()]
    hits = [l for l in lines if re.search(r"INTERNAL ERROR|NOT IMPLEMENTED|^\s*E\d{3}\(|^Error|Error:", l)]
    return "\n".join((hits or lines)[-keep:])
