"""A second N6 compiler step (S6 bite 2f, D71): ST Edge AI Core 4.1.0 from Nick's own install, beside the
known-good 4.0.0 of app/fit/n6.py, never instead of it. Its results live in `targets/n6_st41/`.

What it can and cannot say today:
- It runs natively on the Mac (`[fit] stedgeai41`, default /Applications/ST/STEdgeAI/4.1/Utilities/macarm/stedgeai;
  the install has no Linux build, so it cannot go in the 4.0.0 route's container). Nothing is written to the install.
- PLACEMENT ONLY. ST's last step, the relocatable file the camera loads, links with `arm-none-eabi-gcc`, and there
  is no macOS build of that toolchain on the laptop. So this step runs `stedgeai generate` without `--relocatable`:
  it gives the hardware / hybrid / software steps, and no loadable file and no file size.
- OpenMV's SDK bundles ST Edge AI Core 4.0 (D52). A model compiled with 4.1.0 may not load on the camera's
  current firmware; only S7 on the board settles that.

DATA OFF THE LAPTOP (Nick, 2026-11-27: "don't let anymore ST data sharring leave the network"). ST's tool asks
"Do you allow statistics to improve the command line? (y)es / (n)o" and keeps a uuid and a queue of usage
records in ~/.stmai (command, target, model type, layer count, platform; no frames, weights or file names). With
the network closed it still queues a record even after "n", to send later. So every run here:
  1. answers "n" on stdin;
  2. runs inside macOS's `sandbox-exec` with the network denied and every write to ~/.stmai denied;
  3. logs the checksum of ~/.stmai before and after; a changed checksum stops the job with an error (the
     block did not hold), after compile.log is written.
The guard was checked against v4.1.0-20732 only (Nick's install, 2026-11-27): another 4.1.x may ask or store
differently; check it again before using one.
run() refuses to start the tool any other way. Never start this `stedgeai` outside this module.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from app.fit import n6, reports

DEFAULT_BINARY = "/Applications/ST/STEdgeAI/4.1/Utilities/macarm/stedgeai"
STATS_FILE = Path.home() / ".stmai"
NOTE = ("placement only: no loadable file or file size (ST's relocatable step needs an arm-none-eabi-gcc for "
        "macOS, which is not on this laptop). OpenMV's firmware bundles ST Edge AI Core 4.0, so a 4.1.0 result "
        "may not load on the camera today; S7 on the board settles it")


def profile(stats_file: Path = STATS_FILE) -> str:
    """The sandbox profile: everything allowed except the network and writing the tool's statistics file."""
    return f'(version 1)(allow default)(deny network*)(deny file-write* (literal "{stats_file}"))'


def sandboxed(command: list[str], stats_file: Path = STATS_FILE) -> list[str]:
    return ["sandbox-exec", "-p", profile(stats_file), *command]


def checksum(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "no file"


def run(command: list[str], cwd: Path, stats_file: Path = STATS_FILE) -> tuple[subprocess.CompletedProcess, str]:
    """Start ST's tool. Refuses any command that is not wrapped by sandboxed() with both denials."""
    ok = (len(command) > 3 and command[0] == "sandbox-exec" and command[1] == "-p"
          and "(deny network*)" in command[2] and f'(deny file-write* (literal "{stats_file}"))' in command[2])
    if not ok:
        raise RuntimeError("ST Edge AI Core 4.1.0 may only run with the network and its statistics file blocked "
                           "(app/fit/n6_st41.py, sandboxed()). Nothing was started.")
    before = checksum(stats_file)
    done = subprocess.run(command, cwd=cwd, input="n\n", capture_output=True, text=True)   # "n": no statistics
    after = checksum(stats_file)
    note = f"[n6_st41] {stats_file} sha256 before {before[:16]} after {after[:16]}: " + (
        "unchanged" if before == after else "CHANGED: the block did not hold; tell the lead, this is data that could leave")
    return done, note


class StatsFileChanged(RuntimeError):
    """~/.stmai changed during a run: ST's tool wrote statistics despite the block. A job error, not a result."""


def compile(tflite: Path, out: Path, job: dict) -> dict:
    from app.fit.convert_worker import ToolsMissing
    fit = job["fit"]
    binary = Path(fit.get("stedgeai41") or DEFAULT_BINARY).expanduser()
    config = Path(fit["openmv_dir"]) / "lib" / "stai" / "scripts" / "neuralart.json"
    for path, what, fix in ((binary, "ST Edge AI Core 4.1.0", "it is installed by Nick with ST's own installer; set "
                             "[fit] stedgeai41 in config.local.toml if it is somewhere else"),
                            (config, "OpenMV's neuralart.json", "set [fit] openmv_dir in config.local.toml")):
        if not path.is_file():
            raise ToolsMissing(f"{what} not found: {path}. Fix: {fix}.")
    if not shutil.which("sandbox-exec"):
        raise ToolsMissing("macOS's sandbox-exec not found: ST Edge AI Core 4.1.0 is not run without it (it would "
                           "send usage statistics). Fix: none on this system; use the 4.0.0 step.")
    version, version_note = run(sandboxed([str(binary), "--version"]), out)
    command = [str(binary), "generate", "--target", "stm32n6", "--model", str(tflite),
               "--st-neural-art", f"default@{config}", "--workspace", str(out / "workspace"),
               "--output", str(out / "gen"), "--verbosity", "1", "--quiet"]
    done, note = run(sandboxed(command), out)
    log = version.stdout + done.stdout + done.stderr + "\n" + version_note + "\n" + note + "\n"
    (out / "compile.log").write_text(log)
    if "CHANGED" in version_note or "CHANGED" in note:
        raise StatsFileChanged(f"~/.stmai changed while ST Edge AI Core 4.1.0 ran: the statistics block "
                               f"did not hold. Nothing was recorded as a result; {out / 'compile.log'} has the "
                               "checksums. Fix: do not run this step again; tell the lead.")
    tool = next((l.strip() for l in log.splitlines() if l.startswith("ST Edge AI Core")), "ST Edge AI Core 4.1.0")
    base = {"tool": tool, "tool_args": "--target stm32n6, OpenMV neuralart.json default profile, no --relocatable",
            "limit_bytes": n6.MODEL_AREA_BYTES, "limit_source": n6.LIMIT_SOURCE, "estimate_ms": None,
            "note": NOTE, "stats_file": note}
    report = out / "gen" / "network_generate_report.txt"
    if done.returncode != 0 or not report.is_file():
        return {**base, "converts": False, "error": reports.error_lines(log.replace(version_note, "").replace(note, "")), "failing_layers": []}
    placed = reports.stedgeai_report(report.read_text(errors="replace"))
    return {**base, "converts": True, "file": None, "file_bytes": None, "memory_bytes": None,
            "placement": {"npu": placed["npu"], "hybrid": placed["hybrid"], "cpu": placed["cpu"],
                          "total": placed["total"], "unit": "epochs"},
            "failing_layers": placed["failing_layers"]}
