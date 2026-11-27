"""Setting Build 2's tools up as a job (S6 bite 3d, D51): the tool environments under `tools_dir` and the
pinned model-code checkout, made from `app/fit/toolchain/` with `uv` and `git`. The page shows what is
there and one button queues the job; nothing here needs the terminal (SPEC constraint 7).

Not set up by this job, because they are Nick's earlier installs and not ours to change: the OpenMV SDK,
the OpenMV firmware tree and the `firmware-builder` container image for the N6 compiler ([fit] settings).
status() only says whether they are there.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path

from app.fit import convert
from app.jobctl import Kind
from app.modeldefs import yolox

TOOLCHAIN = Path(__file__).parent / "toolchain"
# name, Python, requirements file (or one pinned package), what it is for
ENVS = [
    ("convert-venv", "3.11", "requirements-convert.txt", "PyTorch → ONNX → 8-bit; LiteRT (the Pi step)"),
    ("vela-5.0.0-venv", "3.12", "ethos-u-vela==5.0.0", "Arm Vela, the AE3 compiler"),
    ("rfdetr-venv", "3.11", "requirements-rfdetr.txt", "RF-DETR Nano and RF-DETR-Seg Nano"),
    ("rfdetr-plus-venv", "3.11", "requirements-rfdetr-plus.txt",
     "RF-DETR Atto / Femto / Pico: Roboflow licence, evaluation only"),
]
CHECKOUTS = [(yolox.CHECKOUT, yolox.CODE["repo"], yolox.CODE["commit"], "YOLOX model code")]
MARK = ".setup_ok"
DOCKER_CHECK_SECONDS = 60     # /fit reloads every 10 s while cells wait; ask Docker at most once a minute (QE PR 82)
_docker_seen: dict[str, tuple[float, bool]] = {}


def want(spec: str) -> str:
    """What an environment should hold: the requirements files' text (with what they include), hashed."""
    path = TOOLCHAIN / spec
    if not path.is_file():
        return hashlib.sha256(spec.encode()).hexdigest()
    text = path.read_text()
    for line in text.splitlines():
        if line.startswith("-r "):
            text += (TOOLCHAIN / line[3:].strip()).read_text()
    return hashlib.sha256(text.encode()).hexdigest()


def items(tools_dir: Path) -> list[dict]:
    """Every tool this job makes, with ok True / False and a reason when it is not."""
    tools_dir, out = Path(tools_dir), []
    for name, python, spec, what in ENVS:
        env = tools_dir / name
        mark = env / MARK
        if not (env / "bin" / "python").is_file():
            ok, why = False, "not there"
        elif not mark.is_file():
            ok, why = False, "not finished, or made by hand: run the set-up once to record it"
        elif mark.read_text().strip() != want(spec):
            ok, why = False, "its requirements have changed since it was made"
        else:
            ok, why = True, ""
        out.append({"kind": "env", "name": name, "python": python, "spec": spec, "what": what,
                    "path": str(env), "ok": ok, "why": why})
    for name, repo, commit, what in CHECKOUTS:
        path = tools_dir / "third_party" / name
        try:
            head = (path / ".git" / "HEAD").read_text().strip()
        except OSError:
            head = ""
        out.append({"kind": "checkout", "name": f"third_party/{name}", "repo": repo, "commit": commit,
                    "what": what, "path": str(path), "ok": head == commit,
                    "why": "" if head == commit else ("not there" if not head else f"at {head[:12]}, not {commit[:12]}")})
    return out


def outside(fit: dict | None) -> list[dict]:
    """The N6 compiler's pieces from Nick's earlier work: only checked, never changed."""
    s = convert.fit_settings(fit)
    out = [{"name": "OpenMV SDK (ST Edge AI Core 4.0.0)", "path": s["sdk_dir"], "ok": Path(s["sdk_dir"]).is_dir()},
           {"name": "OpenMV firmware tree (modelc.py, vela.ini)", "path": s["openmv_dir"],
            "ok": (Path(s["openmv_dir"]) / "tools" / "modelc.py").is_file()}]
    docker = shutil.which("docker")
    seen = _docker_seen.get(s["docker_image"])
    if seen and time.monotonic() - seen[0] < DOCKER_CHECK_SECONDS:
        up = seen[1]
    else:
        try:
            up = bool(docker) and subprocess.run([docker, "image", "inspect", s["docker_image"]],
                                                 capture_output=True, timeout=20).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            up = False
        _docker_seen[s["docker_image"]] = (time.monotonic(), up)
    out.append({"name": f"Docker with the image {s['docker_image']}", "path": docker or "docker not found",
                "ok": up})
    return out


def build_job(p: dict) -> dict:
    uv = p.get("uv") or shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    return {**p, "uv": uv, "git": p.get("git") or shutil.which("git") or "git"}


def progress(p: dict) -> tuple[int, int]:
    all_items = items(Path(p["tools_dir"]))
    return sum(i["ok"] for i in all_items), len(all_items)


def setup_kind() -> Kind:
    return Kind("tools_setup", module="app.fit.setup_worker", prepare=build_job, progress=progress,
                item="tool", pausable=False)
