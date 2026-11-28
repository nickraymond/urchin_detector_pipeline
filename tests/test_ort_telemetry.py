"""Lead call (lead 2, 2026-11-28; QE 11's card 9 check): ONNX Runtime's official builds send telemetry to Microsoft by
default (its own Privacy.md); `ORT_DISABLE_TELEMETRY=1` before it starts stops the uploader, events and device id.
Every way this repo starts ONNX Runtime must have the switch set first, even from an environment without it. A
stand-in `onnxruntime` package records what it sees at import and exits; the real one is never loaded here."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AGENT = REPO / "app" / "bench" / "agent" / "bench_agent.py"


@pytest.fixture
def standin(tmp_path):
    """A fake onnxruntime (and onnx) that writes the switch's value to a file and exits 0 at import."""
    pkg = tmp_path / "standin"
    (pkg / "onnxruntime").mkdir(parents=True)
    seen = tmp_path / "seen.txt"
    (pkg / "onnxruntime" / "__init__.py").write_text(
        "import os, sys\n"
        f"open({str(seen)!r}, 'w').write(os.environ.get('ORT_DISABLE_TELEMETRY', 'UNSET'))\n"
        "sys.exit(0)\n")
    (pkg / "onnx.py").write_text("")
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}    # start from no switch at all
    return pkg, seen, env


def seen_at_import(standin, code):
    pkg, seen, env = standin
    prefix = f"import sys; sys.path.insert(0, {str(pkg)!r}); sys.path.insert(1, {str(REPO)!r}); "
    r = subprocess.run([sys.executable, "-I", "-c", prefix + code], env=env, cwd=pkg, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    return seen.read_text() if seen.exists() else "never imported"


def test_the_app_package_sets_the_switch_before_onnxruntime_starts(standin):
    """The web app and every `python -m app.…` worker or tool step import `app` first."""
    assert seen_at_import(standin, "import app; import onnxruntime") == "1"


def test_the_qdq_step_sets_the_switch_before_onnxruntime_starts(standin):
    assert seen_at_import(standin, "import runpy; sys.argv = ['to_qdq', 'in.onnx', 'out.onnx']; "
                                   "runpy.run_module('app.fit.to_qdq', run_name='__main__')") == "1"


def test_the_pi_agent_sets_the_switch_itself(standin):
    """The agent runs alone on the Pi (no `app` package there), so it sets the switch at its own top."""
    code = ("import importlib.util as u; s = u.spec_from_file_location('bench_agent', %r); m = u.module_from_spec(s); "
            "s.loader.exec_module(m); assert 'app' not in sys.modules; import onnxruntime" % str(AGENT))
    assert seen_at_import(standin, code) == "1"


def test_the_switch_is_set_before_every_onnxruntime_import_in_the_repo():
    """Any file that imports onnxruntime is in the `app` package (app/__init__.py) or sets the switch above it; and the
    agent's own install environment, built from scratch, names it too."""
    for path in REPO.rglob("*.py"):
        if any(part in (".venv", ".git", "node_modules") for part in path.parts) or path.parent.name == "standin":
            continue
        text = path.read_text(errors="replace")
        first = re.search(r"^\s*(import onnxruntime|from onnxruntime)", text, re.M)
        if not first or path.parts[len(REPO.parts)] == "tests":
            continue
        in_app = path.is_relative_to(REPO / "app") and path != AGENT
        sets = text.find('os.environ["ORT_DISABLE_TELEMETRY"] = "1"')
        assert in_app or -1 < sets < first.start(), f"{path.relative_to(REPO)} starts onnxruntime without the switch"
    install = AGENT.read_text().split('"PIP_CONFIG_FILE"')[0].rsplit("env = {", 1)[1]
    assert '"ORT_DISABLE_TELEMETRY": "1"' in install


def test_every_child_env_the_app_builds_names_the_switch():
    for path in (REPO / "app").rglob("*.py"):
        if path == AGENT:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\{\*\*os\.environ|dict\(os\.environ", line):
                assert "ORT_OFF" in line, f"{path.relative_to(REPO)}:{n} builds a child env without ORT_OFF"


def test_this_test_run_has_the_switch():
    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"          # tests/conftest.py
