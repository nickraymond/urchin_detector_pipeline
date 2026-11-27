"""S6 bite 3d: the tools set-up job, with stand-ins for `uv` and `git` (shell scripts that make the
folders the real ones would). Nothing is downloaded or installed."""

import time
from pathlib import Path

import pytest

from app.fit import setup
from app.jobctl import JobControl
from app.modeldefs import yolox

FAKE_UV = """#!/bin/sh
echo "fake uv $@" >> "$(dirname "$0")/calls.log"
if [ "$1" = venv ]; then for last; do :; done; mkdir -p "$last/bin"; : > "$last/bin/python"; fi
if [ -f "$(dirname "$0")/fail" ] && [ "$1" = pip ]; then echo "error: No solution found"; exit 2; fi
"""
FAKE_GIT = """#!/bin/sh
echo "fake git $@" >> "$(dirname "$0")/calls.log"
if [ "$1" = clone ]; then mkdir -p "$3/.git"; fi
if [ "$1" = checkout ]; then echo "$3" > .git/HEAD; fi
"""


@pytest.fixture
def box(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("uv", FAKE_UV), ("git", FAKE_GIT)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    jc = JobControl(tmp_path / "jobs")
    jc.register(setup.setup_kind())
    yield jc, tmp_path / "tools", bin_dir
    jc.close()


def run(jc, tools, bin_dir):
    jid = jc.submit("tools_setup", {"tools_dir": str(tools), "uv": str(bin_dir / "uv"), "git": str(bin_dir / "git")})
    end = time.monotonic() + 30
    while jc.status(jid)["state"] not in ("done", "error") and time.monotonic() < end:
        time.sleep(0.05)
    return jc.status(jid)


def calls(bin_dir) -> list[str]:
    log = bin_dir / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def test_nothing_is_ready_in_an_empty_folder(tmp_path):
    items = setup.items(tmp_path)
    assert [i["name"] for i in items] == ["convert-venv", "vela-5.0.0-venv", "rfdetr-venv", "rfdetr-plus-venv",
                                          "third_party/YOLOX"]
    assert not any(i["ok"] for i in items) and all(i["why"] == "not there" for i in items)
    assert "Roboflow licence" in items[3]["what"]
    assert setup.progress({"tools_dir": str(tmp_path)}) == (0, 5)


def test_set_up_makes_every_tool_once_and_a_second_run_does_nothing(box):
    jc, tools, bin_dir = box
    s = run(jc, tools, bin_dir)
    assert s["state"] == "done" and (s["done"], s["total"]) == (5, 5)
    assert all(i["ok"] for i in setup.items(tools))
    log = calls(bin_dir)
    assert sum(l.startswith("fake uv venv") for l in log) == 4 and sum("pip install" in l for l in log) == 4
    assert any("venv --python 3.11" in l and l.endswith("convert-venv") for l in log)
    assert any("pip install" in l and l.endswith("ethos-u-vela==5.0.0") for l in log)
    assert any(l.endswith("requirements-rfdetr-plus.txt") for l in log)
    assert f"fake git checkout --detach {yolox.CODE['commit']}" in log
    assert (tools / "third_party" / "YOLOX" / ".git" / "HEAD").read_text().strip() == yolox.CODE["commit"]
    before = len(log)
    assert run(jc, tools, bin_dir)["state"] == "done" and len(calls(bin_dir)) == before


def test_a_hand_made_or_outdated_environment_is_made_again(box):
    jc, tools, bin_dir = box
    run(jc, tools, bin_dir)
    (tools / "convert-venv" / setup.MARK).unlink()                       # as if made by hand
    (tools / "rfdetr-venv" / setup.MARK).write_text("an older requirements file\n")
    (tools / "third_party" / "YOLOX" / ".git" / "HEAD").write_text("0" * 40 + "\n")
    why = {i["name"]: i["why"] for i in setup.items(tools) if not i["ok"]}
    assert set(why) == {"convert-venv", "rfdetr-venv", "third_party/YOLOX"}
    assert "made by hand" in why["convert-venv"] and "have changed" in why["rfdetr-venv"]
    assert why["third_party/YOLOX"].startswith("at 000000000000, not 6ddff4824372")
    assert run(jc, tools, bin_dir)["state"] == "done" and all(i["ok"] for i in setup.items(tools))


def test_a_failed_install_stops_the_job_with_the_reason_and_is_not_marked_ready(box):
    jc, tools, bin_dir = box
    (bin_dir / "fail").write_text("x")
    s = run(jc, tools, bin_dir)
    assert s["state"] == "error" and "No solution found" in str(s)
    assert not setup.items(tools)[0]["ok"]
    (bin_dir / "fail").unlink()
    assert run(jc, tools, bin_dir)["state"] == "done"


def test_missing_uv_says_how_to_fix(box):
    jc, tools, bin_dir = box
    jid = jc.submit("tools_setup", {"tools_dir": str(tools), "uv": str(bin_dir / "no-uv"), "git": str(bin_dir / "git")})
    end = time.monotonic() + 30
    while jc.status(jid)["state"] not in ("done", "error") and time.monotonic() < end:
        time.sleep(0.05)
    assert jc.status(jid)["state"] == "error" and "uv not found" in str(jc.status(jid))


def test_outside_pieces_are_only_checked(tmp_path):
    out = setup.outside({"sdk_dir": str(tmp_path / "sdk"), "openmv_dir": str(tmp_path / "omv"),
                         "docker_image": "no-such-image:test"})
    assert [o["ok"] for o in out] == [False, False, False] and "ST Edge AI Core 4.0.0" in out[0]["name"]
    (tmp_path / "sdk").mkdir()
    (tmp_path / "omv" / "tools").mkdir(parents=True)
    (tmp_path / "omv" / "tools" / "modelc.py").write_text("")
    out = setup.outside({"sdk_dir": str(tmp_path / "sdk"), "openmv_dir": str(tmp_path / "omv")})
    assert [o["ok"] for o in out][:2] == [True, True]


def test_page_shows_the_tools_and_the_button_queues_the_job(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.web import server
    data = tmp_path / "data"
    data.mkdir()
    cfg = tmp_path / "config.local.toml"
    cfg.write_text(f'data_dir = "{data}"\nvideo_roots = []\ntools_dir = "{tmp_path / "tools"}"\n')
    monkeypatch.setattr(setup, "build_job", lambda p: {**p, "uv": "/no/uv", "git": "/no/git"})
    app = server.create_app(cfg)
    try:
        client = TestClient(app, base_url="http://127.0.0.1:8000", headers={"origin": "http://127.0.0.1:8000"})
        page = client.get("/fit").text
        assert "Set up the tools" in page and "convert-venv" in page and "not ready" in page
        assert "from Nick's earlier work, not made here" in page
        r = client.post("/fit/setup", follow_redirects=False)
        assert r.status_code == 303 and "Setting+the+tools+up" in r.headers["location"]
        assert app.state.jobctl(server.config.load(cfg)).latest("tools_setup") is not None
    finally:
        app.state.close_jobs()


def test_docker_is_asked_at_most_once_a_minute(monkeypatch, tmp_path):
    """QE PR 82 nit 3: the page reloads every 10 s while cells wait; a slow Docker must not stall each load."""
    calls = []

    class Done:
        returncode = 0

    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(setup.subprocess, "run", lambda cmd, **k: calls.append(cmd) or Done())
    setup._docker_seen.clear()
    fit = {"docker_image": "an-image:1"}
    assert setup.outside(fit)[-1]["ok"] and setup.outside(fit)[-1]["ok"] and len(calls) == 1
    setup._docker_seen["an-image:1"] = (setup.time.monotonic() - setup.DOCKER_CHECK_SECONDS - 1, True)
    setup.outside(fit)
    assert len(calls) == 2
