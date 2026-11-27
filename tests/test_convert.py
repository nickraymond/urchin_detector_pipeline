"""S6 bite 3a: the `convert` job end to end on app/jobctl, with a fake 8-bit step (tests/fake_int8.py)
and a fake compiler (tests/fake_compiler.py). No torch, no tool environment, no hardware."""

import json
import sys
import time
from pathlib import Path

import pytest

from app import modeldefs
from app.fit import convert
from app.jobctl import JobControl, JobError
from app.modeldefs import yolox


@pytest.fixture
def jc(tmp_path):
    c = JobControl(tmp_path / "jobs")
    c.register(convert.convert_kind())
    yield c
    c.close()


def cell(tmp_path, camera="fakecam", fake=None, **kw):
    arch = {**yolox.arch("nano", 256, modeldefs.BUILD2_CLASSES), **({"fake": fake} if fake else {})}
    p = convert.params(arch, camera, tmp_path, tmp_path / "tools")
    return {**p, "compiler": "tests.fake_compiler", "tool_python": sys.executable,
            "int8_module": "tests.fake_int8", **kw}


def wait_state(jc, jid, *states, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s = jc.status(jid)
        if s["state"] in states:
            return s
        time.sleep(0.05)
    raise AssertionError(f"timed out in state {jc.status(jid)['state']}")


def calls(folder: Path) -> int:
    log = folder / "int8" / "calls.log"
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_a_cell_converts_and_writes_a_fit_report(jc, tmp_path):
    p = cell(tmp_path)
    folder = Path(p["folder"])
    assert folder == tmp_path / "fit" / "yolox-nano-conv-256-c5" and convert.read_fit(folder, "fakecam") is None
    assert convert.progress(p) == (0, 2)
    s = wait_state(jc, jc.submit("convert", p), "done", "error")
    assert s["state"] == "done" and (s["done"], s["total"]) == (2, 2)
    fit = convert.read_fit(folder, "fakecam")
    assert fit["converts"] is True and fit["stage"] == "compile"
    assert fit["placement"] == {"npu": 9, "hybrid": 0, "cpu": 0, "total": 9, "unit": "operators"}
    assert (fit["arch_key"], fit["camera"], fit["kind"], fit["licence"]) == (
        "yolox-nano-conv-256-c5", "fakecam", "detector", "Apache-2.0")
    assert fit["file_bytes"] == 10 and fit["limit_bytes"] == 1000 and fit["estimate_ms"] == 12.5
    assert fit["int8"]["tflite_bytes"] == 4 and fit["tool"] == "fake compiler 1.0"
    assert json.loads((folder / "recipe.json").read_text())["arch"]["family"] == "yolox"
    assert "[fake to_int8] start" in (folder / "int8" / "convert.log").read_text()
    assert convert.progress(p) == (2, 2)


def test_second_camera_reuses_the_8_bit_model_and_a_rerun_repeats_nothing(jc, tmp_path):
    folder = Path(cell(tmp_path)["folder"])
    wait_state(jc, jc.submit("convert", cell(tmp_path)), "done")
    wait_state(jc, jc.submit("convert", cell(tmp_path, camera="othercam")), "done")
    wait_state(jc, jc.submit("convert", cell(tmp_path)), "done")
    assert calls(folder) == 1
    assert convert.read_fit(folder, "othercam")["converts"] is True
    wait_state(jc, jc.submit("convert", cell(tmp_path, redo=True)), "done")
    assert calls(folder) == 1 and convert.read_fit(folder, "fakecam")["converts"]   # redo cleared int8/, ran once more


def test_layers_on_the_cpu_are_named(jc, tmp_path):
    p = cell(tmp_path, fake={"cpu_layers": ["transpose_3", "transpose_9"]})
    wait_state(jc, jc.submit("convert", p), "done")
    fit = convert.read_fit(Path(p["folder"]), "fakecam")
    assert fit["converts"] and fit["placement"]["cpu"] == 2
    assert [l["name"] for l in fit["failing_layers"]] == ["transpose_3", "transpose_9"]


def test_a_failed_compile_is_a_result_not_a_job_error(jc, tmp_path):
    p = cell(tmp_path, fake={"compile_fails": "TRANSPOSE is not supported"})
    s = wait_state(jc, jc.submit("convert", p), "done", "error")
    fit = convert.read_fit(Path(p["folder"]), "fakecam")
    assert s["state"] == "done" and fit["converts"] is False and fit["stage"] == "compile"
    assert fit["error"] == "TRANSPOSE is not supported" and fit["failing_layers"][0]["name"] == "transpose_7"


def test_a_failed_8_bit_step_is_a_result_and_is_not_repeated_for_the_next_camera(jc, tmp_path):
    p = cell(tmp_path, fake={"int8_fails": True})
    folder = Path(p["folder"])
    s = wait_state(jc, jc.submit("convert", p), "done", "error")
    fit = convert.read_fit(folder, "fakecam")
    assert s["state"] == "done" and fit["converts"] is False and fit["stage"] == "8-bit model"
    assert "Expand is not supported" in fit["error"] and fit["int8"] is None
    wait_state(jc, jc.submit("convert", cell(tmp_path, camera="othercam", fake={"int8_fails": True})), "done")
    assert calls(folder) == 1 and convert.read_fit(folder, "othercam")["stage"] == "8-bit model"


def test_pause_ends_the_8_bit_step_and_resume_runs_it_again(jc, tmp_path):
    p = cell(tmp_path, fake={"int8_seconds": 30})
    folder = Path(p["folder"])
    jid = jc.submit("convert", p)
    end = time.monotonic() + 20
    while calls(folder) == 0 and time.monotonic() < end:
        time.sleep(0.05)
    jc.pause(jid)
    assert wait_state(jc, jid, "paused")["done"] == 0
    assert not convert.int8_path(folder).exists() and not convert.int8_failed_path(folder).exists()
    assert convert.read_fit(folder, "fakecam") is None
    jc.cancel(jid)
    assert wait_state(jc, jid, "cancelled")
    wait_state(jc, jc.submit("convert", cell(tmp_path)), "done")
    assert calls(folder) == 2 and convert.read_fit(folder, "fakecam")["converts"]


def test_cancel_just_before_the_8_bit_file_is_named_leaves_nothing_to_compile(jc, tmp_path):
    """Cancel kills the whole process group (QE PR 76): the step's file must not exist half-made."""
    p = cell(tmp_path, fake={"before_rename_seconds": 30})
    folder = Path(p["folder"])
    jid = jc.submit("convert", p)
    end = time.monotonic() + 20
    while not (folder / "int8" / "int8.json").exists() and time.monotonic() < end:
        time.sleep(0.05)
    assert (folder / "int8" / "model_int8.tflite.part").exists()
    jc.cancel(jid)
    wait_state(jc, jid, "cancelled")
    assert not convert.int8_path(folder).exists() and convert.read_fit(folder, "fakecam") is None
    assert convert.progress(p) == (0, 2)
    wait_state(jc, jc.submit("convert", cell(tmp_path)), "done")       # the step runs again, whole
    assert calls(folder) == 2 and convert.read_fit(folder, "fakecam")["int8"]["tflite_bytes"] == 4
    assert not (folder / "int8" / "model_int8.tflite.part").exists()


def test_missing_tool_environment_fails_the_job_with_a_fix(jc, tmp_path):
    p = cell(tmp_path)
    del p["tool_python"]
    s = wait_state(jc, jc.submit("convert", p), "done", "error")
    assert s["state"] == "error" and "set the tools up" in json.dumps(s)
    assert convert.read_fit(Path(p["folder"]), "fakecam") is None


def test_submit_refuses_a_bad_recipe_and_an_unknown_camera(jc, tmp_path):
    bad = cell(tmp_path)
    bad["arch"] = {**bad["arch"], "input": [250, 250]}
    assert "multiples of 32" in str(pytest.raises(JobError, convert.build_job, bad).value)
    p = cell(tmp_path, camera="pi")
    del p["compiler"]
    assert "No compiler is set up" in str(pytest.raises(JobError, convert.build_job, p).value)
    assert convert.build_job(cell(tmp_path))["tool_python"] == sys.executable
    p = cell(tmp_path)
    del p["tool_python"]
    assert convert.build_job(p)["tool_python"] == str(tmp_path / "tools" / "convert-venv" / "bin" / "python")


def test_tools_dir_setting(tmp_path):
    from app import config
    f = tmp_path / "config.local.toml"
    f.write_text(f'data_dir = "{tmp_path}"\n')
    assert config.load(f).tools_dir == config.DEFAULT_TOOLS_DIR
    f.write_text(f'data_dir = "{tmp_path}"\ntools_dir = "{tmp_path}/t"\n')
    assert config.load(f).tools_dir == tmp_path / "t"
    f.write_text(f'data_dir = "{tmp_path}"\ntools_dir = 5\n')
    cfg = config.load(f)
    assert cfg.ok and cfg.tools_dir == config.DEFAULT_TOOLS_DIR and "tools_dir" in cfg.warnings[0]



def test_the_8_bit_step_records_this_repos_own_commit(tmp_path):
    from app.fit import to_int8
    assert to_int8.repo_version(tmp_path) == {"commit": None, "dirty": None}    # not a git checkout
    (tmp_path / ".git").mkdir()                                                   # looks like one, git fails
    assert to_int8.repo_version(tmp_path) == {"commit": None, "dirty": None}
    if not (Path(to_int8.__file__).resolve().parents[2] / ".git").exists():
        pytest.skip("this copy of the code is not a git checkout")
    v = to_int8.repo_version()
    assert len(v["commit"]) == 40 and isinstance(v["dirty"], bool)
