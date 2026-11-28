"""S7 bite 2b (D83, D84): the `bench` job end to end on the fake board, through JobControl and the card page.
Each test has its own temporary data folder (QE PR 98 nit 3); no hardware, made-up unit names only."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from app import config
from app.bench import card, riglog, units
from app.bench import run as runs
from app.jobctl import JobControl, JobError
from app.web import server


@pytest.fixture
def site(tmp_path):
    data = tmp_path / "data"
    for key in ("yolox_nano", "yolox_tiny"):
        (data / "fit" / key / "int8").mkdir(parents=True)
        (data / "fit" / key / "int8" / "model_int8.tflite").write_bytes(key.encode())
    p = tmp_path / "config.local.toml"
    p.write_text(f'data_dir = "{data}"\nvideo_roots = []\n'
                 '[bench]\nfield_units = ["field-pi"]\n[bench.units.bench-a]\nssh_host = "h"\ncameras = ["n6"]\n')
    return config.load(p), data


def new_card(data, bench, reviewed=True, **over):
    form = {"question": "q", "unit": "fake", "board": "n6", "windows": "3", "n": "50", "cost_minutes": "1",
            "models": "fit/yolox_nano/int8/model_int8.tflite\nfit/yolox_tiny/int8/model_int8.tflite",
            "pass_bar": "p", "stop_rules": "a rule\n" + card.RESTORE_RULE, "written_by": "lead", **over}
    n = card.save(data, card.build(data, bench, form))
    if reviewed:
        card.review(data, n, "qe")
    return n


@pytest.fixture
def jc(site):
    j = JobControl(site[1] / "jobs")
    j.register(runs.bench_kind())
    yield j
    j.close()


def wait(jc, jid, *states, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if jc.status(jid)["state"] in states:
            return jc.status(jid)
        time.sleep(0.05)
    raise AssertionError(f"job {jid} is {jc.status(jid)['state']}, not {states}")


def test_run_end_to_end(site, jc):
    cfg, data = site
    n = new_card(data, cfg.bench)
    p = runs.params(cfg, n, fake={"ms": {"yolox_nano": 10.6, "yolox_tiny": 20.0}})
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "done", job
    folder = runs.run_dir(data, p["run_id"])
    assert json.loads((folder / "card.json").read_text())["sha256"] == card.load(data, n)["sha256"]
    run = json.loads((folder / "run.json").read_text())
    assert run["outcome"] == "done" and run["board_info"]["kind"] == "n6" and run["route"] == "fake"
    lines = runs.windows_done(folder)
    assert len(lines) == 6 and {(w["window"], w["model"]) for w in lines} == {
        (w, m) for w in (1, 2, 3) for m in ("yolox_nano", "yolox_tiny")}
    s = runs.summary(folder)["models"]
    assert 10.6 * 0.95 <= s["yolox_nano"]["median_ms"] <= 10.6 * 1.05 and s["yolox_tiny"]["windows"] == 3
    assert "yolox_nano: median" in job["message"]
    assert riglog.read(data)[0]["what"] == "load yolox_nano, yolox_tiny"


def test_run_is_refused_without_a_review_or_on_a_changed_card(site):
    cfg, data = site
    n = new_card(data, cfg.bench, reviewed=False)
    with pytest.raises(JobError, match="not reviewed yet"):
        runs.params(cfg, n)
    card.review(data, n, "qe")
    path = card.folder(data) / f"{n}.json"
    path.write_text(path.read_text().replace('"q"', '"q2"'))
    with pytest.raises(JobError, match="changed after its review"):
        runs.params(cfg, n)
    with pytest.raises(JobError, match="No test card 9"):
        runs.params(cfg, 9)


def test_real_unit_card_ends_as_refused_and_fake_settings_are_fake_only(site, jc):
    cfg, data = site
    n = new_card(data, cfg.bench, unit="bench-a")
    with pytest.raises(JobError, match="fake unit only"):
        runs.params(cfg, n, fake={"ms": {}})
    job = wait(jc, jc.submit("bench", runs.params(cfg, n)), "done", "error")
    assert job["state"] == "error" and "not built yet" in job["message"]
    outcomes = [json.loads(f.read_text())["outcome"] for f in (data / "bench").glob("*/run.json")]
    assert outcomes == ["stopped: refused"]                      # QE PR 112 nit 2: recorded, not only a job error


def test_run_refuses_a_real_unit_card_without_the_restore_rule(site):     # QE PR 101 nit 1, D87
    cfg, data = site
    n = new_card(data, cfg.bench, unit="bench-a", reviewed=False)
    path = card.folder(data) / f"{n}.json"
    body = json.loads(path.read_text())
    body["stop_rules"] = ["a rule"]                     # written by hand, past the page's check
    path.write_text(json.dumps(body))
    card.review(data, n, "qe")
    with pytest.raises(JobError, match="RESTORE.md matches the rig log"):
        runs.params(cfg, n)


def test_model_changed_after_the_card_stops_the_run(site, jc):
    cfg, data = site
    n = new_card(data, cfg.bench)
    (data / "fit" / "yolox_tiny" / "int8" / "model_int8.tflite").write_bytes(b"retrained")
    job = wait(jc, jc.submit("bench", runs.params(cfg, n)), "done", "error")
    assert job["state"] == "error" and "changed since card" in job["message"]


def test_stop_rule_is_a_job_error_with_a_hint_and_keeps_finished_windows(site, jc):
    cfg, data = site
    n = new_card(data, cfg.bench)
    p = runs.params(cfg, n, fake={"fail": {"time": "no_answer"}, "fail_at": 4})
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "error" and "stop rule no_answer" in job["message"] and "cable and power" in job["message"]
    folder = runs.run_dir(data, p["run_id"])
    assert len(runs.windows_done(folder)) == 3
    assert json.loads((folder / "run.json").read_text())["outcome"] == "stopped: no_answer"
    assert runs.summary(folder)["models"]["yolox_nano"]["median_ms"] is None     # 2 windows: no median (D83)


def test_pause_after_a_window_then_resume_skips_what_is_done(site, jc):
    cfg, data = site
    n = new_card(data, cfg.bench, windows="4")
    p = runs.params(cfg, n, fake={"delay_s": 0.3})
    jid = jc.submit("bench", p)
    folder = runs.run_dir(data, p["run_id"])
    end = time.monotonic() + 20
    while len(runs.windows_done(folder)) < 1 and time.monotonic() < end:
        time.sleep(0.05)
    jc.pause(jid)
    wait(jc, jid, "paused")
    first = runs.windows_done(folder)
    assert 1 <= len(first) < 8 and len(first) % 2 == 0          # Pause is taken after a whole window
    jc.resume(jid)
    assert wait(jc, jid, "done", "error")["state"] == "done"
    lines = runs.windows_done(folder)
    assert len(lines) == 8 and len({(w["window"], w["model"]) for w in lines}) == 8   # nothing timed twice


def test_cancel_keeps_finished_windows(site, jc):
    cfg, data = site
    n = new_card(data, cfg.bench, windows="5")
    p = runs.params(cfg, n, fake={"delay_s": 0.3})
    jid = jc.submit("bench", p)
    folder = runs.run_dir(data, p["run_id"])
    end = time.monotonic() + 20
    while len(runs.windows_done(folder)) < 2 and time.monotonic() < end:
        time.sleep(0.05)
    jc.cancel(jid)
    wait(jc, jid, "cancelled")
    kept = runs.windows_done(folder)
    assert 2 <= len(kept) < 10
    assert runs.progress(p) == (len(kept), 10)


def test_card_page_runs_a_reviewed_card(tmp_path):
    data = tmp_path / "data"
    (data / "fit" / "yolox_nano" / "int8").mkdir(parents=True)
    (data / "fit" / "yolox_nano" / "int8" / "model_int8.tflite").write_bytes(b"m")
    p = tmp_path / "config.local.toml"
    p.write_text(f'data_dir = "{data}"\nvideo_roots = []\n')
    n = new_card(data, units.Bench(), reviewed=False, models="fit/yolox_nano/int8/model_int8.tflite")
    app = server.create_app(p)
    c = TestClient(app, base_url="http://127.0.0.1:8000", headers={"origin": "http://127.0.0.1:8000"})
    try:
        page = c.get(f"/bench/cards/{n}").text
        assert "Run this card" not in page and "Run is refused until" in page
        r = c.post(f"/bench/cards/{n}/run", follow_redirects=True)
        assert "not reviewed yet" in r.text
        card.review(data, n, "qe")
        r = c.post(f"/bench/cards/{n}/run", follow_redirects=True)
        assert "Run queued" in r.text
        end = time.monotonic() + 30
        while "done" not in (page := c.get(f"/bench/cards/{n}").text) and time.monotonic() < end:
            time.sleep(0.1)
        assert ">done</a>" in page and "yolox_nano" in page and "3 of 3" in page
    finally:
        app.state.close_jobs()
