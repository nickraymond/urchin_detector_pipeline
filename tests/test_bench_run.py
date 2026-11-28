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
                 '[bench]\nfield_units = ["field-pi"]\n[bench.units.bench-a]\nssh_host = "h"\ncameras = ["n6", "ae3"]\n')
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
    rows = riglog.read(data)                                     # newest first: the N6 is left as found (Lead call)
    assert [r["what"] for r in rows] == ["restore ROMFS0 from the backup (DFU alt 3)", "load yolox_nano, yolox_tiny"]
    assert len(run["restores"]) == 1 and run["finish"]["hard_reset"] is True      # then main.py runs again


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
    n = new_card(data, cfg.bench, unit="bench-a", board="ae3")   # the AE3: refused before any link (the N6 has a route)
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


def test_a_restore_only_n6_card_restores_and_times_nothing(site, jc):
    """Lead call (lead 2, PR 123): the rehearsal card before card 4."""
    cfg, data = site
    n = new_card(data, cfg.bench, models="", restore_only="on")
    assert card.load(data, n)["restore_only"] is True and card.load(data, n)["models"] == []
    p = runs.params(cfg, n)
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "done" and "ROMFS0 restored from the backup" in job["message"], job
    run = json.loads((runs.run_dir(data, p["run_id"]) / "run.json").read_text())
    assert run["outcome"] == "done" and len(run["restores"]) == 1 and runs.windows_done(runs.run_dir(data, p["run_id"])) == []
    assert [r["what"] for r in riglog.read(data)] == ["restore ROMFS0 from the backup (DFU alt 3)"]


@pytest.mark.parametrize("over, why", [({"board": "pi", "restore_only": "on", "models": ""}, "only an N6 card"),
                                       ({"restore_only": "on"}, "lists no model files")])
def test_restore_only_is_for_n6_cards_without_models(site, over, why):
    cfg, data = site
    with pytest.raises(card.CardError, match=why):
        new_card(data, cfg.bench, **over)


def test_a_stop_rule_still_ends_with_the_restore_and_a_failed_end_restore_is_loud(site, jc):
    cfg, data = site
    n = new_card(data, cfg.bench)
    p = runs.params(cfg, n, fake={"fail": {"time": "no_answer"}})
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "error" and "stop rule no_answer" in job["message"]
    assert json.loads((runs.run_dir(data, p["run_id"]) / "run.json").read_text())["restores"]
    p = runs.params(cfg, n, fake={"fail": {"restore": "no_answer"}})
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "error" and "stop rule restore_failed" in job["message"] and "end of the card" in job["message"]
    assert json.loads((runs.run_dir(data, p["run_id"]) / "run.json").read_text())["outcome"] == "stopped: restore_failed"


def test_a_plain_error_after_the_load_still_ends_restored(site, jc):
    """QE PR 123 ask 5: an exit that is not a BoardError (a full disk, a bad answer) still runs the end restore."""
    cfg, data = site
    n = new_card(data, cfg.bench)
    p = runs.params(cfg, n, fake={"fail": {"time": "plain_error"}})
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "error" and "did not report" in job["message"] and "put back from the backup" in job["message"]
    run = json.loads((runs.run_dir(data, p["run_id"]) / "run.json").read_text())
    assert run["outcome"] == "stopped: error" and len(run["restores"]) == 1


def test_a_failed_hard_reset_at_the_end_is_a_stop_rule_and_restore_only_ends_with_one(site, jc):
    """Lead 2, after PR 123: every N6 card ends with a hard reset of the booted board."""
    cfg, data = site
    p = runs.params(cfg, new_card(data, cfg.bench), fake={"fail": {"finish": "no_answer"}})
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "error" and "stop rule no_answer" in job["message"] and "hard reset failed" in job["message"]
    run = json.loads((runs.run_dir(data, p["run_id"]) / "run.json").read_text())
    assert run["outcome"] == "stopped: no_answer" and len(run["restores"]) == 1       # restored first, then the reset
    p = runs.params(cfg, new_card(data, cfg.bench, models="", restore_only="on"))
    assert wait(jc, jc.submit("bench", p), "done", "error")["state"] == "done"
    assert json.loads((runs.run_dir(data, p["run_id"]) / "run.json").read_text())["finish"]["hard_reset"] is True



def test_a_card_whose_board_never_answered_is_not_reset(site, jc):
    """QE PR 126 nit 2: no hard reset, and no second error, when the board never answered."""
    cfg, data = site
    p = runs.params(cfg, new_card(data, cfg.bench), fake={"fail": {"info": "no_answer"}})
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "error" and "stop rule no_answer" in job["message"] and "hard reset" not in job["message"]
    assert "finish" not in json.loads((runs.run_dir(data, p["run_id"]) / "run.json").read_text())


@pytest.mark.parametrize("over, why", [({"rom_models": "yolov8n_192.tflite", "board": "pi", "models": ""}, "only an N6 card"),
                                       ({"rom_models": "yolov8n_192.tflite"}, "not both"),
                                       ({"rom_models": "../x.tflite", "models": ""}, "a file name ending in .tflite"),
                                       ({"rom_models": "a.tflite\na.tflite", "models": ""}, "listed twice")])
def test_rom_models_are_for_n6_cards_alone(site, over, why):
    cfg, data = site
    with pytest.raises(card.CardError, match=why):
        new_card(data, cfg.bench, **over)


def test_a_card_of_models_already_in_rom_runs_with_no_write(site, jc):
    cfg, data = site
    n = new_card(data, cfg.bench, models="", rom_models="yolov8n_192.tflite")
    assert card.load(data, n)["rom_models"] == ["yolov8n_192.tflite"] and card.load(data, n)["models"] == []
    p = runs.params(cfg, n)
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "done", job
    run = json.loads((runs.run_dir(data, p["run_id"]) / "run.json").read_text())
    assert run["route"] == "already in /rom (no write)" and run["restores"] == [] and run["finish"]["hard_reset"]
    assert len(runs.windows_done(runs.run_dir(data, p["run_id"]))) == 3
    assert riglog.read(data) == []                           # nothing written: no rig change



def test_csi_first_is_an_n6_option_and_reaches_a_real_n6_only():
    from app.bench.worker import board_settings
    real = units.parse({"field_units": [], "n6_dfu_serial": "ABCDEF0123456789",
                        "units": {"b": {"ssh_host": "b", "cameras": ["n6"]}}}).unit("b")
    assert board_settings({"board": "n6", "csi_first": True}, real) == {"csi_first": True}
    assert board_settings({"board": "n6"}, real) == {} and board_settings({"board": "n6", "csi_first": True},
                                                                          units.Bench().unit("fake")) == {}


@pytest.mark.parametrize("over, ok", [({"board": "n6", "csi_first": "on", "models": "", "rom_models": "fomo_face_detection.tflite"}, True),
                                      ({"board": "pi", "csi_first": "on"}, False),
                                      ({"csi_first": "on", "restore_only": "on", "models": ""}, False)])
def test_the_card_takes_csi_first_on_an_n6_that_times_something(site, over, ok):
    cfg, data = site
    if ok:
        assert card.load(data, new_card(data, cfg.bench, **over))["csi_first"] is True
    else:
        with pytest.raises(card.CardError, match="CSI snapshot first"):
            new_card(data, cfg.bench, **over)


def test_a_board_reset_says_whether_romfs0_was_put_back():
    """Lead 2, card 7: a no-write card's board_reset said "the job put ROMFS0 back"."""
    from app.bench.worker import hint
    assert "put ROMFS0 back" in hint("board_reset", {"restores": [{"restored": True}]})
    no_write = hint("board_reset", {"restores": []})
    assert "nothing was restored" in no_write and "put ROMFS0 back" not in no_write


def stock_image(data, names=("fomo_face_detection.tflite", "fomo_face_detection.txt")):
    """A small stand-in for OpenMV's stock romfs0.img under the data folder (D82f)."""
    from app.bench import romfs
    p = data / "bench" / "firmware" / "v0" / "romfs0.img"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(romfs.encode([romfs.Entry(n, n.encode() * 40, romfs.default_align(n)) for n in names]))
    return p


def test_a_card_writes_a_whole_given_romfs0_image_and_times_its_rom_model(site, jc):
    """D82f (Lead call, lead 2): card c1 then card 8 on stock: the given image is written whole, its /rom model timed,
    and the card ends with the backup restored."""
    cfg, data = site
    img = stock_image(data)
    n = new_card(data, cfg.bench, models="", rom_models="fomo_face_detection.tflite", romfs_image=str(img))
    c = card.load(data, n)
    assert c["romfs_image"]["sha256"] == card.sha256_file(img) and c["romfs_image"]["rom"]["fomo_face_detection.txt"] == 23 * 40
    assert c["models"] == [] and c["frames"] == card.FRAMES_N6
    p = runs.params(cfg, n)
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "done", job
    run = json.loads((runs.run_dir(data, p["run_id"]) / "run.json").read_text())
    assert run["image"]["sha256"] == c["romfs_image"]["sha256"] and len(run["restores"]) == 1 and run["finish"]["hard_reset"]
    assert len(runs.windows_done(runs.run_dir(data, p["run_id"]))) == 3
    assert [r["what"] for r in riglog.read(data)] == ["restore ROMFS0 from the backup (DFU alt 3)", "write the ROMFS0 image romfs0.img"]   # newest first


def test_an_image_card_with_nothing_timed_writes_lists_and_restores(site, jc):
    cfg, data = site
    n = new_card(data, cfg.bench, models="", romfs_image=str(stock_image(data)))
    assert card.load(data, n)["frames"].startswith("none: the card writes the image")
    p = runs.params(cfg, n)
    job = wait(jc, jc.submit("bench", p), "done", "error")
    assert job["state"] == "done", job
    run = json.loads((runs.run_dir(data, p["run_id"]) / "run.json").read_text())
    assert len(run["restores"]) == 1 and runs.windows_done(runs.run_dir(data, p["run_id"])) == []


@pytest.mark.parametrize("over, why", [({"board": "pi", "models": ""}, "only an N6 card"),
                                       ({"restore_only": "on", "models": ""}, "only an N6 card"),
                                       ({}, "not both"),
                                       ({"models": "", "rom_models": "yolov8n_192.tflite"}, "is not in the ROMFS0 image"),
                                       ({"models": "", "romfs_image": "/etc/hosts"}, "not under the data folder"),
                                       ({"models": "", "romfs_image": "fit/yolox_nano/int8/model_int8.tflite"}, "not a ROMFS image")])
def test_a_romfs0_image_card_is_checked_when_saved(site, over, why):
    cfg, data = site
    with pytest.raises(card.CardError, match=why):
        new_card(data, cfg.bench, **{"romfs_image": str(stock_image(data)), **over})


def test_an_image_changed_after_the_card_is_refused_at_run(site, jc):
    cfg, data = site
    img = stock_image(data)
    n = new_card(data, cfg.bench, models="", rom_models="fomo_face_detection.tflite", romfs_image=str(img))
    img.write_bytes(img.read_bytes() + b"x")
    job = wait(jc, jc.submit("bench", runs.params(cfg, n)), "done", "error")
    assert job["state"] == "error" and "changed or went missing" in job["message"]
    assert riglog.read(data) == []


def test_the_card_page_says_an_image_cards_rom_models_are_in_its_image(site, tmp_path):
    cfg, data = site
    n = new_card(data, cfg.bench, models="", rom_models="fomo_face_detection.tflite", romfs_image=str(stock_image(data)))
    c = TestClient(server.create_app(tmp_path / "config.local.toml"), base_url="http://127.0.0.1:8000",
                   headers={"origin": "http://127.0.0.1:8000"})
    page = c.get(f"/bench/cards/{n}").text
    assert "in the ROMFS0 image below: timed once it is written" in page and "timed with no write" not in page
    assert "written whole in place of our built image" in page and "12,577,076" not in page    # the stand-in's size
