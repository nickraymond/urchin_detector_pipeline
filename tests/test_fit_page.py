"""S6 bite 4a: the "What runs where" page and its table logic, on fit.json files written by hand and on
the convert job with the fake 8-bit step and fake compiler. No tools, no torch, no hardware."""

import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

from app import modeldefs
from app.fit import convert, table
from app.web import server

N6_AREA = 0x01800000


def write_fit(data_dir, key, camera, **body):
    path = data_dir / "fit" / key / "targets" / camera / "fit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"arch_key": key, "camera": camera, "tool": "a compiler 1.0", **body}))


@pytest.fixture
def site(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    cfg = tmp_path / "config.local.toml"
    cfg.write_text(f'data_dir = "{data}"\nvideo_roots = []\n')
    real = convert.params
    monkeypatch.setattr(convert, "params", lambda *a, **k: {
        **real(*a, **k), "tool_python": sys.executable, "int8_module": "tests.fake_int8",
        "compiler": "tests.fake_compiler"})
    app = server.create_app(cfg)
    client = TestClient(app, base_url="http://127.0.0.1:8000", headers={"origin": "http://127.0.0.1:8000"})
    yield client, data
    app.state.close_jobs()


def placed(npu, hybrid, cpu, unit="epochs"):
    return {"npu": npu, "hybrid": hybrid, "cpu": cpu, "total": npu + hybrid + cpu, "unit": unit}


def test_headlines():
    ok = {"converts": True, "file_bytes": 1000, "limit_bytes": N6_AREA}
    assert table.headline(None, "n6") == ("not_tried", "")
    assert table.headline({**ok, "placement": placed(117, 2, 0)}, "n6") == ("runs", "all on the NPU (2 hybrid steps)")
    assert table.headline({**ok, "placement": placed(378, 0, 0, "operators")}, "ae3") == ("runs", "all on the NPU")
    assert table.headline({**ok, "placement": placed(143, 0, 20)}, "n6") == ("partly", "20 of 163 steps on the CPU")
    big = {**ok, "file_bytes": N6_AREA + 1, "placement": placed(5, 0, 0)}
    assert table.headline(big, "ae3")[0] == "no" and "too big for the model area" in table.headline(big, "ae3")[1]
    code, why = table.headline({"converts": False, "stage": "8-bit model"}, "n6")
    assert code == "no" and "not a compiler verdict" in why
    assert table.label(code, {"converts": False, "stage": "8-bit model"}) == table.NO_ROUTE
    st = {"converts": False, "stage": "compile", "tool": "ST Edge AI Core v4.0.0", "error": "INTERNAL ERROR: x"}
    assert table.headline(st, "n6") == ("no", "the compiler stopped with an internal error (ST Edge AI Core v4.0.0)")
    assert table.label("no", st) == "does not run"
    assert "with an error (Arm Vela 5.0.0)" in table.headline({**st, "tool": "Arm Vela 5.0.0", "error": "bad"}, "ae3")[1]
    pi = {"converts": True, "file_bytes": 10, "limit_bytes": None, "placement": placed(0, 0, 30, "operators"),
          "failing_layers": []}
    assert table.headline(pi, "pi")[0] == "runs" and "bench Pi" in table.headline(pi, "pi")[1]
    assert table.headline({**pi, "failing_layers": [{"name": "FlexErf"}]}, "pi")[0] == "partly"
    assert table.mib(None) == "–" and table.mib(2048) == "2 KiB" and table.mib(3 * 1048576) == "3.0 MiB"


def test_headline_checks_memory_against_the_cameras_figure():
    """QE PR 80 nit 1: a model that fits the model area but not the memory is not "runs"."""
    ref = {"bytes": 4_090_000, "source": "earlier work"}
    fit = {"converts": True, "file_bytes": 1000, "limit_bytes": N6_AREA, "placement": placed(378, 0, 0, "operators")}
    assert table.headline({**fit, "memory_bytes": 4_000_000, "memory_reference": ref}, "ae3")[0] == "runs"
    code, why = table.headline({**fit, "memory_bytes": 96_154_419, "memory_reference": ref}, "ae3")
    assert code == "no" and "91.7 MiB of memory against 3.9 MiB" in why and "not re-measured" in why
    code, why = table.headline({**fit, "file_bytes": N6_AREA + 1, "placement": placed(3789, 0, 293, "operators")}, "ae3")
    assert code == "no" and "293 of 4082 operators on the CPU" in why and "too big for the model area" in why
    both = table.headline({**fit, "file_bytes": N6_AREA + 1, "memory_bytes": 96_154_419, "memory_reference": ref}, "ae3")
    assert both[0] == "no" and "too big for the model area" in both[1] and "needs 91.7 MiB of memory" in both[1]
    assert table.headline({**fit, "memory_bytes": 10**9}, "n6")[0] == "runs"      # the N6 has no figure yet: not checked
    assert table.memory_figure(fit) is None
    pi = {**fit, "memory_bytes": 600 * 2**20, "memory_limit_bytes": 512 * 2**20}
    assert table.headline(pi, "pi")[0] == "no"


def test_a_roboflow_licensed_candidate_is_never_a_cells_best(tmp_path):
    """QE PR 80 nit 2 (SPEC hard constraint 6): the headline ranks permissive candidates only."""
    write_fit(tmp_path, "rfdetrplus-atto-380-c5", "n6", converts=True, file_bytes=1, limit_bytes=N6_AREA,
              placement=placed(10, 0, 0))
    write_fit(tmp_path, "rfdetr-nano-384-c5", "n6", converts=False, stage="compile", error="INTERNAL ERROR")
    cell = {r["kind"]: r for r in table.table(tmp_path)}["detector"]["cells"]["n6"]
    assert cell["code"] == "no" and cell["best"]["key"] == "rfdetr-nano-384-c5"
    assert cell["best_marked"]["key"] == "rfdetrplus-atto-380-c5" and cell["best_marked"]["code"] == "runs"
    assert {r["kind"]: r for r in table.table(tmp_path)}["detector"]["cells"]["ae3"]["best_marked"] is None


def test_table_cell_headline_is_its_best_candidate(tmp_path):
    write_fit(tmp_path, "yolox-nano-conv-256-c5", "n6", converts=True, file_bytes=1, limit_bytes=N6_AREA,
              placement=placed(117, 2, 0))
    write_fit(tmp_path, "rfdetr-nano-384-c5", "n6", converts=False, stage="compile", error="INTERNAL ERROR")
    rows = {r["kind"]: r for r in table.table(tmp_path, active={("yolox-tiny-conv-256-c5", "n6")})}
    cell = rows["detector"]["cells"]["n6"]
    assert cell["code"] == "runs" and cell["best"]["key"] == "yolox-nano-conv-256-c5"
    by_key = {c["key"]: c for c in cell["candidates"]}
    assert by_key["rfdetr-nano-384-c5"]["code"] == "no" and by_key["yolox-tiny-conv-256-c5"]["active"]
    assert [e["ms"] for e in cell["earlier"]] == [10.6, 31.2]
    assert rows["segmenter"]["cells"]["pi"]["code"] == "not_tried" and rows["segmenter"]["cells"]["pi"]["earlier"] == []
    everything = len(modeldefs.candidates()) * 3
    assert len(table.missing(tmp_path)) == everything - 2


def test_page_shows_results_reasons_marks_and_earlier_work(site):
    client, data = site
    write_fit(data, "tvseg-lraspp-mnv3l-256-c6", "ae3", converts=True, file_bytes=2851504, limit_bytes=N6_AREA,
              memory_bytes=1313984, estimate_ms=56.61, placement=placed(193, 0, 1, "operators"),
              failing_layers=[{"name": "resize_1", "type": "RESIZE_BILINEAR", "why": "Unsupported opType"}])
    write_fit(data, "rfdetr-nano-384-c5", "n6", converts=False, stage="compile",
              error="INTERNAL ERROR: 'NoneType' object is not subscriptable", failing_layers=[])
    write_fit(data, "rfdetrplus-atto-380-c5", "n6", converts=False, stage="8-bit model", error="attn/MatMul failed")
    page = client.get("/fit").text
    assert "does not convert with today&#39;s route (8-bit step)" in page and "not a compiler verdict" in page
    assert "<details><summary>the error text</summary>" in page and "attn/MatMul failed" in page
    assert "evaluation only</b>: rfdetrplus-atto-380-c5 does not convert" in page    # its own line in the cell
    assert page.count("how its 8-bit file is made") >= 6 and "Static (full-integer) INT8 is not supported" in page
    assert "mJ gross" in page and "no memory figure recorded for this camera yet" in page
    assert "the compiler stopped with an internal error (a compiler 1.0)" in page
    assert "What runs where" in page and "OpenMV N6" in page and "Pi Zero 2W" in page and "OpenMV AE3" in page
    assert "runs, partly on the CPU" in page and "RESIZE_BILINEAR" in page and "56.61 ms" in page and "estimate" in page
    assert "INTERNAL ERROR: &#39;NoneType&#39; object is not subscriptable" in page and "does not run" in page
    assert "earlier work: YOLOX-Nano 10.6 ms" in page and "2026-08-24" in page
    assert "Roboflow licence: evaluation only" in page            # the PML sizes are always marked
    assert page.count("our own untrained design") >= 4            # plainseg and yoloxproto say what they are
    assert "not measurements" in page and "http" not in page.replace("http-equiv", "")   # nothing from the internet
    assert 'href="/fit"' in client.get("/").text


def test_run_all_queues_every_missing_cell_once_and_the_table_fills(site, monkeypatch):
    client, data = site
    few = [a for a in modeldefs.candidates() if a["family"] == "plainseg"]
    monkeypatch.setattr(modeldefs, "candidates", lambda kind=None: [a for a in few if kind in (None, a["kind"])])
    r = client.post("/fit/run-all", follow_redirects=False)
    assert r.status_code == 303 and "6+cells+queued" in r.headers["location"]
    r = client.post("/fit/run-all", follow_redirects=False)
    end = time.monotonic() + 60
    while table.missing(data) and time.monotonic() < end:
        time.sleep(0.2)
    assert table.missing(data) == []
    page = client.get("/fit").text
    assert "Run all (0 cells not tried)" in page and "fake compiler 1.0" in page
    jobs = [j for j in client.app.state.jobctl(server.config.load(data.parent / "config.local.toml")).list()
            if j["kind"] == "convert"]
    assert len(jobs) == 6                                        # the second press queued nothing twice
    assert "Nothing+to+run" in client.post("/fit/run-all", follow_redirects=False).headers["location"]


def convert_jobs(client, data):
    return [j for j in client.app.state.jobctl(server.config.load(data.parent / "config.local.toml")).list()
            if j["kind"] == "convert"]


def wait_all_done(client, data, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        jobs = convert_jobs(client, data)
        if all(j["state"] == "done" for j in jobs):
            return jobs
        time.sleep(0.1)
    raise AssertionError("convert jobs still not done: " + str([j["state"] for j in convert_jobs(client, data)]))


def test_run_again_redoes_every_cell_made_from_the_shared_8_bit_file(site):
    """QE PR 80 nit 4: no camera's cell may keep a result from an 8-bit file that was made again."""
    client, data = site
    folder, key = data / "fit" / "plainseg-s-256-c6", "plainseg-s-256-c6"
    for camera in ("n6", "ae3"):
        client.post("/fit/run", data={"key": key, "camera": camera}, follow_redirects=False)
    assert len(wait_all_done(client, data)) == 2                    # both jobs "done" before Run again is pressed
    first = {c: convert.read_fit(folder, c) for c in ("n6", "ae3")}
    assert all(first.values()) and convert.read_fit(folder, "pi") is None
    r = client.post("/fit/run", data={"key": key, "camera": "n6"}, follow_redirects=False)
    assert "queued+again" in r.headers["location"] and "OpenMV+AE3+with+it" in r.headers["location"]
    assert len(wait_all_done(client, data)) == 4                    # 2 first runs, the redo, the AE3 again
    second = {c: convert.read_fit(folder, c) for c in ("n6", "ae3")}
    assert all(second[c]["when"] >= first[c]["when"] and second[c]["converts"] for c in ("n6", "ae3"))
    assert second["ae3"]["int8"] == second["n6"]["int8"]
    assert (folder / "int8" / "calls.log").read_text() == "int8\n"   # cleared, then made once for both cameras
    assert convert.read_fit(folder, "pi") is None                  # a cell never tried is not queued by this


def test_run_again_keeps_a_cell_whose_job_is_still_under_way(site):
    """QE PR 85: a camera whose job is waiting or running when Run again is pressed has no result yet, or is
    about to write one; the redo wipes it, so that cell must be queued again behind the redo."""
    client, data = site
    folder, key = data / "fit" / "plainseg-s-256-c6", "plainseg-s-256-c6"
    client.post("/fit/run", data={"key": key, "camera": "n6"}, follow_redirects=False)
    wait_all_done(client, data)
    slow = next(a for a in modeldefs.candidates() if modeldefs.arch_key(a) == "plainseg-m-256-c6")
    jc = client.app.state.jobctl(server.config.load(data.parent / "config.local.toml"))
    jc.submit("convert", convert.params({**slow, "fake": {"int8_seconds": 2}}, "n6", data, data / "tools"))
    r = client.post("/fit/run", data={"key": key, "camera": "ae3"}, follow_redirects=False)   # waits behind it
    assert r.headers["location"].endswith("queued.") and convert.read_fit(folder, "ae3") is None
    r = client.post("/fit/run", data={"key": key, "camera": "n6"}, follow_redirects=False)
    assert "queued+again" in r.headers["location"] and "OpenMV+AE3+with+it" in r.headers["location"]
    assert len(wait_all_done(client, data)) == 5       # first n6, the slow one, AE3, the redo, AE3 once more
    n6, ae3 = convert.read_fit(folder, "n6"), convert.read_fit(folder, "ae3")
    assert n6 and ae3 and ae3["converts"] and ae3["int8"] == n6["int8"]      # the AE3 cell is there at the end
    assert (folder / "int8" / "calls.log").read_text() == "int8\n"


def test_run_one_cell_again_and_unknown_cells_are_refused(site, monkeypatch):
    client, data = site
    assert "No+such+cell" in client.post("/fit/run", data={"key": "nope", "camera": "n6"},
                                         follow_redirects=False).headers["location"]
    assert "No+such+cell" in client.post("/fit/run", data={"key": "plainseg-s-256-c6", "camera": "moon"},
                                         follow_redirects=False).headers["location"]
    r = client.post("/fit/run", data={"key": "plainseg-s-256-c6", "camera": "ae3"}, follow_redirects=False)
    assert r.headers["location"].endswith("queued.")            # never tried: not "again", no redo
    end = time.monotonic() + 30
    while convert.read_fit(data / "fit" / "plainseg-s-256-c6", "ae3") is None and time.monotonic() < end:
        time.sleep(0.2)
    assert convert.read_fit(data / "fit" / "plainseg-s-256-c6", "ae3")["converts"] is True
    r = client.post("/fit/run", data={"key": "plainseg-s-256-c6", "camera": "ae3"}, follow_redirects=False)
    assert "queued+again" in r.headers["location"]              # it has a result now: from scratch


def test_stored_together_sums_the_best_of_each_kind_against_the_room_left(tmp_path):
    area = N6_AREA
    for key, size in (("yolox-nano-conv-256-c5", 1_200_000), ("yolox-tiny-conv-256-c5", 5_300_000),
                      ("plainseg-s-256-c6", 100_000), ("yoloxproto-nano-conv-256-c5", 1_300_000)):
        write_fit(tmp_path, key, "n6", converts=True, file_bytes=size, limit_bytes=area, placement=placed(9, 2, 0))
    write_fit(tmp_path, "tvseg-deeplabv3-mnv3l-256-c6", "n6", converts=True, file_bytes=11_000_000,
              limit_bytes=area, placement=placed(9, 0, 20))           # partly on the CPU: not the best
    write_fit(tmp_path, "plainseg-s-256-c6", "pi", converts=True, file_bytes=92_912, limit_bytes=None,
              placement=placed(0, 0, 30, "operators"), failing_layers=[])
    write_fit(tmp_path, "rfdetrplus-atto-380-c5", "n6", converts=True, file_bytes=10, limit_bytes=area,
              placement=placed(9, 0, 0))                              # Roboflow licence: never picked (QE nit 2)
    t = table.stored_together(table.table(tmp_path))
    n6 = t["n6"]
    assert table.VENDOR_BYTES == {"n6": 12_577_188, "ae3": 9_269_500}  # image minus its compiled YOLOX files
    assert {k: v["key"] for k, v in n6["best"].items()} == {
        "Detector": "yolox-nano-conv-256-c5", "Segmenter": "plainseg-s-256-c6",
        "Instance segmenter": "yoloxproto-nano-conv-256-c5"}
    assert [c["labels"] for c in n6["combos"]] == [
        ("Detector", "Segmenter"), ("Detector", "Instance segmenter"), ("Segmenter", "Instance segmenter"),
        ("Detector", "Segmenter", "Instance segmenter")]
    three = n6["combos"][-1]
    assert three["bytes"] == 2_600_000 and three["fits"] is True
    assert three["left"] == area - table.VENDOR_BYTES["n6"] - 2_600_000
    assert t["ae3"] == {**t["ae3"], "best": {}, "combos": []}         # nothing runs there yet
    assert list(t["pi"]["best"]) == ["Segmenter"] and t["pi"]["combos"] == []   # one kind only: no pair


def test_stored_together_says_when_it_does_not_fit(tmp_path):
    write_fit(tmp_path, "yolox-tiny-conv-256-c5", "ae3", converts=True, file_bytes=9_000_000, limit_bytes=N6_AREA,
              placement=placed(288, 0, 0, "operators"))
    write_fit(tmp_path, "plainseg-m-256-c6", "ae3", converts=True, file_bytes=8_000_000, limit_bytes=N6_AREA,
              placement=placed(26, 0, 0, "operators"))
    combo = table.stored_together(table.table(tmp_path))["ae3"]["combos"][0]
    assert combo["fits"] is False and combo["left"] == N6_AREA - table.VENDOR_BYTES["ae3"] - 17_000_000


def test_page_has_the_stored_together_table_and_the_time_box_note(site):
    client, data = site
    write_fit(data, "yolox-nano-conv-256-c5", "n6", converts=True, file_bytes=1_194_488, limit_bytes=N6_AREA,
              placement=placed(117, 2, 0))
    write_fit(data, "plainseg-s-256-c6", "n6", converts=True, file_bytes=103_344, limit_bytes=N6_AREA,
              placement=placed(27, 2, 0))
    page = client.get("/fit").text
    assert "stored together" in page and "Detector + Segmenter" in page and "<b>fit</b>" in page
    assert "no candidate runs here yet" in page and "AE3 time box" in page and "measured on the boards (S7)" in page

