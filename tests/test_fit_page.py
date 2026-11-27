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
    assert table.headline(big, "ae3")[0] == "no" and "larger than the model area" in table.headline(big, "ae3")[1]
    assert table.headline({"converts": False, "stage": "8-bit model"}, "n6") == ("no", "stopped at the 8-bit model step")
    pi = {"converts": True, "file_bytes": 10, "limit_bytes": None, "placement": placed(0, 0, 30, "operators"),
          "failing_layers": []}
    assert table.headline(pi, "pi")[0] == "runs" and "bench Pi" in table.headline(pi, "pi")[1]
    assert table.headline({**pi, "failing_layers": [{"name": "FlexErf"}]}, "pi")[0] == "partly"
    assert table.mib(None) == "–" and table.mib(2048) == "2 KiB" and table.mib(3 * 1048576) == "3.0 MiB"


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
    page = client.get("/fit").text
    assert "What runs where" in page and "OpenMV N6" in page and "Pi Zero 2W" in page and "OpenMV AE3" in page
    assert "runs, partly on the CPU" in page and "RESIZE_BILINEAR" in page and "56.61 ms" in page and "estimate" in page
    assert "INTERNAL ERROR: &#39;NoneType&#39; object is not subscriptable" in page and "does not run" in page
    assert "earlier work: YOLOX-Nano 10.6 ms" in page and "2026-08-24" in page
    assert "Roboflow licence: evaluation only" in page            # the PML sizes are always marked
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

