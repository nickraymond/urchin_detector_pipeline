"""S7 bite 1 (D81): bench units from [bench], the Board interface, FakeBoard, the rig log and its page.
No hardware: the fake board only (SPEC constraint 9). Unit names here are made up."""

import json

import pytest
from fastapi.testclient import TestClient

from app import config
from app.bench import riglog, units
from app.bench.board import LOAD_FAILED, NOT_LOADED, BoardError, FakeBoard, Model, open_board
from app.web import server


def write_cfg(tmp_path, bench: str = ""):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    p = tmp_path / "config.local.toml"
    p.write_text(f'data_dir = "{data}"\nvideo_roots = []\n{bench}')
    return p, data


BENCH = """
[bench]
field_units = ["field-cam-1", "field-pi"]
[bench.units.bench-a]
ssh_host = "bench-a-alias"
cameras = ["n6", "ae3"]
"""


def test_no_bench_table_means_fake_only(tmp_path):
    cfg = config.load(write_cfg(tmp_path)[0])
    assert cfg.ok and cfg.bench.units == {} and cfg.bench.names() == ["fake"]
    assert cfg.bench.unit("fake").kinds == ("pi", "n6", "ae3")
    with pytest.raises(units.Refused, match="not a bench unit"):
        cfg.bench.unit("bench-a")


def test_listed_unit_and_refusals(tmp_path):
    cfg = config.load(write_cfg(tmp_path, BENCH)[0])
    u = cfg.bench.unit("bench-a")
    assert (u.ssh_host, u.kinds, u.fake) == ("bench-a-alias", ("pi", "n6", "ae3"), False)
    with pytest.raises(units.Refused, match="field unit"):
        cfg.bench.unit("field-cam-1")
    with pytest.raises(units.Refused, match="listed: bench-a"):
        cfg.bench.unit("someone-else")


@pytest.mark.parametrize("body, why", [
    ('[bench.units.x]\nssh_host = "field-pi"\n', "field_units"),          # a field unit by its ssh host
    ('[bench.units.field-pi]\nssh_host = "h"\n', "field_units"),          # a field unit by name
    ('[bench.units.x]\ncameras = ["n6"]\n', "no ssh_host"),
    ('[bench.units.x]\nssh_host = "h"\ncameras = ["n6", "imx708"]\n', "cameras must be"),
    ('[bench.units.x]\nssh_host = "h"\ncameras = ["n6", "n6"]\n', "cameras must be"),
    ('[bench.units.fake]\nssh_host = "h"\n', "fake board"),
])
def test_bad_unit_is_left_out_with_a_warning(tmp_path, body, why):
    cfg = config.load(write_cfg(tmp_path, '[bench]\nfield_units = ["field-pi"]\n' + body)[0])
    assert cfg.ok and cfg.bench.units == {}
    assert any(why in w for w in cfg.warnings), cfg.warnings


def test_bad_field_units_list_uses_no_unit(tmp_path):
    cfg = config.load(write_cfg(tmp_path, '[bench]\nfield_units = "field-pi"\n'
                                          '[bench.units.x]\nssh_host = "h"\n')[0])
    assert cfg.bench.units == {} and any("field_units must be a list" in w for w in cfg.warnings)


def test_real_unit_has_no_board_yet(tmp_path):
    cfg = config.load(write_cfg(tmp_path, BENCH)[0])
    with pytest.raises(units.Refused, match="not built yet"):
        open_board(cfg.bench.unit("bench-a"), "n6", tmp_path)
    with pytest.raises(ValueError):
        FakeBoard(cfg.bench.unit("bench-a"), "n6", tmp_path)


def model(tmp_path, name="yolox_nano", data=b"tflite bytes"):
    p = tmp_path / f"{name}.tflite"
    p.write_bytes(data)
    return Model(name, p)


def test_fake_board_round_trip_and_rig_log(tmp_path):
    board = open_board(units.Bench().unit("fake"), "n6", tmp_path, ms={"yolox_nano": 10.6}, seed=1)
    assert board.info()["kind"] == "n6"
    m = model(tmp_path)
    assert board.load(m)["loaded"] == ["yolox_nano"]
    w1, w2 = board.time("yolox_nano", 300), board.time("yolox_nano", 300)
    assert w1["n"] == 300 and w1["warmups"] == 2
    assert 10.6 * 0.95 <= w1["ms_min"] <= w1["ms_mean"] <= w1["ms_max"] <= 10.6 * 1.05
    assert w1 != w2                                    # windows differ, so a spread is real
    again = open_board(units.Bench().unit("fake"), "n6", tmp_path, ms={"yolox_nano": 10.6}, seed=1)
    again.load(m)
    assert again.time("yolox_nano", 300) == w1          # seeded: repeatable
    board.reset()
    with pytest.raises(BoardError) as e:
        board.time("yolox_nano", 10)
    assert e.value.stop == NOT_LOADED

    rows = riglog.read(tmp_path)
    assert [r["state"] for r in rows] == ["done", "done"]
    assert rows[0]["fake"] and rows[0]["board"] == "n6" and "ROMFS0" in rows[0]["undo"]
    assert rows[0]["sha256"] == {"yolox_nano": m.sha256()}


def test_load_several_and_memory(tmp_path):
    board = FakeBoard(units.Bench().unit("fake"), "ae3", tmp_path, mem_free=4_000_000, model_bytes=1_500_000)
    out = board.load_several([model(tmp_path, "a"), model(tmp_path, "b")])
    assert out["loaded"] == ["a", "b"] and out["mem_free"] == 1_000_000
    assert board.time("a", 5)["mem_after_load"] == 1_000_000


def test_failure_on_request_is_logged_as_failed(tmp_path):
    board = FakeBoard(units.Bench().unit("fake"), "pi", tmp_path, fail={"load": LOAD_FAILED})
    with pytest.raises(BoardError) as e:
        board.load(model(tmp_path))
    assert e.value.stop == LOAD_FAILED
    (row,) = riglog.read(tmp_path)
    assert row["state"] == "failed" and LOAD_FAILED in row["error"]


def test_rig_log_before_is_written_first_and_cut_off_shows(tmp_path):
    with riglog.change(tmp_path, "bench-a", "pi", "install x", "uninstall x") as cid:
        lines = riglog.path(tmp_path).read_text().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["phase"] == "before"
    # A "before" with no "after" (the app was killed mid-change), and a line that cannot be read.
    with open(riglog.path(tmp_path), "a") as f:
        f.write(json.dumps({"change": "c2", "phase": "before", "unit": "bench-a", "what": "copy y",
                            "undo": "delete y", "at": "t"}) + "\n{not json\n")
    rows = riglog.read(tmp_path)
    assert [r["state"] for r in rows] == ["unreadable", "not finished", "done"]
    assert rows[2]["change"] == cid
    with pytest.raises(ValueError):
        with riglog.change(tmp_path, "bench-a", "pi", "no undo", " "):
            pass


@pytest.fixture
def client(tmp_path):
    p, data = write_cfg(tmp_path, BENCH)
    app = server.create_app(p)
    yield TestClient(app, base_url="http://127.0.0.1:8000"), data
    app.state.close_jobs()


def test_rig_log_page(client, tmp_path):
    c, data = client
    page = c.get("/bench/rig-log").text
    assert "bench-a" in page and "pi, n6, ae3" in page and "2 field units listed" in page
    assert "bench-a-alias" not in page                 # the page does not show addresses
    assert "No changes on a real unit yet" in page
    FakeBoard(units.Bench().unit("fake"), "n6", data).load(model(tmp_path))
    with riglog.change(data, "bench-a", "pi", "copy model.tflite", "delete model.tflite"):
        pass
    with open(riglog.path(data), "a") as f:
        f.write(json.dumps({"change": "c9", "phase": "before", "unit": "bench-a", "board": "n6",
                            "what": "write ROMFS0", "undo": "write the backup's ROMFS0", "at": "t"}) + "\n")
    page = c.get("/bench/rig-log").text
    assert "copy model.tflite" in page and "load yolox_nano" not in page
    assert "1 change did not finish" in page and "1 change on the fake board" in page
    assert "load yolox_nano" in c.get("/bench/rig-log?show_fake=1").text
    assert 'href="/bench/rig-log"' in c.get("/").text
